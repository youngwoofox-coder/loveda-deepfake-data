"""
팀원 GAN(loveda-semantic-editing)의 모델·손실을 그대로 쓰고, 데이터 쪽만 우리 파이프라인으로 바꾼 본학습 스크립트.

팀원 train.py와 같은 것
  - Generator / Discriminator 구조, generator_loss (L1·경계·적대·특징매칭·지각·질감·그래디언트·의미 CE)
  - 판별기 손실, Adam(betas .5,.999), 기울기 자르기 10, AMP 초기 scale 128
  - 체크포인트 형식 {'generator','discriminator',...,'step'} → run_teammate_gan.py, 팀원 evaluate.py 모두 사용 가능
다른 것
  - 데이터: 미리 자른 256 패치(train split = 팀원 학습 목록 2,490장과 동일)를 DataLoader 여러 프로세스로 읽음 → 빠름
  - 구멍: 우리 sample_train_hole (물체 통째 / 경계 띠 / 붓질) + 선택적으로 패치 전체 가리기 비율(--full_ratio)
  - --init none 이면 처음부터 학습, 기본은 semantic_04000.pt에서 이어서 학습
  - 주기적 검증: val split 고정 구멍 복원 오차 + 분할기 의미 일치 + 실제 조작(물 확장) 의도 반영률

출력 (OUT):
  checkpoints/latest.pt        재개용 (G, D, optimizer, scaler, step)
  checkpoints/step_XXXXXX.pt   검증 시점 스냅샷 (G, D, step)
  checkpoints/best.pt          검증 '조작 의도 반영률'이 가장 높았던 시점
  samples/step_XXXXXX.png      검증 예시 그림 (구멍 낸 입력 | 결과 | 원본)
  train_log.csv / val_log.csv

사용법 (백그라운드):
  CUDA_VISIBLE_DEVICES=3 nohup python train_full.py --repo ../loveda-semantic-editing \
     --patches data/LoveDA_patches --out runs/ft_ourholes \
     --steps 20000 --batch 16 > logs/ft_ourholes.log 2>&1 &
  중단 후 이어서: 같은 명령에 --resume
"""
import argparse
import collections
import csv
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from loveda_lib import SCENARIOS, LoveDAInpaintDataset, SeededDrawSampler, load_patch, resolve_init, weights_dir


def to_uint8(t):
    return ((t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()


def append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--patches", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", default="weights/semantic_04000.pt", help="'none'이면 처음부터. 상대경로는 가중치 폴더·REPO 기준")
    ap.add_argument("--weights_dir", type=Path, default=None, help="팀원 가중치 폴더 (기본: REPO/weights). 테스트용")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--full_ratio", type=float, default=0.0, help="패치 전체를 가리는 샘플 비율 (팀원 혼합 버전은 0.2)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--val_n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    # 집중 학습 옵션
    ap.add_argument("--focus", choices=["none", "building", "water"], default="none",
                    help="물체 구멍을 이 클래스 위주로 뽑음")
    ap.add_argument("--focus_weight", type=float, default=8.0)
    ap.add_argument("--hole_probs", default="0.4,0.4,0.2",
                    help="물체,경계띠,붓질[,조작모양] 구멍 비율. 4번째(조작 모양)를 주면 실제 조작과 같은 모양의 구멍을 섞음")
    ap.add_argument("--focus_weights", default=None,
                    help="물체 구멍의 클래스별 가중치 '클래스:가중치,...' (예: '2:4,1:2' = 건물 4배, 배경 2배). --focus보다 우선")
    ap.add_argument("--require", default=None, help="이 클래스가 일정 비율 이상인 패치만 학습 (예: building)")
    ap.add_argument("--require_min", type=float, default=0.05)
    ap.add_argument("--probe", default="water_expand_small",
                    help="검증 때 실제로 해 볼 조작 (water_expand_small / building_add / building_remove ...)")
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo.resolve()))
    from model import (Discriminator, FrozenFeatures, Generator, condition, generator_loss,  # 팀원 코드
                       normalize, onehot, region_mean, segmentation)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not a.no_amp) and dev.type == "cuda"
    out = a.out
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    cfg_path = out / "config.json"
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k != "resume"}
    if cfg_path.exists() and not a.resume:
        raise SystemExit(f"[error] {out} 에 이미 실험이 있어요. 이어서 하려면 --resume, 새로 하려면 다른 --out")
    if cfg_path.exists() and a.resume:
        old = json.loads(cfg_path.read_text())
        diff = {k for k in cfg if k not in ("workers", "val_every", "save_every", "steps") and old.get(k) != cfg[k]}
        if diff:
            raise SystemExit(f"[error] 설정이 이전과 달라요: {sorted(diff)}")
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

    # ---------------- 모델
    g, d = Generator().to(dev), Discriminator().to(dev)
    wdir = weights_dir(a.repo, a.weights_dir)
    if a.init.lower() != "none":
        ip = resolve_init(a.init, a.repo, wdir)
        st = torch.load(ip, map_location="cpu", weights_only=False)
        g.load_state_dict(st["generator"], strict=True)
        if "discriminator" in st:
            d.load_state_dict(st["discriminator"], strict=True)
        del st
        print(f"[init] {ip}")
    else:
        print("[init] 처음부터 학습")
    og = torch.optim.Adam(g.parameters(), lr=a.lr, betas=(.5, .999))
    od = torch.optim.Adam(d.parameters(), lr=a.lr, betas=(.5, .999))
    sg = torch.amp.GradScaler("cuda", enabled=amp, init_scale=128)
    sd = torch.amp.GradScaler("cuda", enabled=amp, init_scale=128)
    teacher = segmentation(wdir / "semantic_teacher.pt", dev)
    features = FrozenFeatures(wdir / "squeezenet_features.pt").to(dev).eval()
    audit_path = wdir / "semantic_audit.pt"
    audit = segmentation(audit_path, dev) if audit_path.exists() else None

    step, best = 0, -1.0
    latest = out / "checkpoints" / "latest.pt"
    if a.resume and latest.exists():
        st = torch.load(latest, map_location="cpu", weights_only=False)
        g.load_state_dict(st["generator"]); d.load_state_dict(st["discriminator"])
        og.load_state_dict(st["optimizer_g"]); od.load_state_dict(st["optimizer_d"])
        sg.load_state_dict(st["scaler_g"]); sd.load_state_dict(st["scaler_d"])
        step, best = st["step"], st.get("best", -1.0)
        del st
        print(f"[resume] step {step}부터")

    def save(path, full=True):
        row = dict(generator=g.state_dict(), discriminator=d.state_dict(), step=step, best=best,
                   input_channels=11, config=cfg)
        if full:
            row.update(optimizer_g=og.state_dict(), optimizer_d=od.state_dict(),
                       scaler_g=sg.state_dict(), scaler_d=sd.state_dict())
        tmp = path.with_suffix(".part")
        torch.save(row, tmp)
        tmp.replace(path)

    # ---------------- 데이터
    focus_k = {"building": 2, "water": 4}.get(a.focus)
    hole_kwargs = dict(probs=tuple(float(x) for x in a.hole_probs.split(",")))
    if a.focus_weights:
        hole_kwargs["class_weights"] = {int(k): float(v) for k, v in
                                        (kv.split(":") for kv in a.focus_weights.split(","))}
    elif focus_k:
        hole_kwargs["class_weights"] = {focus_k: a.focus_weight}
    ids = None
    if a.require:
        with open(a.patches / "index.csv") as f:
            ids = sorted(r["patch_id"] for r in csv.DictReader(f)
                         if r["split"] == "train" and float(r[a.require]) >= a.require_min)
        print(f"[data] {a.require} >= {a.require_min} 인 학습 패치만 사용: {len(ids)}개")
    # 같은 seed면 같은 패치 순서·회전·구멍 (full_ratio만 다른 실험은 일부 샘플만 전체 가리기로 바뀜)
    train_ds = LoveDAInpaintDataset(a.patches, "train", augment=True, ids=ids, hole_kwargs=hole_kwargs,
                                    seed=a.seed, full_ratio=a.full_ratio)
    sampler = SeededDrawSampler(len(train_ds), a.seed, start_draw=step * a.batch)  # 재개하면 끊긴 지점부터
    loader = DataLoader(train_ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                        pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    val_all = LoveDAInpaintDataset(a.patches, "val", augment=False, fixed_holes=True, seed=12345)
    rng = np.random.default_rng(0)
    val_ids = [val_all.ids[i] for i in sorted(rng.choice(len(val_all.ids), min(a.val_n, len(val_all.ids)), replace=False))]
    val_ds = LoveDAInpaintDataset(a.patches, "val", augment=False, fixed_holes=True, seed=12345, ids=val_ids)
    # 실제 조작 검증용: val 패치에 --probe 규칙 적용, 고정
    probe, rule = [], SCENARIOS[a.probe]
    for pid in val_all.ids:
        if len(probe) >= 64:
            break
        img, m = load_patch(a.patches, "val", pid)
        res = rule(m, rng=np.random.default_rng(len(probe)))
        if res is not None:
            probe.append((img, m, res[0], res[1]))
    removal = a.probe.endswith("_remove")  # 삭제는 '원래 클래스가 사라졌나'로 판정
    print(f"train {len(train_ds)}개 패치, val {len(val_ds)}개, 조작 검증 {len(probe)}개, device={dev}, amp={amp}")

    stop = {"flag": False}
    for sgn in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sgn, lambda *_: stop.update(flag=True))
    if (out / "STOP").exists():
        raise SystemExit("STOP 파일이 있어요. 확인 후 지우고 다시 실행하세요.")

    @torch.no_grad()
    def validate():
        g.eval()
        l1s, agree = [], []
        sample = None
        vl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=min(4, a.workers))
        for b in vl:
            img = b["target"].to(dev); lab = b["mask"].to(dev); hole = b["hole"].to(dev)
            raw = g(condition(img, onehot(lab), hole))
            fake = img * (1 - hole) + raw * hole
            h = hole[:, 0] > 0
            l1s.append(((fake - img).abs().mean(1) * 127.5)[h].mean().item())
            if audit is not None:
                pred = audit(normalize(fake))["out"].argmax(1) + 1
                valid = h & (lab > 0)
                agree.append((pred[valid] == lab[valid]).float().mean().item())
            if sample is None:
                sample = (to_uint8(img[:6] * (1 - hole[:6])), to_uint8(fake[:6]), to_uint8(img[:6]))
        intent = float("nan")
        if probe and audit is not None:
            hits = []
            for i in range(0, len(probe), 16):
                chunk = probe[i:i + 16]
                img = torch.from_numpy(np.stack([c[0] for c in chunk])).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
                ed = torch.from_numpy(np.stack([c[2] for c in chunk]).astype(np.int64)).to(dev)
                hole = torch.from_numpy(np.stack([c[3] for c in chunk]).astype(np.float32))[:, None].to(dev)
                raw = g(condition(img, onehot(ed), hole))
                fake = img * (1 - hole) + raw * hole
                pred = audit(normalize(fake))["out"].argmax(1) + 1
                for j, c in enumerate(chunk):
                    ch = torch.from_numpy((c[1] != c[2]) & (c[2] != 0)).to(dev)
                    if ch.any():
                        orig = torch.from_numpy(c[1].astype(np.int64)).to(dev)
                        ok = (pred[j][ch] != orig[ch]) if removal else (pred[j][ch] == ed[j][ch])
                        hits.append(ok.float().mean().item())
            intent = float(np.mean(hits)) if hits else float("nan")
        g.train()
        if sample is not None:
            rows = [np.concatenate(list(col), axis=0) for col in sample]
            Image.fromarray(np.concatenate(rows, axis=1)).save(out / "samples" / f"step_{step:06d}.png")
        return dict(step=step, val_hole_l1=round(float(np.mean(l1s)), 3),
                    val_sem_agree=round(float(np.mean(agree)), 4) if agree else "",
                    edit_intent=round(intent, 4) if intent == intent else "")

    if step == 0:
        v = validate(); append_csv(out / "val_log.csv", v); print("[val]", v)

    g.train(); d.train()
    t0, it, start_step = time.time(), iter(loader), step
    kinds = collections.Counter()
    while step < a.steps and not stop["flag"] and not (out / "STOP").exists():
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        img = b["target"].to(dev, non_blocking=True)
        lab = b["mask"].to(dev, non_blocking=True)
        hole = b["hole"].to(dev, non_blocking=True)
        kinds.update(b["kind"])  # 패치 전체 가리기는 데이터셋에서 이미 반영됨 (kind == "full")
        cond = condition(img, onehot(lab), hole)
        step += 1

        d.requires_grad_(True); od.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            raw = g(cond); fake = img * (1 - hole) + raw * hole
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            support = hole * (lab != 0)[:, None]
            dl = .5 * (region_mean((d(torch.cat([cond, img], 1)).float() - 1).square(), support)
                       + region_mean(d(torch.cat([cond, fake], 1)).float().square(), support))
        if not torch.isfinite(dl):
            raise FloatingPointError(f"step {step}: 판별기 손실이 발산")
        sd.scale(dl).backward(); sd.unscale_(od); torch.nn.utils.clip_grad_norm_(d.parameters(), 10)
        sd.step(od); sd.update()

        d.requires_grad_(False); og.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            gl, metrics = generator_loss(g, d, teacher, features, img, lab, hole, step)
        if not torch.isfinite(gl):
            raise FloatingPointError(f"step {step}: 생성기 손실이 발산")
        sg.scale(gl).backward(); sg.unscale_(og); torch.nn.utils.clip_grad_norm_(g.parameters(), 10)
        sg.step(og); sg.update()

        if step == 1 or step % 50 == 0:
            el = time.time() - t0
            sps = el / max(step - start_step, 1)
            row = dict(step=step, g_loss=round(gl.item(), 4), d_loss=round(dl.item(), 4),
                       l1=round(metrics["l1"].item(), 4), adv=round(metrics["adversarial"].item(), 4),
                       sem_ce=round(metrics["semantic_ce"].item(), 4),
                       sec_per_step=round(sps, 3),
                       gpu_gib=round(torch.cuda.max_memory_allocated(dev) / 1024 ** 3, 2) if dev.type == "cuda" else "",
                       **{f"n_{k}": kinds.get(k, 0) for k in ("object", "boundary", "free", "scenario", "full", "rect")})
            kinds.clear()  # 직전 기록 이후 구멍 종류별 개수
            append_csv(out / "train_log.csv", row)
            print(f"[train] step {step}/{a.steps}  G {row['g_loss']}  D {row['d_loss']}  L1 {row['l1']}  "
                  f"semCE {row['sem_ce']}  {sps:.2f}s/step  남은 시간 약 {(a.steps - step) * sps / 60:.0f}분", flush=True)
        if step % a.save_every == 0:
            save(latest)
        if step % a.val_every == 0 or step == a.steps:
            v = validate(); append_csv(out / "val_log.csv", v); print("[val]", v, flush=True)
            save(out / "checkpoints" / f"step_{step:06d}.pt", full=False)
            if isinstance(v["edit_intent"], float) and v["edit_intent"] > best:
                best = v["edit_intent"]
                save(out / "checkpoints" / "best.pt", full=False)
                print(f"[best] step {step} 조작 의도 반영률 {best}", flush=True)
            save(latest)
    save(latest)
    print("STOPPED_SAFELY" if step < a.steps else "TRAINING_COMPLETE", "step", step, flush=True)


if __name__ == "__main__":
    main()
