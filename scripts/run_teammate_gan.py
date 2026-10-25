"""
팀원 GAN(loveda-semantic-editing)을 우리 평가용 조작 세트에 돌려서
조작 사진을 만들고 지표를 계산한다.

입력 형식이 같다: 가린 사진 3 + 의미지도 one-hot 7 + 구멍 1 = 11채널
  - 사진은 -1~1, 구멍 위치는 0
  - one-hot은 클래스 1~7 순서 (no-data는 모두 0)
결과는 팀원 평가 코드와 같은 방식으로 '구멍 안만 생성값, 밖은 원본'으로 합성한다.

출력 (OUT):
  images/{시나리오}/{patch_id}.png   최종 조작 사진 (PNG, SYNTHETIC 표시)
  raw/{시나리오}/{patch_id}.png      (--save_raw) 합성 전 생성기 출력 전체
  sheets/{시나리오}_{번호}.png        확인용 그림: 원본 | 편집 지도 | 구멍 | 결과 | 원래지도 복원
  results.csv                        사례별 지표
  summary.json                       시나리오별 평균

지표 (분할 모델 = 팀원의 semantic_audit.pt)
  intent_changed   바뀐 픽셀 중 결과 사진에서 '목표 클래스'로 인식된 비율  ← 의도 반영
  gone_changed     바뀐 픽셀 중 결과 사진에서 '원래 클래스가 아니게' 인식된 비율 ← 건물 삭제에서 핵심
                   (건물 삭제는 채움 클래스가 배경/농경지 등으로 다양해서, '건물이 사라졌나'가 더 공정한 기준)
  real_target      같은 픽셀이 원본 사진에서 이미 목표 클래스로 인식된 비율 (비교 기준, 낮아야 정상)
  real_label_agree 같은 픽셀에서 원본 사진 인식이 원래 라벨과 맞는 비율 (분할기 신뢰도 참고)
  recon_mae        원래 지도로 생성했을 때 구멍 안의 원본과의 평균 오차(0~255)  ← 복원 능력
  outside_max_err  구멍 밖 최대 변화량. 항상 0이어야 함

사용법:
  python run_teammate_gan.py --repo ../loveda-semantic-editing \
      --checkpoint ../loveda-semantic-editing/weights/semantic_04000.pt \
      --patches data/LoveDA_patches --manip data/LoveDA_manip \
      --out results/semantic_04000
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, PngImagePlugin
from tqdm import tqdm

from loveda_lib import colorize, load_patch, weights_dir


def overlay(img, hole, color=(255, 0, 255), alpha=0.5):
    out = img.astype(np.float32).copy()
    out[hole] = (1 - alpha) * out[hole] + alpha * np.array(color)
    return out.astype(np.uint8)


def to_rgb(t):
    return ((t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="팀원 저장소 경로 (model.py가 있는 곳)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="기본: REPO/weights/semantic_04000.pt")
    ap.add_argument("--audit", type=Path, default=None, help="기본: REPO/weights/semantic_audit.pt")
    ap.add_argument("--weights_dir", type=Path, default=None, help="팀원 가중치 폴더 (기본: REPO/weights). 테스트용")
    ap.add_argument("--no_audit", action="store_true")
    ap.add_argument("--patches", type=Path, required=True)
    ap.add_argument("--manip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="시나리오별 최대 개수 (빠른 점검용)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save_raw", action="store_true")
    ap.add_argument("--n_sheets", type=int, default=12, help="시나리오별 확인 그림 수")
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo.resolve()))
    from model import Generator, condition, normalize, onehot, segmentation  # 팀원 코드

    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    wdir = weights_dir(a.repo, a.weights_dir)
    ckpt = a.checkpoint or wdir / "semantic_04000.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    g = Generator().to(device).eval()
    g.load_state_dict(state["generator"], strict=True)
    step = state.get("step")
    del state
    audit = None
    if not a.no_audit:
        audit_path = a.audit or wdir / "semantic_audit.pt"
        audit = segmentation(audit_path, device)
    print(f"device={device} checkpoint={ckpt} step={step} audit={'on' if audit else 'off'}")

    with open(a.manip / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    if a.scenarios:
        rows = [r for r in rows if r["scenario"] in a.scenarios]
    if a.limit:
        per = defaultdict(int)
        keep = []
        for r in rows:
            if per[r["scenario"]] < a.limit:
                keep.append(r)
                per[r["scenario"]] += 1
        rows = keep
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sheets").mkdir(exist_ok=True)

    results, sheet_count = [], defaultdict(int)

    @torch.no_grad()
    def predict(rgb_uint8):
        x = torch.from_numpy(rgb_uint8).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
        return audit(normalize(x))["out"].argmax(1).cpu().numpy() + 1  # 클래스 1~7

    for s in tqdm(range(0, len(rows), a.batch), desc="generate"):
        batch = rows[s:s + a.batch]
        imgs, origs, edits, holes = [], [], [], []
        for r in batch:
            img, m = load_patch(a.patches, r["split"], r["patch_id"])
            e = np.array(Image.open(a.manip / r["scenario"] / "edited_masks" / f"{r['patch_id']}.png"))
            h = np.array(Image.open(a.manip / r["scenario"] / "holes" / f"{r['patch_id']}.png")) > 0
            imgs.append(img); origs.append(m); edits.append(e); holes.append(h)
        img_t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
        hole_t = torch.from_numpy(np.stack(holes).astype(np.float32))[:, None].to(device)
        with torch.no_grad():
            sem_e = onehot(torch.from_numpy(np.stack(edits).astype(np.int64)).to(device))
            sem_o = onehot(torch.from_numpy(np.stack(origs).astype(np.int64)).to(device))
            raw_e = to_rgb(g(condition(img_t, sem_e, hole_t)))
            raw_o = to_rgb(g(condition(img_t, sem_o, hole_t)))
        finals, recons = [], []
        for i in range(len(batch)):
            f_ = imgs[i].copy(); f_[holes[i]] = raw_e[i][holes[i]]
            o_ = imgs[i].copy(); o_[holes[i]] = raw_o[i][holes[i]]
            finals.append(f_); recons.append(o_)
        pred_f = predict(np.stack(finals)) if audit else None
        pred_r = predict(np.stack(imgs)) if audit else None

        for i, r in enumerate(batch):
            sc, pid = r["scenario"], r["patch_id"]
            img, m, e, h, fin = imgs[i], origs[i], edits[i], holes[i], finals[i]
            changed = (m != e) & (e != 0)
            outside = int(np.abs(fin.astype(np.int16) - img.astype(np.int16))[~h].max()) if (~h).any() else 0
            if outside:
                raise AssertionError(f"{pid}: 구멍 밖이 바뀜")
            folder = a.out / "images" / sc
            folder.mkdir(parents=True, exist_ok=True)
            info = PngImagePlugin.PngInfo()
            info.add_text("SYNTHETIC", "true")
            info.add_text("configuration", json.dumps(dict(scenario=sc, patch_id=pid, checkpoint=str(ckpt.name), step=step)))
            Image.fromarray(fin).save(folder / f"{pid}.png", pnginfo=info)
            if a.save_raw:
                (a.out / "raw" / sc).mkdir(parents=True, exist_ok=True)
                Image.fromarray(raw_e[i]).save(a.out / "raw" / sc / f"{pid}.png", pnginfo=info)
            row = dict(scenario=sc, patch_id=pid, domain=r["domain"], hole_ratio=r["hole_ratio"],
                       changed_px=int(changed.sum()), outside_max_err=outside,
                       recon_mae=round(float(np.abs(recons[i].astype(float) - img.astype(float))[h].mean()), 3))
            if audit is not None and changed.any():
                row["intent_changed"] = round(float((pred_f[i][changed] == e[changed]).mean()), 4)
                row["gone_changed"] = round(float((pred_f[i][changed] != m[changed]).mean()), 4)
                row["real_target"] = round(float((pred_r[i][changed] == e[changed]).mean()), 4)
                row["real_label_agree"] = round(float((pred_r[i][changed] == m[changed]).mean()), 4)
            results.append(row)

            if sheet_count[sc] < a.n_sheets:
                panels = [("real", img), ("edited map", colorize(e)), ("hole = GT", overlay(img, h)),
                          ("result (edited map)", fin), ("recon (original map)", recons[i])]
                canvas = Image.new("RGB", (256 * 5, 280), "white")
                d = ImageDraw.Draw(canvas)
                for c, (t, p) in enumerate(panels):
                    d.text((c * 256 + 4, 4), t, fill="black")
                    canvas.paste(Image.fromarray(p), (c * 256, 24))
                d.text((4, 266), f"{sc} / {pid} / intent={row.get('intent_changed', '-')}", fill="black")
                canvas.save(a.out / "sheets" / f"{sc}_{sheet_count[sc]:02d}.png")
                sheet_count[sc] += 1

    keys = list(dict.fromkeys(k for r in results for k in r))
    with open(a.out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    summary = {}
    for sc in sorted({r["scenario"] for r in results}):
        rs = [r for r in results if r["scenario"] == sc]
        summary[sc] = {"n": len(rs), "recon_mae": round(float(np.mean([r["recon_mae"] for r in rs])), 3)}
        for k in ["intent_changed", "gone_changed", "real_target", "real_label_agree"]:
            v = [r[k] for r in rs if k in r]
            if v:
                summary[sc][k] = round(float(np.mean(v)), 4)
    summary["_meta"] = dict(checkpoint=str(ckpt), step=step, audit=bool(audit),
                            note="intent_changed는 분할기 기준 보조 지표. 자연스러움은 그림과 KID로 따로 판단.")
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n시나리오별 평균")
    print(f"{'scenario':20s} {'n':>5s} {'intent':>8s} {'gone':>7s} {'real_tgt':>9s} {'real_ok':>8s} {'recon':>7s}")
    for sc, v in summary.items():
        if sc.startswith("_"):
            continue
        print(f"{sc:20s} {v['n']:5d} {v.get('intent_changed', float('nan')):8.3f} {v.get('gone_changed', float('nan')):7.3f} "
              f"{v.get('real_target', float('nan')):9.3f} {v.get('real_label_agree', float('nan')):8.3f} {v['recon_mae']:7.2f}")
    print("저장:", a.out)


if __name__ == "__main__":
    main()
