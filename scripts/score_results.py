"""
조작 결과 사진 채점 (생성 방법과 무관: 제안 GAN, 조건 없는 GAN, 복사-붙여넣기 등 모두 같은 채점)

입력: RESULT/images/{시나리오}/{patch_id}.png (최종 조작 사진), 조작 세트(manifest, 편집 지도, 구멍), 원본 패치
분할 모델: 팀원의 semantic_audit.pt (학습 손실의 teacher와 다른 가중치)

영역 (평가 방식 초안과 같음)
  E = 원래 지도와 편집 지도가 다른 픽셀 (편집 목표)
  M = 구멍 (모델이 다시 그린 곳 = 조작 위치 정답)
  C = M 안에서 E가 아닌 곳 (다시 그렸지만 의미는 그대로여야 하는 띠)
  O = 구멍 밖 (원본 그대로여야 함)
  pred = 결과 사진의 분할 예측, pred_r = 원본 사진의 예측, tgt = 편집 지도

사례별 지표
  [기존과 같은 이름] intent_changed  E에서 pred == tgt (목표 클래스로 보이는 비율)
                     gone_changed    E에서 pred != 원래 클래스
                     real_target     E에서 pred_r == tgt (조작 전에도 이미 목표로 보이던 비율, 기준선)
                     real_label_agree E에서 pred_r == 원래 라벨 (분할기 정확도 참고)
  [building_add]     add_recall      E에서 pred == 건물 (= intent_changed)
                     add_precision_M M에서 pred == 건물인 픽셀 중 tgt == 건물인 비율
                     add_iou_M       M에서 (pred == 건물)과 (tgt == 건물)의 IoU
                     overgen_C       C에서 tgt != 건물인 픽셀 중 pred == 건물 비율 (목표 밖으로 넘친 건물)
  [building_remove]  remove_residual E에서 pred == 건물 (남은 건물, 낮을수록 좋음)
                     fill_match      E에서 pred == tgt (지정한 채움 클래스로 보이는 비율)
                     keep_buildings  E 밖에서 tgt == 건물인 픽셀 중 pred == 건물 (남겨야 할 건물 유지). 없으면 빈칸
  [공통]             c_sem_keep      C에서 pred == tgt
                     c_rgb_mae       C에서 결과와 원본의 평균 절대 차이 (0~255). 그림자·경계 연결 때문에 0일 필요는 없음
                     outside_max_err O에서 최대 차이. 0이 아니면 합성 오류로 중단
  real_* 이 붙은 지표는 같은 계산을 원본 사진 예측으로 한 기준선

선명도 (팀원 제안: 라플라시안 분산). 회색조 사진에 라플라시안 필터를 걸고 영역 안 값의 분산을 본다. 흐릴수록 작다.
  sharp_E          결과 사진의 E 영역 라플라시안 분산
  real_sharp_E     원본 사진의 같은 영역 (조작 전. 건물 추가면 원래 밭·빈 땅의 선명도)
  sharp_ref        같은 원본 사진에서 구멍 밖의, 목표 클래스(추가 = 건물, 삭제 = 채움 클래스) 진짜 픽셀의 분산
  sharp_ratio      sharp_E / sharp_ref. 1에 가까우면 같은 사진의 진짜 지물만큼 선명, 1보다 많이 작으면 흐림
  주의: 노이즈·격자 무늬 같은 생성 잡음도 분산을 키우므로, 값이 높다고 무조건 좋은 것은 아니다. 그림과 같이 볼 것.
  요약에는 평균과 함께 중앙값(_median)도 넣는다 (작은 영역은 값이 크게 튀기 때문)

크기 구간: E의 연결 요소마다 side = sqrt(면적)으로 small(20px 미만) / mid(20~50) / large(50 이상).
  요소별 add_recall 또는 remove_residual 을 results_components.csv에, 구간별 평균을 summary.json에 저장

출력 (OUT): results.csv, results_components.csv, summary.json (시나리오별, 도메인별, 크기 구간별 평균)

사용법:
  python score_results.py --repo ../loveda-semantic-editing --results results/copy_paste \
      --patches data/LoveDA_patches --manip data/LoveDA_manip
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from loveda_lib import BUILDING, NODATA, load_patch, weights_dir

SIZE_BINS = (("small", 0, 20), ("mid", 20, 50), ("large", 50, 1e9))
SUMMARY_KEYS = ["intent_changed", "gone_changed", "real_target", "real_label_agree",
                "add_recall", "add_precision_M", "add_iou_M", "overgen_C",
                "remove_residual", "fill_match", "keep_buildings",
                "c_sem_keep", "c_rgb_mae",
                "real_add_precision_M", "real_overgen_C", "real_keep_buildings", "real_c_sem_keep",
                "recon_mae", "sharp_E", "real_sharp_E", "sharp_ref", "sharp_ratio"]
MEDIAN_KEYS = ["sharp_E", "real_sharp_E", "sharp_ref", "sharp_ratio"]
CARRY_KEYS = ("recon_mae", "donor_coverage")  # 다시 채점할 때 이전 results.csv에서 가져올 값 (생성 쪽에서만 알 수 있음)


def lap_var(lap, region, min_px=16):
    return round(float(lap[region].var()), 2) if int(region.sum()) >= min_px else ""


def sharpness(final, img, orig, edited, hole, E):
    """라플라시안 분산 기반 선명도 (위 설명 참고)"""
    lap_f = ndi.laplace(final.astype(np.float32).mean(axis=2))
    lap_r = ndi.laplace(img.astype(np.float32).mean(axis=2))
    targets = np.unique(edited[E]) if E.any() else []
    ref = ~hole & np.isin(orig, targets) & (orig != NODATA)
    out = {"sharp_E": lap_var(lap_f, E), "real_sharp_E": lap_var(lap_r, E), "sharp_ref": lap_var(lap_r, ref, 64)}
    out["sharp_ratio"] = (round(out["sharp_E"] / out["sharp_ref"], 4)
                          if out["sharp_E"] != "" and out["sharp_ref"] not in ("", 0, 0.0) else "")
    return out


def frac(num_mask, den_mask):
    d = int(den_mask.sum())
    return round(float((num_mask & den_mask).sum()) / d, 4) if d else ""


def size_bin(area):
    side = float(np.sqrt(area))
    return next(name for name, lo, hi in SIZE_BINS if lo <= side < hi)


def case_metrics(scenario, orig, edited, hole, final, img, pred, pred_r):
    """사례 하나의 지표 dict와 연결 요소 목록. pred/pred_r 이 None이면 분할 지표는 빼고 계산"""
    E = (orig != edited) & (edited != NODATA)
    M = hole & (orig != NODATA)
    C = M & ~E
    O = ~hole
    outside = int(np.abs(final.astype(np.int16) - img.astype(np.int16))[O].max()) if O.any() else 0
    row = {"changed_px": int(E.sum()), "outside_max_err": outside,
           "c_rgb_mae": round(float(np.abs(final.astype(float) - img.astype(float))[C].mean()), 3) if C.any() else "",
           **sharpness(final, img, orig, edited, hole, E)}
    comps = []
    if pred is None:
        return row, comps
    tb = edited == BUILDING
    for tag, p in (("", pred), ("real_", pred_r)):
        pb = p == BUILDING
        if tag == "":
            row["intent_changed"] = frac(p == edited, E)
            row["gone_changed"] = frac(p != orig, E)
        else:
            row["real_target"] = frac(p == edited, E)
            row["real_label_agree"] = frac(p == orig, E)
        row[tag + "c_sem_keep"] = frac(p == edited, C)
        if scenario == "building_add":
            if tag == "":
                row["add_recall"] = frac(pb, E)
                inter, union = int((pb & tb & M).sum()), int(((pb | tb) & M).sum())
                row["add_iou_M"] = round(inter / union, 4) if union else ""
            row[tag + "add_precision_M"] = frac(tb, pb & M)
            row[tag + "overgen_C"] = frac(pb, C & ~tb)
        elif scenario == "building_remove":
            if tag == "":
                row["remove_residual"] = frac(pb, E)
                row["fill_match"] = frac(p == edited, E)
            row[tag + "keep_buildings"] = frac(pb, ~E & tb)
    if scenario in ("building_add", "building_remove"):
        lab, n = ndi.label(E)
        for k in range(1, n + 1):
            comp = lab == k
            area = int(comp.sum())
            val = float((pred[comp] == BUILDING).mean())
            comps.append({"comp": k, "area": area, "side": round(float(np.sqrt(area)), 1), "size_bin": size_bin(area),
                          ("add_recall" if scenario == "building_add" else "remove_residual"): round(val, 4)})
    return row, comps


def mean_of(rows, keys):
    out = {}
    for k in keys:
        v = [float(r[k]) for r in rows if r.get(k, "") != ""]
        if v:
            out[k] = round(float(np.mean(v)), 4)
            if k in MEDIAN_KEYS:
                out[k + "_median"] = round(float(np.median(v)), 4)
    return out


def summarize(results, components, meta):
    summary = {}
    for sc in sorted({r["scenario"] for r in results}):
        rs = [r for r in results if r["scenario"] == sc]
        s = {"n": len(rs), **mean_of(rs, SUMMARY_KEYS)}
        s["by_domain"] = {d: {"n": len(x), **mean_of(x, SUMMARY_KEYS)}
                          for d in sorted({r["domain"] for r in rs}) for x in [[r for r in rs if r["domain"] == d]]}
        cs = [c for c in components if c["scenario"] == sc]
        key = "add_recall" if sc == "building_add" else "remove_residual"
        if cs:
            s["by_size"] = {b: {"n_components": len(x), **mean_of(x, [key])}
                            for b, _, _ in SIZE_BINS for x in [[c for c in cs if c["size_bin"] == b]] if x}
        summary[sc] = s
    summary["_meta"] = meta
    return summary


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_audit(repo, wdir, audit_path, device):
    if str(Path(repo).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(repo).resolve()))
    from model import segmentation  # 팀원 코드
    return segmentation(audit_path or Path(wdir) / "semantic_audit.pt", device)


def make_predict(audit, device):
    import torch
    sys_model = sys.modules["model"]

    @torch.no_grad()
    def predict(rgb_uint8):
        x = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
        return audit(sys_model.normalize(x))["out"].argmax(1).cpu().numpy() + 1  # 클래스 1~7
    return predict


def score(results_dir, manip, patches, predict=None, extra=None, meta=None, batch=16, rows=None, progress=True):
    """results_dir/images/{시나리오}/{pid}.png 를 채점해 results_dir에 results.csv, results_components.csv, summary.json 저장.
    extra: {(시나리오, pid): {"recon_mae": ...}} 처럼 생성 쪽에서만 알 수 있는 값을 합칠 때 사용"""
    results_dir, manip, patches = Path(results_dir), Path(manip), Path(patches)
    if rows is None:
        with open(manip / "manifest.csv") as f:
            rows = [r for r in csv.DictReader(f) if (results_dir / "images" / r["scenario"] / f"{r['patch_id']}.png").exists()]
    if not rows:
        raise SystemExit(f"[error] {results_dir / 'images'} 에 채점할 결과가 없어요")
    if extra is None and (results_dir / "results.csv").exists():  # 다시 채점: 생성 쪽 값(recon_mae 등)은 이전 결과에서
        with open(results_dir / "results.csv") as f:
            extra = {(r["scenario"], r["patch_id"]): {k: float(r[k]) for k in CARRY_KEYS if r.get(k, "") != ""}
                     for r in csv.DictReader(f)}
    extra = extra or {}
    results, components = [], []
    it = range(0, len(rows), batch)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc="score")
    for s in it:
        chunk = rows[s:s + batch]
        data = []
        for r in chunk:
            img, m = load_patch(patches, r["split"], r["patch_id"])
            e = np.array(Image.open(manip / r["scenario"] / "edited_masks" / f"{r['patch_id']}.png"))
            h = np.array(Image.open(manip / r["scenario"] / "holes" / f"{r['patch_id']}.png")) > 0
            fin = np.array(Image.open(results_dir / "images" / r["scenario"] / f"{r['patch_id']}.png").convert("RGB"))
            data.append((r, img, m, e, h, fin))
        pf = predict(np.stack([d[5] for d in data])) if predict else [None] * len(data)
        pr = predict(np.stack([d[1] for d in data])) if predict else [None] * len(data)
        for (r, img, m, e, h, fin), p, q in zip(data, pf, pr):
            met, comps = case_metrics(r["scenario"], m, e, h, fin, img, p, q)
            if met["outside_max_err"]:
                raise AssertionError(f"{r['scenario']}/{r['patch_id']}: 구멍 밖이 원본과 달라요 (최대 {met['outside_max_err']})")
            base = dict(scenario=r["scenario"], patch_id=r["patch_id"], domain=r["domain"], hole_ratio=r["hole_ratio"])
            results.append({**base, **met, **extra.get((r["scenario"], r["patch_id"]), {})})
            components += [{"scenario": r["scenario"], "patch_id": r["patch_id"], "domain": r["domain"], **c} for c in comps]
    write_csv(results_dir / "results.csv", results)
    write_csv(results_dir / "results_components.csv", components)
    summary = summarize(results, components, dict(meta or {}, audit=bool(predict),
                        note="분할기 기준 지표는 보조 지표. 자연스러움은 그림·사람 평가·KID로 따로 판단."))
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def print_summary(summary):
    print("\n시나리오별 평균")
    print(f"{'scenario':16s} {'n':>5s} {'intent':>7s} {'gone':>6s} {'prec_M':>7s} {'overgen':>8s} "
          f"{'keepB':>6s} {'real_tgt':>9s} {'recon':>7s} {'sharp':>6s}")
    for sc, v in summary.items():
        if sc.startswith("_"):
            continue
        g = lambda k, d=3: f"{v[k]:.{d}f}" if k in v else "-"
        print(f"{sc:16s} {v['n']:5d} {g('intent_changed'):>7s} {g('gone_changed'):>6s} {g('add_precision_M'):>7s} "
              f"{g('overgen_C'):>8s} {g('keep_buildings'):>6s} {g('real_target'):>9s} {g('recon_mae', 2):>7s} "
              f"{g('sharp_ratio_median', 2):>6s}")


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="팀원 저장소 (분할 모델 코드)")
    ap.add_argument("--weights_dir", type=Path, default=None)
    ap.add_argument("--audit", type=Path, default=None, help="기본: 가중치 폴더/semantic_audit.pt")
    ap.add_argument("--no_audit", action="store_true")
    ap.add_argument("--results", type=Path, required=True, help="결과 폴더 (images/ 가 있는 곳)")
    ap.add_argument("--manip", type=Path, required=True)
    ap.add_argument("--patches", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    predict = None
    if not a.no_audit:
        predict = make_predict(load_audit(a.repo, weights_dir(a.repo, a.weights_dir), a.audit, device), device)
    old = a.results / "summary.json"
    meta = json.loads(old.read_text()).get("_meta", {}) if old.exists() else {}
    meta.pop("note", None)
    summary = score(a.results, a.manip, a.patches, predict, batch=a.batch, meta=dict(meta, rescored_from=str(a.results)))
    print_summary(summary)
    print("저장:", a.results)


if __name__ == "__main__":
    main()
