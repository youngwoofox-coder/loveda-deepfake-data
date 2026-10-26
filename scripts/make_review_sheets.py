"""
사람 비교용 그림: 같은 조작 사례를 여러 모델 결과로 나란히, 모델 이름을 가리고 순서를 섞어서 보여 준다.
점수(분할기)와 눈으로 본 품질이 다를 때, 사람이 직접 비교하기 위함.

입력: 이름=결과폴더 여러 개 (각 폴더에 images/{시나리오}/{patch_id}.png), 조작 세트, 패치
출력 (OUT):
  sheets/{번호}_{시나리오}.png   [원본 | 편집 지도 | 결과 1 | 결과 2 | ...]  위: 전체, 아래: 편집 부분 확대
  key.csv                        번호별로 결과 1, 2, ...가 어느 모델인지 (판정이 끝난 뒤에 열어 볼 것)
  review.csv                     사람이 채우는 표: 결과별 '건물로 보이나(추가) / 건물이 사라졌나(삭제)'와 '자연스러운가' 1~5점

사용법:
  python make_review_sheets.py --results start=results_val/start A=results_val/screen_A_base E0=results_val/screen_E_ce0 \
      --manip data/LoveDA_manip_val --patches data/LoveDA_patches --out review/round1 --n 10
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from loveda_lib import NODATA, colorize, load_patch


def focus_box(E, size=256, min_side=64):
    ys, xs = np.nonzero(E)
    if not len(ys):
        return 0, 0, size
    side = int(min(size, max(min_side, 2 * max(np.ptp(ys) + 1, np.ptp(xs) + 1))))
    cy, cx = (ys.min() + ys.max() + 1) // 2, (xs.min() + xs.max() + 1) // 2
    y = int(np.clip(cy - side // 2, 0, size - side))
    x = int(np.clip(cx - side // 2, 0, size - side))
    return y, x, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="이름=결과폴더 (2개 이상)")
    ap.add_argument("--manip", type=Path, required=True)
    ap.add_argument("--patches", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10, help="시나리오별 사례 수")
    ap.add_argument("--scenarios", nargs="+", default=["building_add", "building_remove"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    runs = []
    for item in a.results:
        name, path = item.split("=", 1)
        runs.append((name, Path(path)))
    if len(runs) < 2:
        raise SystemExit("[error] 비교하려면 결과 폴더가 2개 이상 필요해요")

    with open(a.manip / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(a.seed)
    chosen = []
    for sc in a.scenarios:
        cand = [r for r in rows if r["scenario"] == sc
                and all((p / "images" / sc / f"{r['patch_id']}.png").exists() for _, p in runs)]
        idx = sorted(rng.choice(len(cand), min(a.n, len(cand)), replace=False)) if cand else []
        chosen += [cand[i] for i in idx]
    if not chosen:
        raise SystemExit("[error] 모든 결과 폴더에 공통으로 있는 사례가 없어요")

    (a.out / "sheets").mkdir(parents=True, exist_ok=True)
    key, review = [], []
    k = len(runs)
    for i, r in enumerate(chosen):
        sc, pid = r["scenario"], r["patch_id"]
        img, m = load_patch(a.patches, r["split"], pid)
        e = np.array(Image.open(a.manip / sc / "edited_masks" / f"{pid}.png"))
        E = (m != e) & (e != NODATA)
        order = rng.permutation(k)
        outs = [np.array(Image.open(runs[j][1] / "images" / sc / f"{pid}.png").convert("RGB")) for j in order]
        panels = [("original", img), ("edited map", colorize(e))] + [(f"result {n + 1}", o) for n, o in enumerate(outs)]
        y, x, side = focus_box(E)
        W = 256 * len(panels)
        canvas = Image.new("RGB", (W, 24 + 256 + 4 + 256 + 20), "white")
        d = ImageDraw.Draw(canvas)
        for c, (title, p) in enumerate(panels):
            d.text((c * 256 + 4, 4), title, fill="black")
            canvas.paste(Image.fromarray(p), (c * 256, 24))
            crop = Image.fromarray(p).crop((x, y, x + side, y + side)).resize((256, 256), Image.Resampling.NEAREST)
            canvas.paste(crop, (c * 256, 24 + 256 + 4))
        d.text((4, 24 + 512 + 6), f"#{i + 1:02d} {sc}  (아래 줄 = 편집 부분 확대)", fill="black")
        canvas.save(a.out / "sheets" / f"{i + 1:02d}_{sc}.png")
        key.append({"no": i + 1, "scenario": sc, "patch_id": pid,
                    **{f"result_{n + 1}": runs[j][0] for n, j in enumerate(order)}})
        review.append({"no": i + 1, "scenario": sc,
                       **{f"result_{n + 1}_{q}": "" for n in range(k) for q in ("intent_1to5", "natural_1to5")},
                       "note": ""})

    for name, data in (("key.csv", key), ("review.csv", review)):
        with open(a.out / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"그림 {len(chosen)}장 → {a.out / 'sheets'}")
    print("review.csv 를 채운 뒤에 key.csv 를 열어 어느 결과가 어느 모델인지 확인하세요.")
    print("점수 기준: intent = 추가는 '건물로 보이나', 삭제는 '건물이 사라졌나' / natural = 주변과 자연스러운가 (1~5)")


if __name__ == "__main__":
    main()
