"""
복사-붙여넣기 베이스라인 (하한선): 다른 사진에서 해당 클래스가 있는 부분을 오려 그대로 붙인다.
생성 모델 없이 할 수 있는 고전적인 조작이라, 제안 GAN이 이것보다 자연스러워야 의미가 있다.

같은 조작 세트(manifest.csv)를 쓰고, 결과 형식과 채점은 run_teammate_gan.py 와 같다.
  - 기증 패치 후보: 같은 split, 다른 원본(src_id)의 패치에서 무작위 --n_pool 개 (seed 고정)
  - building_add: 편집 영역 E 위치에서 라벨이 '건물'인 비율이 가장 높은 기증 패치를 골라, 그 RGB를 E에만 붙임
  - building_remove: 지운 건물 요소마다 채움 클래스 f에 대해, 그 위치에서 라벨이 f인 비율이 가장 높은
                     기증 패치를 골라 그 RGB를 붙임
  - E 밖(구멍의 나머지 포함)은 원본 그대로

출력 (OUT): images/{시나리오}/{patch_id}.png (SYNTHETIC, generator=copy_paste), sheets/, results.csv, summary.json

사용법:
  python baseline_copy_paste.py --repo ../loveda-semantic-editing --patches data/LoveDA_patches \
      --manip data/LoveDA_manip --out results/copy_paste
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin
from scipy import ndimage as ndi
from tqdm import tqdm

from loveda_lib import BUILDING, NODATA, colorize, load_patch, weights_dir
from score_results import load_audit, make_predict, print_summary, score


def source_key(r):
    return (r["src_split"], r["domain"], r["src_id"])


def best_donor(pool_masks, region, cls, banned):
    """region 위치에서 라벨이 cls인 비율이 가장 높은 기증 패치 번호 (같은 원본은 제외)"""
    cover = (pool_masks[:, region] == cls).mean(axis=1)
    cover[banned] = -1
    return int(cover.argmax()), float(cover.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=Path, required=True)
    ap.add_argument("--manip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n_pool", type=int, default=2000, help="기증 후보 패치 수 (메모리: 개당 64KB)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", nargs="*", default=["building_add", "building_remove"])
    ap.add_argument("--limit", type=int, default=None, help="시나리오별 최대 개수 (빠른 점검용)")
    ap.add_argument("--n_sheets", type=int, default=8)
    # 채점용
    ap.add_argument("--repo", type=Path, default=None, help="팀원 저장소 (분할 모델 채점용). 없으면 분할 지표 생략")
    ap.add_argument("--weights_dir", type=Path, default=None)
    ap.add_argument("--audit", type=Path, default=None)
    ap.add_argument("--no_audit", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    with open(a.manip / "manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["scenario"] in a.scenarios]
    if a.limit:
        per, keep = defaultdict(int), []
        for r in rows:
            if per[r["scenario"]] < a.limit:
                keep.append(r); per[r["scenario"]] += 1
        rows = keep
    if not rows:
        raise SystemExit("[error] 조작 세트에 해당 시나리오가 없어요")
    split = rows[0]["split"]
    with open(a.patches / "index.csv") as f:
        index = {r["patch_id"]: r for r in csv.DictReader(f) if r["split"] == split}

    rng = np.random.default_rng(a.seed)
    pool_ids = sorted(index)
    pool_ids = [pool_ids[i] for i in sorted(rng.choice(len(pool_ids), min(a.n_pool, len(pool_ids)), replace=False))]
    pool_masks = np.stack([load_patch(a.patches, split, pid)[1] for pid in tqdm(pool_ids, desc="donor pool")])
    pool_src = [source_key(index[pid]) for pid in pool_ids]
    img_cache = {}

    def donor_img(j):
        if j not in img_cache:
            img_cache[j] = load_patch(a.patches, split, pool_ids[j])[0]
        return img_cache[j]

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sheets").mkdir(exist_ok=True)
    extra, sheets = {}, defaultdict(int)
    for r in tqdm(rows, desc="copy-paste"):
        sc, pid = r["scenario"], r["patch_id"]
        img, m = load_patch(a.patches, r["split"], pid)
        e = np.array(Image.open(a.manip / sc / "edited_masks" / f"{pid}.png"))
        E = (m != e) & (e != NODATA)
        banned = np.array([s == source_key(index[pid]) for s in pool_src])
        fin, used = img.copy(), []
        if sc == "building_add":
            parts = [(E, BUILDING)]
        else:  # 지운 건물 요소마다 그 자리를 채운 클래스
            lab, n = ndi.label(E)
            parts = [(lab == k, int(np.bincount(e[lab == k]).argmax())) for k in range(1, n + 1)]
        for region, cls in parts:
            j, cover = best_donor(pool_masks, region, cls, banned)
            fin[region] = donor_img(j)[region]
            used.append({"donor": pool_ids[j], "class": cls, "coverage": round(cover, 4)})
        extra[(sc, pid)] = {"donor_coverage": round(float(np.mean([u["coverage"] for u in used])), 4)}

        folder = a.out / "images" / sc
        folder.mkdir(parents=True, exist_ok=True)
        info = PngImagePlugin.PngInfo()
        info.add_text("SYNTHETIC", "true")
        info.add_text("generator", "copy_paste")
        info.add_text("configuration", json.dumps(dict(scenario=sc, patch_id=pid, donors=used, seed=a.seed)))
        Image.fromarray(fin).save(folder / f"{pid}.png", pnginfo=info)
        if sheets[sc] < a.n_sheets:
            panels = [("real", img), ("edited map", colorize(e)), ("copy-paste", fin)]
            canvas = Image.new("RGB", (256 * 3, 280), "white")
            d = ImageDraw.Draw(canvas)
            for c, (t, p) in enumerate(panels):
                d.text((c * 256 + 4, 4), t, fill="black")
                canvas.paste(Image.fromarray(p), (c * 256, 24))
            d.text((4, 266), f"{sc} / {pid}", fill="black")
            canvas.save(a.out / "sheets" / f"{sc}_{sheets[sc]:02d}.png")
            sheets[sc] += 1

    predict = None
    if a.repo is not None and not a.no_audit:
        import torch
        device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                              ("cpu" if a.device == "auto" else a.device))
        predict = make_predict(load_audit(a.repo, weights_dir(a.repo, a.weights_dir), a.audit, device), device)
    summary = score(a.out, a.manip, a.patches, predict, extra=extra, rows=rows, batch=a.batch,
                    meta=dict(generator="copy_paste", n_pool=len(pool_ids), seed=a.seed))
    print_summary(summary)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
