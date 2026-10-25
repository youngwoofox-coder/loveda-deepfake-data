"""
평가용 조작 세트를 '고정'해서 저장한다.
모든 생성 방법(제안 GAN, SD 베이스라인, 복사-붙여넣기 등)이 똑같은 사진·똑같은 조작 영역으로
비교되도록, 조작 내용(편집 마스크 + 구멍)을 미리 한 번만 만들어 둔다.
이 세트가 그대로 기말 탐지 데이터의 정답(조작 위치)이 된다.

- 하나의 test 패치는 하나의 시나리오에만 사용 (시나리오 간 중복 없음)
- 조작에 쓰지 않은 test 패치는 real_pool.csv 로 따로 저장 → 기말 '진짜' 이미지 후보
  (같은 사진의 조작 전/후가 진짜·가짜 양쪽에 들어가지 않게 하기 위함)

출력 (OUT):
  {시나리오}/edited_masks/{patch_id}.png   편집 마스크 (클래스 0~7)
  {시나리오}/holes/{patch_id}.png          구멍 = 조작 위치 정답 (0/255)
  manifest.csv     manip_id, scenario, patch_id, split, domain, hole_ratio, changed_ratio, info
  real_pool.csv    조작에 쓰지 않은 test 패치 목록

사용법:
  python make_manip_set.py --patches data/LoveDA_patches \
                           --out data/LoveDA_manip --n_per 1000
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from loveda_lib import SCENARIO_CANDIDATE, SCENARIOS, WATER, load_patch


def water_looks_like_water(img, m):
    """라벨 오류 거르기: '물' 라벨 픽셀이 나머지보다 어두운지 확인.
    공사장 흙바닥 등이 물로 잘못 라벨링된 패치를 제외하기 위한 간단한 휴리스틱."""
    lum = img.astype(np.float32).mean(axis=2)
    w = m == WATER
    other = (m != WATER) & (m != 0)
    if w.sum() < 50 or other.sum() < 50:
        return True
    return lum[w].mean() < lum[other].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_per", type=int, default=1000, help="시나리오별 조작 개수")
    ap.add_argument("--scenarios", nargs="+", default=["building_remove", "building_add"],
                    choices=list(SCENARIOS), help="기본: 건물 두 시나리오. 물 시나리오도 이름을 주면 사용 가능")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_water_check", action="store_true", help="물 라벨 밝기 검사 끄기")
    a = ap.parse_args()

    with open(a.patches / "index.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == a.split]
    rng = np.random.default_rng(a.seed)
    used, manifest = set(), []

    # 후보가 적은 시나리오(건물)부터 뽑아서 패치를 먼저 확보
    order = sorted(a.scenarios, key=lambda s: sum(
        SCENARIO_CANDIDATE[s](r) for r in rows))
    for name in order:
        pool = [r for r in rows if SCENARIO_CANDIDATE[name](r) and r["patch_id"] not in used]
        (a.out / name / "edited_masks").mkdir(parents=True, exist_ok=True)
        (a.out / name / "holes").mkdir(parents=True, exist_ok=True)
        ok, tried, label_skip = 0, 0, 0
        bar = tqdm(total=a.n_per, desc=name)
        for j in rng.permutation(len(pool)):
            if ok >= a.n_per:
                break
            r = pool[j]
            pid = r["patch_id"]
            tried += 1
            img, m = load_patch(a.patches, a.split, pid)
            if name.startswith("water") and not a.no_water_check and not water_looks_like_water(img, m):
                label_skip += 1
                continue
            # 패치마다 고정 시드 → 다시 실행해도 같은 조작
            res = SCENARIOS[name](m, rng=np.random.default_rng([a.seed, tried]))
            if res is None:
                continue
            edited, hole, info = res
            Image.fromarray(edited.astype(np.uint8)).save(a.out / name / "edited_masks" / f"{pid}.png")
            Image.fromarray(hole.astype(np.uint8) * 255).save(a.out / name / "holes" / f"{pid}.png")
            used.add(pid)
            ok += 1
            bar.update(1)
            manifest.append({"manip_id": f"{name}/{pid}", "scenario": name, "patch_id": pid,
                             "split": a.split, "domain": r["domain"],
                             "hole_ratio": info["hole_ratio"], "changed_ratio": info["changed_ratio"],
                             "info": json.dumps(info, ensure_ascii=False)})
        bar.close()
        msg = f"  {name}: {ok}개 저장 (시도 {tried}개, 성공률 {ok / max(tried, 1) * 100:.1f}%, 라벨 의심으로 제외 {label_skip}개)"
        if ok < a.n_per:
            msg += f"  ← 후보 부족으로 목표 {a.n_per}개 미달"
        print(msg)

    with open(a.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    with open(a.out / "real_pool.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch_id", "split", "domain"])
        w.writerows([r["patch_id"], r["split"], r["domain"]] for r in rows if r["patch_id"] not in used)

    print(f"\n조작 {len(manifest)}개 → {a.out / 'manifest.csv'}")
    print(f"조작에 안 쓴 진짜 후보 {len(rows) - len(used)}개 → {a.out / 'real_pool.csv'}")
    for name in a.scenarios:
        hr = [m["hole_ratio"] for m in manifest if m["scenario"] == name]
        if hr:
            print(f"  {name}: 조작 면적 평균 {np.mean(hr) * 100:.1f}% (최소 {min(hr) * 100:.1f}%, 최대 {max(hr) * 100:.1f}%)")


if __name__ == "__main__":
    main()
