"""
짧은 시험 결과 비교표 (마크다운). screen.sh 마지막에 자동 실행되고, 따로 실행해도 된다.

  python experiments/compare.py                 # runs/screen_* 와 results_val/screen_*
  python experiments/compare.py --prefix screen --base A_base

읽는 것
  runs/{prefix}_{시험}/config.json, val_log.csv, train_log.csv
  results_val/{prefix}_{시험}/summary.json   (val 조작 세트로 best.pt 평가한 결과)

판정 가이드 (학습 시험 가이드 4절과 같음)
  주 기준: edit_score = (건물 추가 intent + 건물 삭제 gone) / 2
  탈락 표시: 기준(A_base) 대비 삭제 gone -0.05 이하, 추가 precision -0.05 이하, 복원 오차 +1.0 이상
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"full_ratio": 0.0, "hole_probs": "0.4,0.4,0.2", "focus": "none", "focus_weights": None,
            "require": None, "no_sem": False, "init": "weights/semantic_04000.pt", "lr": 0.0001,
            "l1_weight": 50.0, "adv_weight": 2.0, "ce_weight": 3.0}


def read_csv(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def collect(run_dir, res_dir):
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    val = read_csv(run_dir / "val_log.csv")
    scored = [r for r in val if num(r.get("edit_score")) is not None]
    best = max(scored, key=lambda r: num(r["edit_score"])) if scored else {}
    train = read_csv(run_dir / "train_log.csv")
    summ = json.loads((res_dir / "summary.json").read_text()) if (res_dir / "summary.json").exists() else {}
    add, rem = summ.get("building_add", {}), summ.get("building_remove", {})
    a_int, r_gone = num(add.get("intent_changed")), num(rem.get("gone_changed"))
    return {
        "setting": ", ".join(f"{k}={cfg[k]}" for k, d in DEFAULTS.items() if k in cfg and cfg[k] != d) or "기본값",
        "steps": cfg.get("steps"), "batch": cfg.get("batch"), "seed": cfg.get("seed"),
        "best_step": best.get("step", "-"), "val_edit": num(best.get("edit_score")),
        "add_intent": a_int, "add_precision": num(add.get("add_precision_M")), "overgen": num(add.get("overgen_C")),
        "remove_gone": r_gone, "keep_bld": num(rem.get("keep_buildings")),
        "edit_score": (a_int + r_gone) / 2 if a_int is not None and r_gone is not None else None,
        "recon_add": num(add.get("recon_mae")), "recon_rem": num(rem.get("recon_mae")),
        "sec_per_step": num(train[-1]["sec_per_step"]) if train else None,
        "n": (add.get("n"), rem.get("n")),
        "git": (cfg.get("git", {}).get("data_repo", {}) or {}).get("commit", "")[:7],
    }


def flags(r, b):
    if b is None or r is b:
        return ""
    out = []
    if None not in (r["remove_gone"], b["remove_gone"]) and r["remove_gone"] <= b["remove_gone"] - 0.05:
        out.append("삭제 하락")
    if None not in (r["add_precision"], b["add_precision"]) and r["add_precision"] <= b["add_precision"] - 0.05:
        out.append("정밀도 하락")
    for k in ("recon_add", "recon_rem"):
        if None not in (r[k], b[k]) and r[k] >= b[k] + 1.0:
            out.append("복원 악화")
            break
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="screen")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--results", type=Path, default=ROOT / "results_val")
    ap.add_argument("--base", default="A_base", help="탈락 판정 기준 시험")
    a = ap.parse_args()

    names = sorted(p.name[len(a.prefix) + 1:] for p in a.runs.glob(f"{a.prefix}_*") if p.is_dir())
    if not names:
        raise SystemExit(f"[error] {a.runs}/{a.prefix}_* 시험이 없어요")
    rows = {n: collect(a.runs / f"{a.prefix}_{n}", a.results / f"{a.prefix}_{n}") for n in names}
    base = rows.get(a.base)

    print(f"## 짧은 시험 비교 (val 조작 세트, prefix={a.prefix})\n")
    print("| 시험 | 바꾼 설정 | best step | edit_score ↑ | 추가 intent ↑ | 추가 precision ↑ | 과잉 생성 ↓ "
          "| 삭제 gone ↑ | 주변 건물 유지 ↑ | 복원 오차 추가/삭제 ↓ | 초/step | 탈락 |")
    print("|" + "---|" * 12)
    for n in sorted(names, key=lambda k: -(rows[k]["edit_score"] or -1)):
        r = rows[n]
        print(f"| {n} | {r['setting']} | {r['best_step']} | {fmt(r['edit_score'])} | {fmt(r['add_intent'])} "
              f"| {fmt(r['add_precision'])} | {fmt(r['overgen'])} | {fmt(r['remove_gone'])} | {fmt(r['keep_bld'])} "
              f"| {fmt(r['recon_add'], 2)} / {fmt(r['recon_rem'], 2)} | {fmt(r['sec_per_step'], 2)} | {flags(r, base)} |")
    any_r = next(iter(rows.values()))
    print(f"\n공통: steps={any_r['steps']}, batch={any_r['batch']}, seed={any_r['seed']}, "
          f"평가 조작 수(추가, 삭제)={any_r['n']}, 코드 커밋={any_r['git'] or '-'}")
    print("edit_score 차이가 0.02 미만이면 차이 없음으로 보고 더 단순한 설정(A_base)을 고른다.")


if __name__ == "__main__":
    main()
