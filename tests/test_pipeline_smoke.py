"""가짜 데이터로 조작 세트 → 학습 2 step → 평가가 끝까지 도는지 확인"""
import csv
import json

from helpers import run_script, train_args


def test_manip_set_building_only(fake_patches, tmp_path):
    out = tmp_path / "manip"
    run_script("make_manip_set.py", "--patches", fake_patches, "--out", out, "--n_per", 3)
    scen = {r["scenario"] for r in csv.DictReader(open(out / "manifest.csv"))}
    assert scen <= {"building_add", "building_remove"} and scen


def test_train_then_evaluate(fake_patches, fake_weights, team_repo, tmp_path):
    manip = tmp_path / "manip"
    run_script("make_manip_set.py", "--patches", fake_patches, "--out", manip, "--n_per", 3)
    run = tmp_path / "run"
    run_script("train_full.py", *train_args(team_repo, fake_weights, fake_patches, run))
    assert (run / "checkpoints" / "best.pt").exists()
    res = tmp_path / "res"
    run_script("run_teammate_gan.py", "--repo", team_repo, "--weights_dir", fake_weights,
               "--checkpoint", run / "checkpoints" / "best.pt", "--patches", fake_patches,
               "--manip", manip, "--out", res, "--batch", 4)
    summary = json.loads((res / "summary.json").read_text())
    assert "building_add" in summary or "building_remove" in summary
