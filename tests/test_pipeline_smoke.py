"""가짜 데이터로 조작 세트 → 학습 2 step → 평가가 끝까지 도는지 확인"""
import csv
import json

from helpers import run_script, train_args


def test_manip_set_building_only(fake_patches, tmp_path):
    out = tmp_path / "manip"
    run_script("make_manip_set.py", "--patches", fake_patches, "--out", out, "--n_per", 3)
    scen = {r["scenario"] for r in csv.DictReader(open(out / "manifest.csv"))}
    assert scen <= {"building_add", "building_remove"} and scen


def test_validation_both_building_scenarios(fake_patches, fake_weights, team_repo, tmp_path):
    val_manip = tmp_path / "manip_val"
    run_script("make_manip_set.py", "--patches", fake_patches, "--out", val_manip, "--split", "val", "--n_per", 3)
    run = tmp_path / "run"
    run_script("train_full.py", *train_args(team_repo, fake_weights, fake_patches, run, steps=2,
                                            extra=["--probe_manip", val_manip, "--val_every", 1]))
    rows = list(csv.DictReader(open(run / "val_log.csv")))
    assert len(rows) == 3  # step 0, 1, 2
    for col in ("val_hole_l1", "add_intent", "remove_gone", "edit_score"):
        assert col in rows[0]
    assert (run / "checkpoints" / "best.pt").exists()


def test_probe_set_must_be_val(fake_patches, fake_weights, team_repo, tmp_path):
    test_manip = tmp_path / "manip_test"
    run_script("make_manip_set.py", "--patches", fake_patches, "--out", test_manip, "--n_per", 3)
    res = run_script("train_full.py", *train_args(team_repo, fake_weights, fake_patches, tmp_path / "run",
                                                  extra=["--probe_manip", test_manip]), check=False)
    assert res.returncode != 0 and "val" in (res.stdout + res.stderr)


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
