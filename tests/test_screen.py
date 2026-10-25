"""T8: 시험 프리셋이 끝까지 돌고, 다시 실행하면 이어서 하는지"""
import os
import subprocess
from pathlib import Path

from helpers import run_script

ROOT = Path(__file__).resolve().parents[1]


def test_screen_runs_and_resumes(fake_patches, fake_weights, team_repo, tmp_path):
    val_manip = tmp_path / "manip_val"
    run_script("make_manip_set.py", "--patches", fake_patches, "--out", val_manip, "--split", "val", "--n_per", 3)
    env = dict(os.environ, PATCHES=str(fake_patches), VAL_MANIP=str(val_manip), TEAM_REPO=str(team_repo),
               RUNS=str(tmp_path / "runs"), RESULTS=str(tmp_path / "res"), STEPS="2", BATCH="2", WORKERS="0",
               VAL_EVERY="1", PROBE_N="3", NO_AMP="1", EXPS="A_base B_full20", EXTRA=f"--weights_dir {fake_weights}")
    for _ in range(2):  # 두 번째는 --resume으로 이어서 (이미 끝났으므로 평가만 다시)
        res = subprocess.run(["bash", str(ROOT / "experiments" / "screen.sh")], env=env, capture_output=True, text=True)
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    table = (tmp_path / "res" / "screen_compare.md").read_text()
    assert "| A_base |" in table and "| B_full20 | full_ratio=0.2" in table
