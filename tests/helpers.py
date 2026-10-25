import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run_script(name, *args, check=True):
    """scripts/ 아래 스크립트를 별도 프로세스로 실행 (실제 사용과 같은 방식)"""
    cmd = [sys.executable, str(SCRIPTS / name), *map(str, args)]
    res = subprocess.run(cmd, cwd=SCRIPTS, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise AssertionError(f"{name} 실패 (코드 {res.returncode})\n--- stdout ---\n{res.stdout[-3000:]}\n--- stderr ---\n{res.stderr[-3000:]}")
    return res


def train_args(team_repo, weights, patches, out, steps=2, batch=2, extra=()):
    return ["--repo", team_repo, "--weights_dir", weights, "--patches", patches, "--out", out,
            "--steps", steps, "--batch", batch, "--workers", 0, "--val_every", steps, "--save_every", steps,
            "--val_n", 4, "--probe_n", 4, "--no_amp", *extra]
