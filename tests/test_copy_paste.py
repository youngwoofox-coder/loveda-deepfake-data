"""T7: 복사-붙여넣기 베이스라인"""
import csv

import numpy as np
from PIL import Image

from helpers import run_script
from loveda_lib import NODATA, load_patch


def test_copy_paste_only_changes_E_and_is_deterministic(fake_patches, fake_weights, team_repo, tmp_path):
    manip = tmp_path / "manip"
    run_script("make_manip_set.py", "--patches", fake_patches, "--out", manip, "--n_per", 3)
    outs = []
    for k in range(2):
        out = tmp_path / f"cp{k}"
        run_script("baseline_copy_paste.py", "--patches", fake_patches, "--manip", manip, "--out", out,
                   "--repo", team_repo, "--weights_dir", fake_weights)
        outs.append(out)
    rows = list(csv.DictReader(open(manip / "manifest.csv")))
    assert rows
    for r in rows:
        img, m = load_patch(fake_patches, r["split"], r["patch_id"])
        e = np.array(Image.open(manip / r["scenario"] / "edited_masks" / f"{r['patch_id']}.png"))
        E = (m != e) & (e != NODATA)
        a = np.array(Image.open(outs[0] / "images" / r["scenario"] / f"{r['patch_id']}.png").convert("RGB"))
        b = np.array(Image.open(outs[1] / "images" / r["scenario"] / f"{r['patch_id']}.png").convert("RGB"))
        assert np.array_equal(a, b)
        assert np.array_equal(a[~E], img[~E])
        assert not np.array_equal(a[E], img[E])
    assert (outs[0] / "summary.json").exists()
