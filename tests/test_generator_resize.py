"""생성기 변형 resize: 같은 입출력, 인코더 이어받기, 학습·평가가 끝까지 도는지"""
import json

import torch

from helpers import run_script, train_args


def test_resize_interface_and_partial_load(fake_weights):
    from model_local import build_generator, load_matching, same_interface
    assert same_interface("resize") == (1, 3, 256, 256)
    st = torch.load(fake_weights / "semantic_04000.pt", map_location="cpu", weights_only=False)
    g = build_generator("resize")
    took, fresh = load_matching(g, st["generator"])
    assert any(k.startswith("down.") for k in took) and any(k.startswith("semantic_bias.") for k in took)
    assert all(k.split(".")[0] in ("up", "out") for k in fresh) and fresh
    assert torch.equal(g.state_dict()["down.0.0.weight"], st["generator"]["down.0.0.weight"])


def test_resize_train_and_evaluate(fake_patches, fake_weights, team_repo, tmp_path):
    run = tmp_path / "run"
    run_script("train_full.py", *train_args(team_repo, fake_weights, fake_patches, run, extra=["--generator", "resize"]))
    assert json.loads((run / "config.json").read_text())["generator"] == "resize"
    manip = tmp_path / "manip"
    run_script("make_manip_set.py", "--patches", fake_patches, "--out", manip, "--n_per", 3)
    res = tmp_path / "res"
    out = run_script("run_teammate_gan.py", "--repo", team_repo, "--weights_dir", fake_weights,
                     "--checkpoint", run / "checkpoints" / "best.pt", "--patches", fake_patches, "--manip", manip,
                     "--out", res).stdout
    assert "generator=resize" in out
    assert json.loads((res / "summary.json").read_text())["_meta"]["generator_arch"] == "resize"
