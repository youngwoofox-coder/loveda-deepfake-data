"""T5: 조건 없는 GAN (--no_sem)"""
import json

import torch

from helpers import run_script, train_args


def _inputs(fake_patches):
    from loveda_lib import LoveDAInpaintDataset
    ds = LoveDAInpaintDataset(fake_patches, "train", seed=0, hole_kwargs=dict(probs=(1, 0, 0)))
    b = [ds[i] for i in range(2)]
    return (torch.stack([x["target"] for x in b]), torch.stack([x["mask"] for x in b]),
            torch.stack([x["hole"] for x in b]))


def test_local_loss_matches_teammate(fake_patches, fake_weights):
    from losses_local import generator_loss_local
    from model import Discriminator, FrozenFeatures, Generator, generator_loss, segmentation
    torch.manual_seed(0)
    g, d = Generator().eval(), Discriminator().eval()
    teacher = segmentation(fake_weights / "semantic_teacher.pt", torch.device("cpu"))
    features = FrozenFeatures(fake_weights / "squeezenet_features.pt").eval()
    img, lab, hole = _inputs(fake_patches)
    with torch.no_grad():
        l0, m0 = generator_loss(g, d, teacher, features, img, lab, hole, 300)
        l1, m1 = generator_loss_local(g, d, teacher, features, img, lab, hole, 300, use_sem=True)
        l2, m2 = generator_loss_local(g, d, teacher, features, img, lab, hole, 300, use_sem=False)
    assert torch.allclose(l0, l1)
    for k in m0:
        assert torch.allclose(m0[k].float(), m1[k].float()), k
    assert float(m2["semantic_ce"]) == 0.0 and torch.isfinite(l2)


def test_nosem_train_resume_and_map_blind(fake_patches, fake_weights, team_repo, tmp_path):
    run = tmp_path / "nosem"
    run_script("train_full.py", *train_args(team_repo, fake_weights, fake_patches, run, steps=2,
                                            extra=["--no_sem", "--save_every", 1]))
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["no_sem"] is True
    run_script("train_full.py", *train_args(team_repo, fake_weights, fake_patches, run, steps=3,
                                            extra=["--no_sem", "--save_every", 1, "--resume"]))
    manip = tmp_path / "manip"
    run_script("make_manip_set.py", "--patches", fake_patches, "--out", manip, "--n_per", 3)
    res = tmp_path / "res"
    run_script("run_teammate_gan.py", "--repo", team_repo, "--weights_dir", fake_weights,
               "--checkpoint", run / "checkpoints" / "latest.pt", "--patches", fake_patches,
               "--manip", manip, "--out", res, "--save_raw")
    assert json.loads((res / "summary.json").read_text())["_meta"]["no_sem"] is True
    # 지도를 못 보므로, 지도를 바꿔도 생성 결과가 완전히 같아야 함
    from model import Generator, condition
    st = torch.load(run / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    g = Generator().eval(); g.load_state_dict(st["generator"])
    img, lab, hole = _inputs(fake_patches)
    from losses_local import semantic_input
    edited = lab.clone(); edited[edited == 1] = 2
    with torch.no_grad():
        a = g(condition(img, semantic_input(lab, False), hole))
        b = g(condition(img, semantic_input(edited, False), hole))
    assert torch.equal(a, b)
