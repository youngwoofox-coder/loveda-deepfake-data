"""T2 재현 가능한 학습 입력, T4 조작 모양 구멍"""
import numpy as np
import torch

from loveda_lib import (NODATA, LoveDAInpaintDataset, SeededDrawSampler, _scenario_hole, load_patch,
                        sample_train_hole)


def draws(ds, sampler, n):
    it = iter(sampler)
    return [ds[next(it)] for _ in range(n)]


def test_same_seed_same_inputs(fake_patches):
    a = LoveDAInpaintDataset(fake_patches, "train", seed=7)
    b = LoveDAInpaintDataset(fake_patches, "train", seed=7)
    xa = draws(a, SeededDrawSampler(len(a), 7), 20)
    xb = draws(b, SeededDrawSampler(len(b), 7), 20)
    for u, v in zip(xa, xb):
        assert u["pid"] == v["pid"] and u["kind"] == v["kind"]
        assert torch.equal(u["input"], v["input"])


def test_epochs_differ(fake_patches):
    ds = LoveDAInpaintDataset(fake_patches, "train", seed=7)
    n = len(ds)
    # 같은 패치라도 epoch가 다르면 구멍이 달라야 함
    same = sum(torch.equal(ds[i]["input"], ds[n + i]["input"]) for i in range(n))
    assert same < n


def test_full_ratio_only_changes_full_samples(fake_patches):
    a = LoveDAInpaintDataset(fake_patches, "train", seed=3, full_ratio=0.0)
    b = LoveDAInpaintDataset(fake_patches, "train", seed=3, full_ratio=0.5)
    xa = draws(a, SeededDrawSampler(len(a), 3), 24)
    xb = draws(b, SeededDrawSampler(len(b), 3), 24)
    n_full = 0
    for u, v in zip(xa, xb):
        assert u["pid"] == v["pid"]
        if v["kind"] == "full":
            n_full += 1
            assert torch.equal(v["hole"][0] > 0, v["mask"] != NODATA)
        else:
            assert torch.equal(u["input"], v["input"])
    assert 0 < n_full < 24


def test_sampler_resume():
    full = SeededDrawSampler(10, 1)
    it = iter(full)
    seq = [next(it) for _ in range(35)]
    it2 = iter(SeededDrawSampler(10, 1, start_draw=17))
    assert [next(it2) for _ in range(18)] == seq[17:]
    assert sorted(x % 10 for x in seq[:10]) == list(range(10))  # 한 epoch에 모든 패치 한 번씩


def test_three_probs_is_old_behavior(fake_patches):
    _, m = load_patch(fake_patches, "train", LoveDAInpaintDataset(fake_patches, "train").ids[0])
    for s in range(10):
        h3, k3 = sample_train_hole(m, np.random.default_rng(s), probs=(0.4, 0.4, 0.2))
        assert k3 != "scenario"


def test_scenario_holes(fake_patches):
    ds = LoveDAInpaintDataset(fake_patches, "train")
    got = 0
    for pid in ds.ids:
        _, m = load_patch(fake_patches, "train", pid)
        hole, kind = sample_train_hole(m, np.random.default_rng(11), probs=(0, 0, 0, 1))
        if kind != "scenario":
            continue
        got += 1
        rng = np.random.default_rng(11)  # sample_train_hole의 재시도 과정을 그대로 재현
        expect = None
        for _ in range(12):
            rng.choice(4, p=np.array([0, 0, 0, 1.0]))
            h = _scenario_hole(m, rng)
            if h is None:
                continue
            h &= m != NODATA
            if 0.03 <= h.mean() <= 0.40:
                expect = h
                break
        assert expect is not None and np.array_equal(hole, expect)
    assert got > 0
