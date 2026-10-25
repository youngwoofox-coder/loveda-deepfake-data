"""
테스트용 가짜 데이터·가짜 가중치 (실제 LoveDA와 GPU 없이 코드를 검사하기 위함)

- make_fake_patches(root): make_patches.py 출력과 같은 구조
    root/{train,val,test}/{images,masks}/{patch_id}.png, root/index.csv
- make_fake_weights(dir): loveda-semantic-editing/weights 와 같은 파일 이름·형식의 무작위 가중치
    semantic_04000.pt (generator, discriminator), semantic_teacher.pt, semantic_audit.pt, squeezenet_features.pt

주의: 무작위 가중치라 생성 결과·지표 값에는 의미가 없다. 코드가 끝까지 도는지만 본다.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_REPO = Path(os.environ.get("TEAM_REPO", REPO_ROOT.parent / "loveda-semantic-editing")).resolve()
CLASS_KEYS = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
COLORS = {0: (0, 0, 0), 1: (150, 150, 150), 2: (200, 60, 60), 3: (90, 90, 90), 4: (30, 60, 120),
          5: (170, 150, 120), 6: (40, 110, 40), 7: (150, 170, 80)}


def fake_mask(rng, size=256):
    """배경 위에 도로, 건물(도로 옆), 농경지, 숲, 작은 물웅덩이가 있는 지도 (클래스 0~7)"""
    m = np.full((size, size), 1, np.uint8)
    m[size // 2:, : size // 2] = 7                           # 농경지
    m[size // 2:, size // 2 + 30:] = 6                       # 숲
    y = int(rng.integers(50, 75)); m[y:y + 8, :] = 3        # 가로 도로
    x = int(rng.integers(100, 140)); m[:, x:x + 8] = 3       # 세로 도로
    for _ in range(14):                                      # 도로 옆 건물
        h, w = int(rng.integers(10, 26)), int(rng.integers(10, 26))
        yy = int(np.clip(y - h - int(rng.integers(3, 12)), 0, size - h))
        xx = int(rng.integers(0, size - w))
        if (m[yy:yy + h, xx:xx + w] == 1).all():
            m[yy:yy + h, xx:xx + w] = 2
    for _ in range(8):                                       # 도로 아래쪽 건물
        h, w = int(rng.integers(10, 22)), int(rng.integers(10, 22))
        yy, xx = int(rng.integers(y + 12, size // 2 - h)), int(rng.integers(0, size - w))
        if (m[max(yy - 3, 0):yy + h + 3, max(xx - 3, 0):xx + w + 3] == 1).all():
            m[yy:yy + h, xx:xx + w] = 2
    m[200:230, 20:60] = 4                                    # 물웅덩이
    m[size - 3:, size - 3:] = 0                              # no-data 조금
    return m


def fake_image(m, rng):
    img = np.zeros(m.shape + (3,), np.float32)
    for k, c in COLORS.items():
        img[m == k] = c
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_fake_patches(root, n_src=3, per_src=4, size=256, seed=0):
    """split마다 원본 n_src개 × 패치 per_src개"""
    root = Path(root)
    rng = np.random.default_rng(seed)
    rows = []
    plan = [("train", "Train"), ("val", "Val"), ("test", "Val")]
    for si, (split, src_split) in enumerate(plan):
        for sub in ("images", "masks"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        for s in range(n_src):
            domain = "Urban" if s % 2 == 0 else "Rural"
            src_id = str(1000 * si + s)
            for p in range(per_src):
                r, c = divmod(p, 4)
                m = fake_mask(rng, size)
                img = fake_image(m, rng)
                pid = f"{src_split}-{domain}-{src_id}-r{r}c{c}"
                Image.fromarray(img).save(root / split / "images" / f"{pid}.png")
                Image.fromarray(m).save(root / split / "masks" / f"{pid}.png")
                ratios = [round(float((m == k).mean()), 4) for k in range(1, 8)]
                rows.append([pid, split, src_split, domain, src_id, r * size, c * size,
                             round(float((m == 0).mean()), 4)] + ratios)
    with open(root / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch_id", "split", "src_split", "domain", "src_id", "y", "x", "nodata"] + CLASS_KEYS)
        w.writerows(rows)
    return root


def make_fake_weights(out, seed=0):
    """loveda-semantic-editing/weights 와 같은 형식의 무작위 가중치"""
    import torch
    from torchvision.models import squeezenet1_1
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    if str(TEAM_REPO) not in sys.path:
        sys.path.insert(0, str(TEAM_REPO))
    from model import Discriminator, Generator
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.save({"generator": Generator().state_dict(), "discriminator": Discriminator().state_dict(), "step": 4000},
               out / "semantic_04000.pt")
    for name in ("semantic_teacher.pt", "semantic_audit.pt"):
        net = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=7, aux_loss=False)
        torch.save({"model": net.state_dict()}, out / name)
    torch.save({"model": squeezenet1_1(weights=None).features[:8].state_dict()}, out / "squeezenet_features.pt")
    return out
