"""
위성영상 부분 조작 프로젝트 공용 코드 (데이터 담당)

1) 조작 규칙 (추론 = 조작할 때)
   water_expand(mask)    물 영역을 넓힘  → 편집 마스크, 구멍 마스크, 정보
   building_remove(mask) 건물을 지움     → 편집 마스크, 구멍 마스크, 정보
2) 학습용 구멍 샘플링
   sample_train_hole(mask)  물체 단위 / 경계 띠 / 자유 붓질 구멍
   → 모델이 '마스크를 무시하고 주변만 보고 채우는' 지름길을 못 쓰게 하기 위한 설계
3) 모델 입력 만들기 (학습·조작 공통)
   make_input(img, cond_mask, hole) → 11채널 입력, 정답 사진
4) 학습용 Dataset
   LoveDAInpaintDataset(root, split)

클래스 번호: 0 no-data, 1 background, 2 building, 3 road, 4 water, 5 barren, 6 forest, 7 agriculture
구멍 마스크(hole): True = 모델이 다시 그릴 픽셀. 조작할 때는 이것이 곧 '조작 위치 정답'.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

NODATA, BACKGROUND, BUILDING, ROAD, WATER, BARREN, FOREST, AGRI = range(8)
N_CLASSES = 7  # one-hot 채널 수 (1~7). no-data는 모든 채널 0

PALETTE = np.array([
    [0, 0, 0],        # 0 no-data
    [255, 255, 255],  # 1 background
    [255, 0, 0],      # 2 building
    [255, 255, 0],    # 3 road
    [0, 0, 255],      # 4 water
    [159, 129, 183],  # 5 barren
    [0, 255, 0],      # 6 forest
    [255, 195, 128],  # 7 agriculture
], dtype=np.uint8)


# ---------------------------------------------------------------- 가중치 경로
def weights_dir(repo, override=None):
    """팀원 가중치 폴더. 기본은 REPO/weights, 테스트에서는 --weights_dir 로 가짜 가중치를 가리킨다."""
    return Path(override) if override else Path(repo) / "weights"


def resolve_init(init, repo, wdir):
    """--init 경로 해석. 절대경로 → 그대로, 'weights/xxx.pt' 같은 상대경로 → 가중치 폴더, REPO, 현재 폴더 순으로 찾는다."""
    p = Path(init)
    if p.is_absolute():
        return p
    cands = [Path(wdir) / p.name] if p.parts and p.parts[0] == "weights" else []
    cands += [Path(wdir) / p, Path(repo) / p, p]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"초기 가중치를 못 찾았어요: {init} (찾아본 곳: {[str(c) for c in cands]})")


# ---------------------------------------------------------------- 기본 연산
def dilate(b, r):
    """반경 r 원형 팽창 (거리 변환 기반이라 r이 커도 빠름)"""
    if r <= 0 or not b.any() or b.all():
        return b.copy()
    return ndi.distance_transform_edt(~b) <= r


def erode(b, r):
    """반경 r 원형 침식. 패치 바깥은 같은 값으로 이어진다고 보고 가장자리는 깎지 않음"""
    if r <= 0 or not b.any() or b.all():
        return b.copy()
    return ndi.distance_transform_edt(b) > r


def smooth_noise(shape, rng, sigma=8.0):
    """경계를 자연스럽게 울퉁불퉁하게 만들 부드러운 잡음 (평균 0, 표준편차 1)"""
    n = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return n / (n.std() + 1e-8)


def colorize(mask):
    return PALETTE[mask]


# ---------------------------------------------------------------- 조작 규칙
WATER_LEVELS = {"small": (0.03, 0.08), "large": (0.10, 0.25)}  # 조작 강도별 면적(패치 대비)


def water_expand(mask, rng=None, level="small", allowed=(BACKGROUND, BARREN, AGRI),
                 area_range=None, radii=(4, 6, 8, 12, 16, 24, 32, 48, 64),
                 jitter=0.35, margin=2, min_island=200):
    """물을 주변으로 넓힌다 (침수·범람 위조).

    - 허용 클래스(배경·나지·농경지)만 물로 바뀜. 건물·도로·숲은 그대로
    - 물과 '이어진' 곳만 바뀜 → 도로나 건물을 건너뛰어 물이 생기지 않음
    - 경계에 잡음을 줘서 해안선과 평행한 띠처럼 보이지 않게 함
    - 바뀐 면적이 level 범위(small 3~8%, large 10~25%)에 들어오도록 반경을 자동 선택
      (area_range를 직접 주면 그것을 우선 사용)
    반환: (편집 마스크, 구멍 마스크, 정보) / 조건을 못 맞추면 None
    """
    rng = rng or np.random.default_rng()
    area_range = area_range or WATER_LEVELS[level]
    water = mask == WATER
    if not water.any():
        return None
    passable = np.isin(mask, allowed) | water
    dist = ndi.distance_transform_edt(~water)
    noise = smooth_noise(mask.shape, rng)

    for r in radii:
        reach = (dist <= r + jitter * r * noise) & passable
        reach = ndi.binary_propagation(water, mask=reach)  # 물과 이어진 부분만
        # 물에 둘러싸여 남은 작은 땅 조각(어색한 점 같은 섬)도 물로 채움
        islands, n = ndi.label(~reach)
        if n:
            sizes = np.bincount(islands.ravel())
            touches = np.zeros(n + 1, bool)
            for edge in (islands[0], islands[-1], islands[:, 0], islands[:, -1]):
                touches[edge] = True
            small = (sizes < min_island) & ~touches
            small[0] = False
            reach |= small[islands] & passable
        changed = reach & ~water
        ratio = float(changed.mean())
        if ratio >= area_range[0]:
            break
    if not (area_range[0] <= ratio <= area_range[1]):
        return None

    edited = mask.copy()
    edited[changed] = WATER
    hole = dilate(changed, margin) & (mask != NODATA)  # 경계 섞임 여유까지 다시 그림
    info = {"scenario": "water_expand", "level": level, "radius": int(r), "changed_ratio": round(ratio, 4),
            "hole_ratio": round(float(hole.mean()), 4)}
    return edited, hole, info


def water_shrink(mask, rng=None, area_range=(0.03, 0.15), radii=(4, 6, 8, 12, 16, 24, 32),
                 jitter=0.35, margin=2, fill=BARREN, land=(BACKGROUND, BARREN, FOREST, AGRI)):
    """물가를 메워 땅으로 바꾼다 (매립·간척 위조). 팀원 GAN 평가와 같게 나지(5)로 채움.

    - 땅(배경·나지·숲·농경지)과 맞닿은 물가에서 안쪽으로 반경만큼만 바뀜
    - 도로·건물과 맞닿은 물가는 출발점이 아님 (제방·다리 옆이 갑자기 메워지지 않게)
    - 경계에 잡음을 줘 자연스러운 매립선 모양
    반환: (편집 마스크, 구멍 마스크, 정보) / 조건을 못 맞추면 None
    """
    rng = rng or np.random.default_rng()
    water = mask == WATER
    landm = np.isin(mask, land)
    if not water.any() or not landm.any():
        return None
    dist = ndi.distance_transform_edt(~landm)  # 땅까지의 거리
    noise = smooth_noise(mask.shape, rng)
    for r in radii:
        changed = water & (dist <= r + jitter * r * noise)
        ratio = float(changed.mean())
        if ratio >= area_range[0]:
            break
    if not (area_range[0] <= ratio <= area_range[1]):
        return None
    edited = mask.copy()
    edited[changed] = fill
    hole = dilate(changed, margin) & (mask != NODATA)
    info = {"scenario": "water_shrink", "radius": int(r), "changed_ratio": round(ratio, 4),
            "hole_ratio": round(float(hole.mean()), 4)}
    return edited, hole, info


def building_remove(mask, rng=None, area_range=(0.01, 0.15), min_px=30, max_comp_frac=0.08,
                    max_buildings=3, ring=6, margin=6,
                    fill_exclude=(NODATA, BUILDING, ROAD)):
    """건물 덩어리를 지우고 주변에서 가장 많은 클래스로 채운다 (철거·파괴 은폐 위조).

    - 너무 작은 덩어리(min_px 미만)와 너무 큰 덩어리(패치의 max_comp_frac 초과, 대형 아파트 등)는 제외
    - 전체 바뀐 면적은 area_range 이내 (기본 1~15%)
    - 주변 ring 픽셀 띠에서 가장 많은 클래스로 채움 (도로·건물·no-data는 후보에서 제외)
    - 구멍은 margin만큼 넓혀 건물 가장자리와 일부 그림자까지 다시 그리게 함
    반환: (편집 마스크, 구멍 마스크, 정보) / 조건을 못 맞추면 None
    """
    rng = rng or np.random.default_rng()
    bld = mask == BUILDING
    lab, n = ndi.label(bld)
    if n == 0:
        return None
    sizes = ndi.sum(bld, lab, index=np.arange(1, n + 1))
    ids = [i + 1 for i, s in enumerate(sizes) if min_px <= s <= max_comp_frac * mask.size]
    if not ids:
        return None
    rng.shuffle(ids)

    edited = mask.copy()
    changed = np.zeros_like(bld)
    fills = []
    for i in ids:
        comp = lab == i
        if (changed | comp).mean() > area_range[1]:
            continue
        around = mask[dilate(comp, ring) & ~comp]
        around = around[~np.isin(around, fill_exclude)]
        fill = int(np.bincount(around.astype(np.int64), minlength=8).argmax()) if around.size else BACKGROUND
        edited[comp] = fill
        changed |= comp
        fills.append(fill)
        if len(fills) >= max_buildings:
            break
    ratio = float(changed.mean())
    if not fills or ratio < area_range[0]:
        return None

    hole = dilate(changed, margin) & (mask != NODATA)
    info = {"scenario": "building_remove", "n_buildings": len(fills), "fill_classes": fills,
            "changed_ratio": round(ratio, 4), "hole_ratio": round(float(hole.mean()), 4)}
    return edited, hole, info


def _rect_shape(rng, side=(18, 60)):
    """회전된 직사각형 건물 모양 (0.3m 해상도에서 약 5~18m)"""
    h, w = int(rng.integers(*side)), int(rng.integers(*side))
    ang = float(rng.uniform(0, np.pi / 2))
    c, s_ = np.cos(ang), np.sin(ang)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rot = pts @ np.array([[c, -s_], [s_, c]]).T
    rot -= rot.min(0)
    W_, H_ = int(np.ceil(rot[:, 0].max())) + 1, int(np.ceil(rot[:, 1].max())) + 1
    im = Image.new("L", (W_, H_), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in rot], fill=255)
    return np.array(im) > 0


def building_add(mask, rng=None, n_range=(1, 3), area_range=(0.01, 0.12), allowed=(BACKGROUND, BARREN, AGRI),
                 near_px=24, gap=4, margin=6, size_px=(150, 2500), donor_prob=0.7, tries=80):
    """빈 땅에 건물을 새로 짓는다 (없는 시설 날조).

    - 배경·나지·농경지 위에만, 기존 도로나 건물에서 near_px 이내 (외딴 들판 한가운데 X)
    - 기존 건물·도로와 gap 픽셀 이상 띄움, 새 건물끼리도 겹치지 않음
    - 모양: 70%는 같은 패치의 실제 건물 모양을 복사(방향·크기가 동네와 맞음), 나머지는 회전 사각형
    - 구멍은 margin만큼 넓혀 그림자·마당 경계까지 다시 그리게 함
    반환: (편집 마스크, 구멍 마스크, 정보) / 조건을 못 맞추면 None
    """
    rng = rng or np.random.default_rng()
    H, W = mask.shape
    bld, road = mask == BUILDING, mask == ROAD
    if not (bld.any() or road.any()):
        return None
    near = dilate(bld | road, near_px)
    blocked = dilate(bld | road, gap) | ~np.isin(mask, allowed)
    free = near & ~blocked
    if not free.any():
        return None
    lab, n = ndi.label(bld)
    donors = []
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        for i in range(1, n + 1):
            if size_px[0] <= sizes[i] <= size_px[1]:
                ys, xs = np.nonzero(lab == i)
                if ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1:
                    continue  # 패치 가장자리에서 잘린 건물은 모양이 온전하지 않아 제외
                donors.append(lab[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == i)
    target = int(rng.integers(n_range[0], n_range[1] + 1))
    changed = np.zeros(mask.shape, bool)
    kinds = []
    for _ in range(tries):
        if len(kinds) >= target:
            break
        use_donor = bool(donors) and rng.random() < donor_prob
        shape = donors[int(rng.integers(len(donors)))] if use_donor else _rect_shape(rng)
        h, w = shape.shape
        if h >= H or w >= W:
            continue
        cand = free & ~dilate(changed, gap)
        ys, xs = np.nonzero(cand)
        if not len(ys):
            break
        j = int(rng.integers(len(ys)))
        y = int(np.clip(ys[j] - h // 2, 0, H - h))
        x = int(np.clip(xs[j] - w // 2, 0, W - w))
        region = np.zeros(mask.shape, bool)
        region[y:y + h, x:x + w] = shape
        if (region & (blocked | dilate(changed, gap))).any():
            continue
        if (changed | region).mean() > area_range[1]:
            continue
        changed |= region
        kinds.append("copy" if use_donor else "rect")
    ratio = float(changed.mean())
    if not kinds or ratio < area_range[0]:
        return None
    edited = mask.copy()
    edited[changed] = BUILDING
    hole = dilate(changed, margin) & (mask != NODATA)
    info = {"scenario": "building_add", "n_buildings": len(kinds), "shapes": kinds,
            "changed_ratio": round(ratio, 4), "hole_ratio": round(float(hole.mean()), 4)}
    return edited, hole, info


SCENARIOS = {
    "water_expand_small": lambda m, rng=None: water_expand(m, rng=rng, level="small"),
    "water_expand_large": lambda m, rng=None: water_expand(m, rng=rng, level="large"),
    "water_shrink": water_shrink,
    "building_remove": building_remove,
    "building_add": building_add,
}


def _ratio(r, k):
    return float(r[k])


SCENARIO_CANDIDATE = {  # index.csv 한 줄(dict)을 받아 후보인지 판단
    "water_expand_small": lambda r: _ratio(r, "water") >= 0.05,
    "water_expand_large": lambda r: _ratio(r, "water") >= 0.05,
    "water_shrink": lambda r: _ratio(r, "water") >= 0.05,
    "building_remove": lambda r: _ratio(r, "building") >= 0.05,
    # 지을 빈 땅이 있고, 주변에 건물이나 도로가 있는 패치
    "building_add": lambda r: (_ratio(r, "building") >= 0.02 or _ratio(r, "road") >= 0.03)
                              and _ratio(r, "background") + _ratio(r, "barren") + _ratio(r, "agriculture") >= 0.20,
}


# ---------------------------------------------------------------- 학습용 구멍
def _object_hole(mask, rng, area_range=(0.03, 0.40), min_px=20, class_weights=None):
    """물체(웅덩이, 건물, 작은 숲 등) 1~3개를 통째로 + 여유까지 가림.
    주변 사진만 봐서는 안에 뭐가 있었는지 알 수 없으므로, 마스크를 봐야만 맞힐 수 있음."""
    class_weights = class_weights or {WATER: 3.0, BUILDING: 3.0}  # 기본: 물·건물 우선
    max_px = area_range[1] * mask.size * 0.8  # 패치를 뒤덮는 큰 덩어리(넓은 농경지 등)는 제외
    cands, weights = [], []
    for k in range(1, 8):
        lab, n = ndi.label(mask == k)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        for i, s in enumerate(sizes):
            if min_px <= s <= max_px:
                cands.append((lab, i + 1))
                weights.append(class_weights.get(k, 1.0))
    if not cands:
        return None
    w = np.array(weights) / sum(weights)
    picks = rng.choice(len(cands), size=min(int(rng.integers(1, 4)), len(cands)), replace=False, p=w)
    comp = np.zeros(mask.shape, bool)
    for j in picks:
        lab, i = cands[j]
        comp |= lab == i
    for margin in (int(rng.integers(4, 13)), 16, 24):  # 너무 작으면 여유를 늘림
        hole = dilate(comp, margin)
        if hole.mean() >= area_range[0]:
            break
    return hole


def _boundary_band_hole(mask, rng):
    """클래스 경계(특히 물가)를 따라 띠 모양으로 가림.
    경계가 정확히 어디인지는 마스크를 봐야만 알 수 있어서, 수역 확장 조작과 비슷한 상황을 연습시킴."""
    r = int(rng.integers(3, 13))
    water = mask == WATER
    if water.any() and rng.random() < 0.6:
        band = dilate(water, r) & ~erode(water, r)
    else:
        edge = np.zeros(mask.shape, bool)
        edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        edge[1:, :] |= mask[1:, :] != mask[:-1, :]
        if not edge.any():
            return None
        band = dilate(edge, r)
    band &= mask != NODATA
    if not band.any():
        return None
    # 띠 전체가 아니라 띠 위의 한 점을 중심으로 일부 구간만 남겨 면적과 모양을 다양하게
    H, W = mask.shape
    ys, xs = np.nonzero(band)
    j = rng.integers(len(ys))
    yy, xx = np.ogrid[:H, :W]
    d2 = (yy - ys[j]) ** 2 + (xx - xs[j]) ** 2
    rad = float(rng.integers(H // 8, H // 2))
    for _ in range(4):  # 너무 작으면 범위를 넓힘
        hole = band & (d2 <= rad * rad)
        if hole.mean() >= 0.03:
            break
        rad *= 1.6
    return hole


def _free_form_hole(shape, rng):
    """붓으로 그은 듯한 자유 모양 구멍 (일반화용, 소수만 섞음)"""
    H, W = shape
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(1, 4))):
        pts = [(int(rng.integers(0, W)), int(rng.integers(0, H))) for _ in range(int(rng.integers(2, 6)))]
        d.line(pts, fill=255, width=int(rng.integers(10, 35)), joint="curve")
    return np.array(im) > 0


def sample_train_hole(mask, rng=None, area_range=(0.03, 0.40), probs=(0.4, 0.4, 0.2), tries=12,
                      class_weights=None):
    """학습용 구멍 하나를 뽑는다. 반환: (구멍 bool 배열, 종류 이름)
    class_weights: 물체 구멍에서 클래스별 선택 가중치 (예: {2: 8.0} → 건물을 훨씬 자주 가림)"""
    rng = rng or np.random.default_rng()
    kinds = ["object", "boundary", "free"]
    for _ in range(tries):
        kind = kinds[rng.choice(3, p=probs)]
        if kind == "object":
            hole = _object_hole(mask, rng, area_range, class_weights=class_weights)
        elif kind == "boundary":
            hole = _boundary_band_hole(mask, rng)
        else:
            hole = _free_form_hole(mask.shape, rng)
        if hole is None:
            continue
        hole &= mask != NODATA
        if area_range[0] <= hole.mean() <= area_range[1]:
            return hole, kind
    # 모두 실패하면 사각형 구멍 (거의 안 쓰임)
    H, W = mask.shape
    h, w = H // 3, W // 3
    y, x = rng.integers(0, H - h), rng.integers(0, W - w)
    hole = np.zeros(mask.shape, bool)
    hole[y:y + h, x:x + w] = True
    return hole & (mask != NODATA), "rect"


# ---------------------------------------------------------------- 모델 입력
def make_input(img, cond_mask, hole):
    """학습·조작 공통 입력 생성.

    img: (H,W,3) uint8 원본 사진 / cond_mask: (H,W) 0~7 조건 마스크 / hole: (H,W) bool
    반환: x (11,H,W) float32 = [구멍 낸 사진 3 (-1~1, 구멍=0), one-hot 7, 구멍 1]
          target (3,H,W) float32 원본 사진 (-1~1)
    """
    import torch
    import torch.nn.functional as F
    target = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    h = torch.from_numpy(np.ascontiguousarray(hole)).float()[None]
    m = torch.from_numpy(np.ascontiguousarray(cond_mask)).long()
    onehot = F.one_hot(m, 8)[..., 1:].permute(2, 0, 1).float()  # 0(no-data) 버리고 1~7만
    x = torch.cat([target * (1 - h), onehot, h], dim=0)
    return x, target


def load_patch(root, split, pid):
    root = Path(root)
    img = np.array(Image.open(root / split / "images" / f"{pid}.png").convert("RGB"))
    m = np.array(Image.open(root / split / "masks" / f"{pid}.png"))
    return img, m


# ---------------------------------------------------------------- Dataset
try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # torch 없이 규칙만 쓸 때
    _TorchDataset = object


class LoveDAInpaintDataset(_TorchDataset):
    """학습/검증용 Dataset. 마스크는 고치지 않고(원본 그대로) 구멍만 뚫는다.

    반환 dict
      input  (11,H,W)  모델 입력
      target (3,H,W)   정답 = 원본 사진 (-1~1)
      hole   (1,H,W)   구멍 (손실을 구멍 안/밖으로 나눠 계산할 때 사용)
      mask   (H,W)     원본 클래스 0~7 (의미 일치 손실 등에 사용)
      kind   str       구멍 종류
      pid    str       패치 ID

    fixed_holes=True 이면 같은 index에 항상 같은 구멍 → 검증 점수 비교용
    """

    def __init__(self, root, split="train", augment=True, fixed_holes=False, seed=0,
                 ids=None, hole_kwargs=None):
        self.root, self.split = Path(root), split
        self.augment, self.fixed_holes, self.seed = augment, fixed_holes, seed
        self.hole_kwargs = hole_kwargs or {}
        self.ids = ids if ids is not None else sorted(p.stem for p in (self.root / split / "images").glob("*.png"))
        if not self.ids:
            raise FileNotFoundError(f"{self.root / split / 'images'} 에 패치가 없어요.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torch
        pid = self.ids[i]
        img, m = load_patch(self.root, self.split, pid)
        rng = np.random.default_rng(self.seed + i) if self.fixed_holes else np.random.default_rng()
        if self.augment:  # 위성사진은 방향이 의미가 없어서 회전·뒤집기 사용 (사진과 마스크에 똑같이)
            k = int(rng.integers(0, 4))
            img, m = np.rot90(img, k), np.rot90(m, k)
            if rng.random() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
        hole, kind = sample_train_hole(m, rng, **self.hole_kwargs)
        x, target = make_input(img, m, hole)
        return {"input": x, "target": target, "hole": x[-1:], "mask": torch.from_numpy(m.astype(np.int64)),
                "kind": kind, "pid": pid}
