"""
생성기 변형 (팀원 model.py는 수정하지 않고 여기에 둔다)

convT  : 팀원 Generator 그대로. 디코더가 ConvTranspose2d(커널 4, stride 2)로 그림을 2배씩 키운다
resize : 같은 구조에서 '키우는 층'만 Upsample(nearest, 2배) + Conv2d(3×3)로 바꾼 것

왜: 짧은 시험 결과 그림에서 새 건물 자리가 지붕이 아니라 규칙적인 점 격자로 채워졌고, 시작 모델과
추가 학습 모델 모두 같았다. ConvTranspose로 키울 때 생기는 격자 결함(checkerboard artifact,
Odena et al. 2016 "Deconvolution and Checkerboard Artifacts")으로 보고, 표준 해결법인
'늘린 뒤 일반 conv'로 바꿔 비교한다.

가중치 이어받기 (load_matching): 이름과 크기가 같은 텐서만 가져온다.
  resize 생성기는 인코더(down)와 지도 주입(semantic_bias)을 semantic_04000에서 그대로 가져오고,
  디코더(up)와 출력층(out)은 새로 학습한다.
"""
import torch
from torch import nn

from model import Generator  # 팀원 코드

GENERATORS = ("convT", "resize")


class GeneratorResize(Generator):
    """팀원 Generator와 입력·출력·forward가 같고, 디코더의 ConvTranspose만 Upsample + Conv3×3으로 바꿈"""

    def __init__(self):
        super().__init__()
        self.up = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(a, b, 3, 1, 1, bias=False),
                          nn.InstanceNorm2d(b), nn.ReLU())
            for a, b in zip([256, 512, 512, 256, 128], [256, 256, 128, 64, 32])])
        self.out = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh())


def build_generator(kind="convT"):
    if kind == "convT":
        return Generator()
    if kind == "resize":
        return GeneratorResize()
    raise ValueError(f"모르는 생성기 종류: {kind} ({GENERATORS})")


def generator_kind(state):
    """체크포인트에 기록된 생성기 종류 (기록이 없으면 팀원 구조 convT)"""
    return ((state.get("config") or {}).get("generator") or "convT") if isinstance(state, dict) else "convT"


def load_matching(module, state_dict):
    """이름과 크기가 같은 텐서만 불러온다. 반환: (불러온 키 목록, 건너뛴 키 목록)"""
    own = module.state_dict()
    take = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    own.update(take)
    module.load_state_dict(own, strict=True)
    skipped = sorted(set(own) - set(take))
    return sorted(take), skipped


@torch.no_grad()
def same_interface(kind="resize", size=256):
    """테스트용: 입력 11채널 → 출력 3채널, 같은 크기"""
    g = build_generator(kind).eval()
    return tuple(g(torch.zeros(1, 11, size, size)).shape)
