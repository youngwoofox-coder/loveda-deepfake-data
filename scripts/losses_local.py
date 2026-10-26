"""
팀원 model.generator_loss 의 복사본 + '지도 없이 학습' 옵션 (조건 없는 GAN 비교군용)

팀원 저장소(loveda-semantic-editing)는 수정하지 않는다. 그래서 손실 함수를 여기에 그대로 복사하고
use_sem=False 일 때만 두 가지를 바꾼다.
  1) 생성기·판별기 입력의 지도 7채널을 0으로 (지도를 못 봄)
  2) 의미 CE 항(분할 교사가 목표 클래스로 인식하게 만드는 손실)을 0으로. 이 항도 라벨 정보를 쓰기 때문
use_sem=True, 가중치 기본값일 때는 model.generator_loss 와 수치적으로 같다 (tests/test_nosem.py 에서 확인).

손실 가중치(weights)도 바꿀 수 있다 (흐려짐 원인 시험용). 기본값 = 팀원 값
  l1 50, boundary 10, adv 2, fm 3, perceptual 2, texture 0.5, gradient 5, ce 3
  ce = 0 이면 분할 교사(teacher)를 아예 돌리지 않는다 (의미 CE 끔)

주의: no-data(0) 픽셀을 손실에서 빼는 'valid' 마스크는 그대로 둔다. 학습 신호를 어디서 계산할지의 문제일 뿐
생성기 입력으로 들어가는 정보가 아니기 때문.
"""
import torch
from torch.nn import functional as F

from model import condition, discriminator_features, feature_losses, normalize, onehot, region_mean  # 팀원 코드


def semantic_input(label, use_sem=True):
    """생성기·판별기에 넣을 지도 7채널. use_sem=False 면 전부 0"""
    s = onehot(label)
    return s if use_sem else torch.zeros_like(s)


DEFAULT_WEIGHTS = dict(l1=50.0, boundary=10.0, adv=2.0, fm=3.0, perceptual=2.0, texture=0.5, gradient=5.0, ce=3.0)


def generator_loss_local(g, d, teacher, features, image, label, mask, iteration, use_sem=True, weights=None):
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    use_ce = use_sem and w["ce"] > 0
    semantic = semantic_input(label, use_sem); cond = condition(image, semantic, mask)
    raw = g(cond); fake = image * (1 - mask) + raw * mask
    valid = (label != 0)[:, None].float(); support = mask * valid
    score, df = discriminator_features(d, torch.cat([cond, fake], 1))
    adv = region_mean((score.float() - 1).square(), support)
    l1 = region_mean((raw.float() - image).abs(), support)
    ring = (mask + F.max_pool2d(-mask, 7, 1, 3)).clamp_min(0) * valid
    boundary = region_mean((raw.float() - image).abs(), ring)
    with torch.no_grad():
        _, dr = discriminator_features(d, torch.cat([cond, image], 1)); real_features = features(image)
        if use_ce:
            probabilities = teacher(normalize(image))['out'].float().softmax(1)
            confidence, prediction = probabilities.max(1)
    fm = sum(region_mean((a.float() - b.float()).abs(), support) for a, b in zip(df, dr)) / len(df)
    perceptual, texture = feature_losses(features(fake), real_features, support)
    gradient = sum(region_mean((torch.diff(fake.float(), dim=axis) - torch.diff(image, dim=axis)).abs(),
                               support[:, :, :, 1:] if axis == -1 else support[:, :, 1:, :]) for axis in [-1, -2]) / 2
    if use_ce:
        target = label.long() - 1; target = target.masked_fill(label == 0, 255)
        allowed = (prediction == target) & (confidence >= .5) & (label >= 2) & (mask[:, 0] > 0)
        logits = teacher(normalize(fake))['out'].float()
        ce = region_mean(F.cross_entropy(logits, target, reduction='none', ignore_index=255)[:, None],
                         allowed[:, None].float())
    else:
        allowed = torch.zeros_like(label, dtype=torch.bool)
        ce = l1 * 0
    loss = w["l1"] * l1 + w["boundary"] * boundary + w["adv"] * adv + w["fm"] * fm + w["perceptual"] * perceptual \
        + w["texture"] * texture + w["gradient"] * gradient + w["ce"] * min(1., iteration / 500) * ce
    return loss, dict(l1=l1, boundary=boundary, adversarial=adv, feature_matching=fm, perceptual=perceptual,
                      texture=texture, gradient=gradient, semantic_ce=ce,
                      semantic_supervised_fraction=allowed.sum() / support.sum().clamp_min(1),
                      semantic_allowed_pixels=torch.stack([(allowed & (label == k)).sum() for k in range(1, 8)]))
