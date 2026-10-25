# LoveDA 위성영상 부분 조작 — 데이터 파이프라인

신뢰가능AI 팀 프로젝트(딥페이크 생성·탐지)의 **데이터·전처리 담당 코드**입니다.
실제 위성사진의 토지피복 지도를 규칙에 따라 고쳐서(건물 생성·삭제, 물 확장·축소),
생성 모델이 그 부분만 다시 그리게 만들고, **조작 위치 정답**을 함께 저장합니다.

모델은 재환님 저장소 [loveda-semantic-editing](https://github.com/ruet66/loveda-semantic-editing)의
GAN을 **수정하지 않고 그대로** 불러다 씁니다. 이 저장소는 데이터를 준비하고, 그 모델을
이 데이터로 학습·평가하는 부분만 담당합니다.

```
[이 저장소]  원본 → 256 패치(팀 기준 분할) → 학습 데이터 / 평가용 조작 세트
                                   │                    │
[재환님 저장소] model.py (생성기·판별기·손실) ◀── train_full.py   run_teammate_gan.py ──▶ 결과·지표
```

---

## 1. 준비

```bash
git clone https://github.com/wooyeon-0923/loveda-deepfake-data.git
git clone https://github.com/ruet66/loveda-semantic-editing.git     # 같은 상위 폴더에 나란히
pip install -r loveda-deepfake-data/requirements.txt                # torch/torchvision은 GPU에 맞게 먼저 설치

# 팀원 가중치 설치 (Releases v2026.10.22의 1022_weights.zip)
cd loveda-semantic-editing && python setup_assets.py --archive /경로/1022_weights.zip && cd ..

# LoveDA 원본 다운로드 (Train 4.0GB + Val 2.4GB, md5 확인·압축 해제까지)
cd loveda-deepfake-data && bash scripts/download_loveda.sh data/LoveDA
```

## 2. 한 번에 실행

```bash
GPU=0 nohup bash run_pipeline.sh > logs/pipeline.log 2>&1 &          # 데이터 + 조작 세트 + 기존 모델 평가
TRAIN=1 GPU=0 nohup bash run_pipeline.sh > logs/pipeline.log 2>&1 &  # 위 + 학습(2만 step) + 학습 모델 평가
grep "=====" logs/pipeline.log                                        # 진행 단계 확인
```
경로가 다르면 `LOVEDA=... TEAM_REPO=... bash run_pipeline.sh`처럼 앞에 붙여서 바꿉니다.
이미 만든 패치·조작 세트는 건너뜁니다.

---

## 3. 데이터 분할 (재환님 저장소 목록을 그대로 따름)

| split | 원본 | 용도 |
|---|---|---|
| train | 공식 Train 2,490장 (`data/train_sources.json`) | 생성 모델 학습. 팀원 학습 목록과 동일 |
| val | 공식 Train 과거 평가용 32장 + 공식 Val 중 예약 안 된 562장 = 594장 | 학습 중 검증·설정 선택 |
| test | 공식 Val 1,107장 (`data/final_test_sources.json`) | 평가용 조작 세트, 최종 수치, 기말 탐지 데이터 |

- **원본 사진 단위**로 나눕니다. 같은 사진에서 나온 패치가 학습과 평가에 섞이지 않습니다.
- 패치는 256×256, 겹치지 않게 원본 1장당 16개. no-data가 10% 넘는 패치는 제외합니다.
- 모든 파일은 PNG(무손실). 진짜·가짜의 저장 형식 차이를 탐지기가 배우지 않게 하기 위함입니다.

## 4. 모델 입력 (11채널, 팀원 모델과 동일)

| 채널 | 내용 | 값 |
|---|---|---|
| 0~2 | 구멍 낸 사진 RGB | -1~1, 구멍 위치는 0 |
| 3~9 | 토지피복 지도 one-hot (1 배경, 2 건물, 3 도로, 4 물, 5 나지, 6 숲, 7 농경지) | 0/1, no-data는 전부 0 |
| 10 | 구멍 (다시 그릴 곳) | 0/1 |

- **학습:** 지도는 원본 그대로, 구멍만 뚫고 원본 사진을 정답으로 복원 학습
- **조작:** 지도를 규칙으로 고쳐서 넣음 → 구멍 안에 거짓 내용이 그려짐. 구멍 = 조작 위치 정답
- 최종 결과 = 생성 × 구멍 + 원본 × (1 − 구멍). 구멍 밖은 항상 원본과 동일

## 5. 조작 시나리오 (`scripts/loveda_lib.py`)

| 시나리오 | 규칙 | 의미하는 거짓 |
|---|---|---|
| `building_add` | 배경·나지·농경지 위, 기존 건물·도로 24px 이내에 1~3채. 모양은 같은 패치의 실제 건물 복사(70%) 또는 회전 사각형 | 없는 시설 날조 |
| `building_remove` | 패치의 8% 이하 크기 건물 1~3개를 지우고 주변 최빈 클래스로 채움 (도로 제외) | 철거·파괴 은폐 |
| `water_expand_small` / `_large` | 물과 이어진 배경·나지·농경지로만 확장 (도로·건물·숲에 막힘), 면적 3~8% / 10~25% | 침수·범람 |
| `water_shrink` | 땅과 맞닿은 물가부터 메워 나지로 | 매립·간척 |

- 구멍은 바뀐 영역보다 2~6px 넓게 잡아 경계·그림자까지 다시 그리게 합니다.
- 물 라벨이 주변보다 밝은 패치(라벨 오류 의심)는 물 조작에서 제외합니다.

## 6. 학습용 구멍 (모델이 지도를 무시하는 것 방지)

학습 때 지도가 원본과 같으므로, 모델이 지도를 안 보고 주변만 보고 채우는 지름길을 배울 수 있습니다.
이를 막기 위해 구멍을 세 종류로 뽑습니다 (`sample_train_hole`).

| 종류 | 비율 | 설명 |
|---|---|---|
| object | 40% | 물체(건물·웅덩이 등) 1~3개를 통째로 가림 → 지도를 봐야만 맞힐 수 있음 |
| boundary | 40% | 클래스 경계(특히 물가)를 따라 띠 모양 → 경계 위치는 지도에만 있음 |
| free | 20% | 붓질 모양 (일반화용) |

## 7. 스크립트

| 파일 | 입력 | 출력 |
|---|---|---|
| `download_loveda.sh` | — | `data/LoveDA/{Train,Val}` |
| `check_loveda.py` | 원본 폴더 | 짝·개수·클래스 비율 점검 출력 |
| `make_patches.py` | 원본 + 팀원 저장소(분할 목록) | `data/LoveDA_patches/{train,val,test}/{images,masks}`, `index.csv` |
| `make_manip_set.py` | 패치 | `data/LoveDA_manip/{시나리오}/{edited_masks,holes}`, `manifest.csv`, `real_pool.csv`(조작 안 한 진짜 후보, 기말용) |
| `demo_rules.py` | 패치 | `outputs/rules_*.png` (원본 → 원본 지도 → 편집 지도 → 구멍 → 모델 입력) |
| `train_full.py` | 패치 + 팀원 저장소 | `runs/{이름}/checkpoints/*.pt`, 검증 기록, 샘플 그림 |
| `run_teammate_gan.py` | 체크포인트 + 패치 + 조작 세트 | `results/{이름}/images`, `sheets`, `results.csv`, `summary.json` |
| `score_results.py` | 결과 사진 + 조작 세트 | 채점만 (`results.csv`, `results_components.csv`, `summary.json`). 생성 방법과 무관 |
| `baseline_copy_paste.py` | 패치 + 조작 세트 | 복사-붙여넣기 베이스라인 결과와 채점 |
| `losses_local.py` | (공용) | 팀원 `generator_loss` 복사본 + 지도 없이 학습 옵션 |
| `loveda_lib.py` | (공용) | 조작 규칙, 학습 구멍, Dataset, 입력 생성, `SeededDrawSampler` |
| `experiments/screen.sh` | 패치 + val 조작 세트 | 짧은 학습 시험 여러 개를 같은 조건으로 학습·평가 → 비교표 |
| `experiments/compare.py` | `runs/`, `results_val/` | 시험 비교표 (마크다운) |

### train_full.py와 재환님 train.py의 차이
모델 구조·손실·옵티마이저·AMP 설정은 재환님 코드를 그대로 import 합니다. 다른 점은 데이터뿐입니다.
- 미리 자른 패치를 DataLoader 여러 프로세스로 읽음 (팀원 코드는 1024 원본을 매번 열어 자름)
- 학습 구멍이 위 3종류 (팀원 코드는 클래스 주변 사각형 + 패치 전체)
- `--init none`으로 처음부터 학습 가능
- 1,000 step마다 실제 조작(`--probe`)을 해 보고 가장 잘 된 시점을 `best.pt`로 저장

주요 옵션: `--steps`, `--batch`, `--init`, `--full_ratio`(패치 전체 가리기 비율), `--focus building|water`, `--require building`, `--probe`, `--resume`

#### 추가된 옵션 (exp/building-holes)

| 옵션 | 기본값 | 뜻 |
|---|---|---|
| `--seed` | 0 | 패치 순서·회전·구멍을 정함. 같은 seed면 같은 입력, `--full_ratio`만 다른 실험은 일부 샘플만 전체 가리기로 바뀜 |
| `--hole_probs` | `0.4,0.4,0.2` | 4번째 값 = 실제 조작 모양 구멍(건물 옆 빈 땅 / 건물 위). 원래 지도 그대로라 "지도대로 그리기"를 학습 |
| `--focus_weights` | 없음 | 물체 구멍의 클래스별 가중치, 예: `2:4,1:2` (건물 4배, 배경 2배). `--focus`보다 우선 |
| `--probe` | `building_add building_remove` | 검증 때 해 볼 조작들. `best.pt`는 점수 평균 `edit_score`로 선택 |
| `--probe_manip` | 없음 | `make_manip_set.py --split val` 결과. 주면 고정 조작 세트로 검증 (권장). test 세트는 거부 |
| `--probe_n` | 64 | 시나리오별 검증 조작 수 |
| `--no_sem` | 꺼짐 | 지도 없이 학습 (조건 없는 GAN 비교군): 지도 입력과 의미 CE 손실을 끔 |
| `--weights_dir` | `REPO/weights` | 팀원 가중치 폴더 (테스트용) |

`config.json`에 두 저장소의 git 커밋과 수정 여부가 기록됩니다. `train_log.csv`에 구멍 종류별 개수(`n_object` …)가 남습니다.

#### 짧은 시험 (학습 방식 고르기)

```bash
python scripts/make_manip_set.py --patches data/LoveDA_patches --out data/LoveDA_manip_val --split val --n_per 300
STEPS=3000 BATCH=8 NO_AMP=1 nohup bash experiments/screen.sh > logs/screen.log 2>&1 &   # 1060: NO_AMP=1 필수
python experiments/compare.py
```
시험: `A_base`(기준), `B_full20`(전체 가리기 20%), `C_scenario`(조작 모양 구멍 30%), `D_nosem`(조건 없는 GAN). `EXPS="A_base C_scenario"`처럼 골라서 실행.

#### 테스트

```bash
python -m pytest -q tests      # 가짜 데이터·무작위 가중치, CPU 약 2분. 실제 LoveDA·GPU 불필요
```

## 8. 평가 지표 (`run_teammate_gan.py`, `score_results.py`)

영역: E = 편집 지도가 바뀐 픽셀, M = 구멍(다시 그린 곳 = 정답), C = M − E (다시 그렸지만 의미는 그대로여야 함), O = 구멍 밖.
아래 기존 지표에 더해 `add_precision_M`·`add_iou_M`·`overgen_C`(건물 추가), `remove_residual`·`fill_match`·`keep_buildings`(건물 삭제),
`c_sem_keep`·`c_rgb_mae`(주변 보존), 원본 기준선 `real_*`, 건물 크기 구간별(20px 미만 / 20~50 / 50 이상)·도메인별 평균을 계산합니다.
정의는 `scripts/score_results.py` 맨 위 설명에 있습니다.

채점은 재환님의 별도 분할 모델(`semantic_audit.pt`, 학습 손실용 teacher와 다른 가중치)로 합니다.

| 지표 | 의미 | 방향 |
|---|---|---|
| `intent` | 바뀐 픽셀 중 결과가 **목표 클래스**로 인식된 비율 (생성의 핵심) | ↑ |
| `gone` | 바뀐 픽셀 중 결과가 **원래 클래스가 아니게** 인식된 비율 (삭제의 핵심) | ↑ |
| `real_tgt` | 같은 픽셀이 조작 전 원본에서 이미 목표 클래스로 인식된 비율 (기준선) | 참고 |
| `real_ok` | 원본에서 분할 모델이 원래 라벨을 맞힌 비율 (채점관 정확도) | 참고 |
| `recon` | 원래 지도로 생성했을 때 구멍 안 평균 오차 (0~255) | ↓ |

주의: 이 지표는 "의도대로 그렸나"를 분할 모델 기준으로 본 것이고, **사실성(자연스러움)은 따로 봐야 합니다** (그림, KID, 사람 평가).

결과는 [RESULTS.md](RESULTS.md)에 정리했습니다.

## 데이터 이용 조건
LoveDA (Wang et al., NeurIPS 2021 Datasets and Benchmarks), CC BY-NC-SA 4.0, 학술 목적 한정.
Google Earth 이용 약관을 따릅니다. 생성된 이미지에는 PNG 메타데이터로 `SYNTHETIC=true`를 기록합니다.
