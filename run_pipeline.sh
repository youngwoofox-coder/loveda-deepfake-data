#!/usr/bin/env bash
# 데이터 준비 → 조작 세트 → (선택) 학습 → 평가를 순서대로 실행한다.
# 경로와 GPU는 아래 변수를 바꾸거나, 실행할 때 앞에 붙여서 지정한다.
#   예) TEAM_REPO=~/loveda-semantic-editing GPU=3 bash run_pipeline.sh
# 백그라운드:  nohup bash run_pipeline.sh > logs/pipeline.log 2>&1 &
set -euo pipefail
log(){ echo -e "\n===== [$(date '+%m-%d %H:%M:%S')] $* ====="; }
cd "$(dirname "$0")"

LOVEDA=${LOVEDA:-data/LoveDA}                    # 원본 (Train, Val 폴더가 있는 곳)
PATCHES=${PATCHES:-data/LoveDA_patches}          # 256 패치
MANIP=${MANIP:-data/LoveDA_manip}                # 평가용 조작 세트
TEAM_REPO=${TEAM_REPO:-../loveda-semantic-editing}   # 팀원 GAN 저장소 (가중치 설치 완료 상태)
GPU=${GPU:-0}
SCENARIOS=${SCENARIOS:-"building_remove building_add"}   # 물 시나리오는 이름을 붙여 쓰면 사용 가능
N_PER=${N_PER:-1000}
TRAIN=${TRAIN:-0}                                # 1이면 학습까지
STEPS=${STEPS:-20000}
RUN=${RUN:-runs/ft_ourholes}
mkdir -p logs outputs results

log "0. 확인"
test -d "$LOVEDA/Train" || { echo "원본이 없어요: $LOVEDA  →  bash scripts/download_loveda.sh $LOVEDA"; exit 1; }
test -f "$TEAM_REPO/model.py" || { echo "팀원 저장소가 없어요: $TEAM_REPO"; exit 1; }
test -f "$TEAM_REPO/weights/semantic_04000.pt" || { echo "팀원 가중치 미설치: $TEAM_REPO/setup_assets.py 실행 필요"; exit 1; }

log "1. 패치 + 팀 기준 분할"
if [ ! -f "$PATCHES/index.csv" ]; then
  python scripts/make_patches.py --root "$LOVEDA" --out "$PATCHES" --team_repo "$TEAM_REPO"
else echo "이미 있음 → 건너뜀 (다시 만들려면 $PATCHES 삭제)"; fi

log "2. 평가용 조작 세트"
if [ ! -f "$MANIP/manifest.csv" ]; then
  rm -rf "$MANIP"
  python scripts/make_manip_set.py --patches "$PATCHES" --out "$MANIP" --n_per "$N_PER" --scenarios $SCENARIOS
else echo "이미 있음 → 건너뜀 (다시 만들려면 $MANIP 삭제)"; fi

log "3. 규칙 확인 그림"
python scripts/demo_rules.py --patches "$PATCHES" --out outputs --scenarios $SCENARIOS

log "4. 팀원 기존 모델 평가"
CUDA_VISIBLE_DEVICES=$GPU python scripts/run_teammate_gan.py --repo "$TEAM_REPO" \
  --patches "$PATCHES" --manip "$MANIP" --out results/semantic_04000

if [ "$TRAIN" = "1" ]; then
  log "5. 학습 ($STEPS step) → $RUN"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/train_full.py --repo "$TEAM_REPO" --patches "$PATCHES" \
    --out "$RUN" --steps "$STEPS" --batch 16
  log "6. 학습 모델 평가"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_teammate_gan.py --repo "$TEAM_REPO" \
    --checkpoint "$RUN/checkpoints/step_$(printf %06d "$STEPS").pt" \
    --patches "$PATCHES" --manip "$MANIP" --out "results/$(basename "$RUN")"
fi
log "전부 완료"
