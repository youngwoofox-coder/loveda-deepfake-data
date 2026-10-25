#!/usr/bin/env bash
# 짧은 학습 시험 여러 개를 '같은 조건'으로 돌리고, val 조작 세트로 평가한 뒤 비교표를 만든다.
# 다른 점은 시험마다 한 가지 설정뿐 (seed·배치·step·검증 세트는 모두 같음). test 분할은 쓰지 않는다.
#
#   STEPS=3000 BATCH=8 NO_AMP=1 bash experiments/screen.sh               # 1060: 혼합 정밀도 끄기 필수
#   EXPS="A_base B_full20" STEPS=3000 BATCH=16 bash experiments/screen.sh  # P100 등
#   nohup ... > logs/screen.log 2>&1 &                                   # 백그라운드
#
# 시험 목록
#   A_base      기준 (현재 ft_ourholes와 같은 구멍: 물체 40%, 경계 띠 40%, 붓질 20%)
#   B_full20    A + 패치 전체 가리기 20%  (--full_ratio 0.2)
#   C_scenario  구멍의 30%를 실제 조작 모양으로 (--hole_probs 0.3,0.3,0.1,0.3)
#   D_nosem     A + 지도 없이 학습 (조건 없는 GAN 비교군, --no_sem). 보고서용이라 본 학습 때 주 모델과 같이 돌림
# 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 한다 (끝난 시험은 학습을 건너뛰고 평가만 다시 함).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$(command -v python >/dev/null 2>&1 && echo python || echo python3)}
PATCHES=${PATCHES:-data/LoveDA_patches}
VAL_MANIP=${VAL_MANIP:-data/LoveDA_manip_val}
TEAM_REPO=${TEAM_REPO:-../loveda-semantic-editing}
STEPS=${STEPS:-3000}
BATCH=${BATCH:-8}
SEED=${SEED:-0}
VAL_EVERY=${VAL_EVERY:-500}
PROBE_N=${PROBE_N:-200}
WORKERS=${WORKERS:-4}
NO_AMP=${NO_AMP:-0}
GPU=${GPU:-0}
PREFIX=${PREFIX:-screen}
RUNS=${RUNS:-runs}               # 학습 결과 폴더
RESULTS=${RESULTS:-results_val}  # 평가 결과 폴더
EXPS=${EXPS:-"A_base B_full20 C_scenario"}
EXTRA=${EXTRA:-}            # 모든 학습·평가에 덧붙일 옵션 (예: 테스트용 --weights_dir)
EXTRA_TRAIN=${EXTRA_TRAIN:-}  # 학습에만 덧붙일 옵션 (예: --init none)

log(){ echo -e "\n===== [$(date '+%m-%d %H:%M:%S')] $* ====="; }
exp_args(){
  case "$1" in
    A_base)     echo "" ;;
    B_full20)   echo "--full_ratio 0.2" ;;
    C_scenario) echo "--hole_probs 0.3,0.3,0.1,0.3" ;;
    D_nosem)    echo "--no_sem" ;;
    *) echo "[error] 모르는 시험 이름: $1 (A_base B_full20 C_scenario D_nosem)" >&2; exit 1 ;;
  esac
}

test -f "$VAL_MANIP/manifest.csv" || { echo "[error] val 조작 세트가 없어요: $VAL_MANIP
  $PY scripts/make_manip_set.py --patches $PATCHES --out $VAL_MANIP --split val --n_per 300"; exit 1; }
AMP_FLAG=""; [ "$NO_AMP" = "1" ] && AMP_FLAG="--no_amp"
mkdir -p "$RUNS" "$RESULTS" logs
export CUDA_VISIBLE_DEVICES=$GPU

for E in $EXPS; do
  ARGS=$(exp_args "$E")
  RUN=$RUNS/${PREFIX}_$E
  RESUME=""; [ -f "$RUN/config.json" ] && RESUME="--resume"
  log "학습 $E ($ARGS) → $RUN"
  # shellcheck disable=SC2086
  $PY scripts/train_full.py --repo "$TEAM_REPO" --patches "$PATCHES" --out "$RUN" \
    --steps "$STEPS" --batch "$BATCH" --seed "$SEED" --workers "$WORKERS" \
    --val_every "$VAL_EVERY" --save_every "$VAL_EVERY" \
    --probe_manip "$VAL_MANIP" --probe_n "$PROBE_N" $AMP_FLAG $ARGS $RESUME $EXTRA $EXTRA_TRAIN
  log "평가 $E (val 조작 세트, best.pt)"
  # shellcheck disable=SC2086
  $PY scripts/run_teammate_gan.py --repo "$TEAM_REPO" --checkpoint "$RUN/checkpoints/best.pt" \
    --patches "$PATCHES" --manip "$VAL_MANIP" --out "$RESULTS/${PREFIX}_$E" $EXTRA
done

log "비교표"
$PY experiments/compare.py --prefix "$PREFIX" --runs "$RUNS" --results "$RESULTS" | tee "$RESULTS/${PREFIX}_compare.md"
log "전부 완료"
