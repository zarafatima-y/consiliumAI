#!/bin/bash
#SBATCH --job-name=deon_frozen200_base
#SBATCH --account=def-enaskt_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=0-06:00
#SBATCH --array=0-3%2
#SBATCH --output=/scratch/fatimay4/consiliumai/newlogs/deon_frozen200_base-%A_%a.out
#SBATCH --error=/scratch/fatimay4/consiliumai/newlogs/deon_frozen200_base-%A_%a.err
#SBATCH --mail-user=fyusuf04@my.yorku.ca
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module load StdEnv/2023 python/3.11 cuda/12.2 gcc/12.3 arrow/24.0.0
source "$HOME/envs/consiliumai_env/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/scratch/fatimay4/consiliumai/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SCRIPT="${SCRIPT:-/scratch/fatimay4/consiliumai/scripts/newscripts/STar/evaluate_hendrycks_frozen_200_baselines_v1.py}"
FROZEN_DIR="${FROZEN_DIR:-/scratch/fatimay4/consiliumai/outputs/deontology/hendrycks_frozen_200_v1}"
PUBLIC_INPUTS="${PUBLIC_INPUTS:-$FROZEN_DIR/hendrycks_frozen_200_public.jsonl}"
PRIVATE_LABELS="${PRIVATE_LABELS:-$FROZEN_DIR/hendrycks_frozen_200_private_labels.jsonl}"
RUN_ROOT="${RUN_ROOT:-/scratch/fatimay4/consiliumai/outputs/deontology/hendrycks_frozen_200_baselines_v1}"
QWEN_MODEL="${QWEN_MODEL:-/scratch/fatimay4/consiliumai/models/qwen3_8b}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-/scratch/fatimay4/consiliumai/models/deepseek_r1_distill_qwen_7b}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

case "$TASK_ID" in
  0)
    MODEL_NAME="qwen3_8b"
    MODEL="$QWEN_MODEL"
    CONDITION="direct_verdict"
    ;;
  1)
    MODEL_NAME="qwen3_8b"
    MODEL="$QWEN_MODEL"
    CONDITION="rationale_verdict"
    ;;
  2)
    MODEL_NAME="deepseek_r1_distill_qwen_7b"
    MODEL="$DEEPSEEK_MODEL"
    CONDITION="direct_verdict"
    ;;
  3)
    MODEL_NAME="deepseek_r1_distill_qwen_7b"
    MODEL="$DEEPSEEK_MODEL"
    CONDITION="rationale_verdict"
    ;;
  *)
    echo "[ABORT] expected array task 0-3; found $TASK_ID"
    exit 1
    ;;
esac

OUT_DIR="$RUN_ROOT/$MODEL_NAME/$CONDITION"

for path in "$SCRIPT" "$PUBLIC_INPUTS" "$PRIVATE_LABELS" "$MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "[ABORT] missing path: $path"
    exit 1
  fi
done
mkdir -p /scratch/fatimay4/consiliumai/newlogs "$OUT_DIR"

python - <<'PY'
import accelerate
import torch
import transformers
print(f"[dependency] transformers={transformers.__version__}")
print(f"[dependency] torch={torch.__version__}")
print(f"[dependency] accelerate={accelerate.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("[ABORT] CUDA is not available")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"[ABORT] expected 1 GPU; found {torch.cuda.device_count()}")
print(f"[gpu] name={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
PY

echo "[slurm] job=$SLURM_JOB_ID task=$TASK_ID host=$(hostname) start=$(date)"
echo "[config] model_name=$MODEL_NAME"
echo "[config] model=$MODEL"
echo "[config] condition=$CONDITION"
echo "[config] public_inputs=$PUBLIC_INPUTS"
echo "[config] private_labels=$PRIVATE_LABELS"
echo "[config] output=$OUT_DIR"

python -u "$SCRIPT" \
  --public-inputs "$PUBLIC_INPUTS" \
  --private-labels "$PRIVATE_LABELS" \
  --model "$MODEL" \
  --model-name "$MODEL_NAME" \
  --condition "$CONDITION" \
  --output-dir "$OUT_DIR" \
  --expected-records 200 \
  --max-input-tokens 4096 \
  --max-rationale-tokens 128 \
  --max-rationale-words 80 \
  --max-rationale-sentences 4 \
  --seed 42 \
  --log-every 10

echo "[slurm] done $(date)"
