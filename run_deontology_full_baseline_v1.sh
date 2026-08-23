#!/bin/bash
#SBATCH --job-name=deon_full_baseline_v1
#SBATCH --account=def-enaskt_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --output=/scratch/fatimay4/consiliumai/newlogs/deon_full_baseline_v1-%j.out
#SBATCH --error=/scratch/fatimay4/consiliumai/newlogs/deon_full_baseline_v1-%j.err
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

SCRIPT="${SCRIPT:-/scratch/fatimay4/consiliumai/scripts/newscripts/STar/evaluate_deontology_full_baseline_v1.py}"
INPUT_DIR="${INPUT_DIR:-/scratch/fatimay4/consiliumai/outputs/deontology/full_retrieval_inputs_v1}"
QUERIES="${QUERIES:-$INPUT_DIR/hendrycks_queries_label_blind.jsonl}"
PRIVATE_LABELS="${PRIVATE_LABELS:-$INPUT_DIR/hendrycks_private_labels.jsonl}"
MODEL="${MODEL:-/scratch/fatimay4/consiliumai/models/r1_distill_qwen_7b}"
OUT_DIR="${OUT_DIR:-/scratch/fatimay4/consiliumai/outputs/deontology/hendrycks_full_no_retrieval_baseline_v1}"

for path in "$SCRIPT" "$QUERIES" "$PRIVATE_LABELS" "$MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "[ABORT] missing path: $path"
    exit 1
  fi
done

if ! grep -q 'deontology_mechanism_checklist_retrieval_v1' "$SCRIPT"; then
  echo "[ABORT] evaluator does not contain the frozen-50 prompt version"
  exit 1
fi

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

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) start=$(date)"
echo "[config] queries=$QUERIES"
echo "[config] private_labels=$PRIVATE_LABELS"
echo "[config] model=$MODEL"
echo "[config] output=$OUT_DIR"

python -u "$SCRIPT" \
  --queries "$QUERIES" \
  --private-labels "$PRIVATE_LABELS" \
  --model "$MODEL" \
  --output-dir "$OUT_DIR" \
  --batch-size 1 \
  --max-input-tokens 8192 \
  --max-new-tokens 2048 \
  --log-every 20 \
  --seed 42 \
  --expected-records 3595

echo "[slurm] done $(date)"
