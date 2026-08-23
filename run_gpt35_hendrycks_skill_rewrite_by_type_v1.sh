#!/bin/bash
set -euo pipefail

module load StdEnv/2023 python/3.11
source "$HOME/envs/consiliumai_env/bin/activate"

export PYTHONUNBUFFERED=1
: "${OPENAI_API_KEY:?Export OPENAI_API_KEY before running}"

SCRIPT="${SCRIPT:-/scratch/fatimay4/consiliumai/scripts/newscripts/STar/gpt35_hendrycks_skill_rewrite_by_type_v1.py}"

EXEMPLAR_DIR="${EXEMPLAR_DIR:-/scratch/fatimay4/consiliumai/outputs/deontology/deon_hendrycks_skill_exemplers}"
REQUEST_SEEDS="${REQUEST_SEEDS:-$EXEMPLAR_DIR/hendrycks_request_seed_bank_16.jsonl}"
DUTY_ROLE_SEEDS="${DUTY_ROLE_SEEDS:-$EXEMPLAR_DIR/hendrycks_duty_role_seed_bank_16.jsonl}"

SPLIT_DIR="${SPLIT_DIR:-/home/fatimay4/scratch/consiliumai/outputs/deontology/full_test_split_v1}"
REQUEST_INPUTS="${REQUEST_INPUTS:-$SPLIT_DIR/hendrycks_test_request.jsonl}"
DUTY_ROLE_INPUTS="${DUTY_ROLE_INPUTS:-$SPLIT_DIR/hendrycks_test_duty_role.jsonl}"

OUT_DIR="${OUT_DIR:-/scratch/fatimay4/consiliumai/outputs/deontology/hendrycks_test_skills_gpt35_by_type_v1}"
MODEL="${MODEL:-gpt-3.5-turbo}"

for path in \
  "$SCRIPT" \
  "$REQUEST_SEEDS" \
  "$DUTY_ROLE_SEEDS" \
  "$REQUEST_INPUTS" \
  "$DUTY_ROLE_INPUTS"; do
  if [[ ! -f "$path" ]]; then
    echo "[ABORT] missing file: $path"
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

echo "[login] host=$(hostname) start=$(date)"
echo "[config] model=$MODEL"
echo "[config] request_seeds=$REQUEST_SEEDS"
echo "[config] duty_role_seeds=$DUTY_ROLE_SEEDS"
echo "[config] request_inputs=$REQUEST_INPUTS"
echo "[config] duty_role_inputs=$DUTY_ROLE_INPUTS"
echo "[config] output=$OUT_DIR"

python -u "$SCRIPT" \
  --request-seed-bank "$REQUEST_SEEDS" \
  --duty-role-seed-bank "$DUTY_ROLE_SEEDS" \
  --request-inputs "$REQUEST_INPUTS" \
  --duty-role-inputs "$DUTY_ROLE_INPUTS" \
  --out-dir "$OUT_DIR" \
  --model "$MODEL" \
  --max-tokens 200 \
  --batch-size 500 \
  --audit-size 100 \
  --seed 42 \
  --no-wait

echo "[login] finished $(date)"
