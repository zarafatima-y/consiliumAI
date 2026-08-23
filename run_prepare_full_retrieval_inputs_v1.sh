#!/bin/bash
set -euo pipefail

module load StdEnv/2023 python/3.11
source "$HOME/envs/consiliumai_env/bin/activate"

export PYTHONUNBUFFERED=1

SCRIPT="${SCRIPT:-/scratch/fatimay4/consiliumai/scripts/newscripts/STar/prepare_full_retrieval_inputs_v1.py}"

HENDRYCKS_SKILLS="${HENDRYCKS_SKILLS:-/scratch/fatimay4/consiliumai/outputs/deontology/hendrycks_test_skills_gpt35_by_type_v1/hendrycks_test_skills_combined.jsonl}"
HENDRYCKS_PRIVATE_LABELS="${HENDRYCKS_PRIVATE_LABELS:-/home/fatimay4/scratch/consiliumai/outputs/deontology/full_test_split_v1/hendrycks_test_private_labels.jsonl}"

MORALREASON_SKILLS="${MORALREASON_SKILLS:-/scratch/fatimay4/consiliumai/outputs/deontology/moralreason_train_skills_gpt35_v1/moralreason_train_skills_gpt35.jsonl}"
MORALREASON_SOURCE_CSV="${MORALREASON_SOURCE_CSV:-/home/fatimay4/scratch/consiliumai/datasets/moral_reason_training/moralreason_train_deontological.csv}"

OUT_DIR="${OUT_DIR:-/scratch/fatimay4/consiliumai/outputs/deontology/full_retrieval_inputs_v1}"

for path in \
  "$SCRIPT" \
  "$HENDRYCKS_SKILLS" \
  "$HENDRYCKS_PRIVATE_LABELS" \
  "$MORALREASON_SKILLS" \
  "$MORALREASON_SOURCE_CSV"; do
  if [[ ! -f "$path" ]]; then
    echo "[ABORT] missing file: $path"
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

echo "[config] hendrycks_skills=$HENDRYCKS_SKILLS"
echo "[config] hendrycks_private_labels=$HENDRYCKS_PRIVATE_LABELS"
echo "[config] moralreason_skills=$MORALREASON_SKILLS"
echo "[config] moralreason_source_csv=$MORALREASON_SOURCE_CSV"
echo "[config] output=$OUT_DIR"

python -u "$SCRIPT" \
  --hendrycks-skills "$HENDRYCKS_SKILLS" \
  --hendrycks-private-labels "$HENDRYCKS_PRIVATE_LABELS" \
  --moralreason-skills "$MORALREASON_SKILLS" \
  --moralreason-source-csv "$MORALREASON_SOURCE_CSV" \
  --output-dir "$OUT_DIR" \
  --expected-hendrycks 3595 \
  --expected-moralreason 664
