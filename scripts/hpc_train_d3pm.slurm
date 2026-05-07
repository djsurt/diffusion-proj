#!/bin/bash
#SBATCH --job-name=d3pm-zeroaccess
#SBATCH --partition=gpu                 # SJSU HPC GPU partition; change if your cluster uses a different name
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/d3pm_train_%j.out
#SBATCH --error=logs/d3pm_train_%j.err

set -euo pipefail

echo "=================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started at:    $(date)"
echo "Working dir:   $(pwd)"
echo "=================================================="

# ── 1. Modules ──────────────────────────────────────────────────────────────
module purge
module load python/3.12 || module load python/3.11 || module load python/3.10
module load cuda || true

# ── 2. Activate venv (created once via scripts/HPC.md) ──────────────────────
source .venv/bin/activate
python -c "import torch; print(f'torch={torch.__version__}, cuda available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

mkdir -p logs checkpoints synthetic eval_results

# ── 3. Train D3PM (chunked, full data) ──────────────────────────────────────
# Knobs (defaults match mdiff.train CLI defaults — override via env vars):
#   FAMILY=zeroaccess EPOCHS=30
FAMILY=${FAMILY:-zeroaccess}
EPOCHS=${EPOCHS:-30}
N_GEN=${N_GEN:-200}

echo "[$(date)] Training D3PM on '$FAMILY' for $EPOCHS epochs..."
python -m mdiff.train \
    --variant d3pm \
    --families "$FAMILY" \
    --epochs "$EPOCHS"

# ── 4. Generate synthetic + build shared W2V real reference ────────────────
echo "[$(date)] Generating $N_GEN synthetic samples for '$FAMILY'..."
python -m mdiff.generate \
    --variant d3pm \
    --family "$FAMILY" \
    --n "$N_GEN"

# ── 5. Embedding-level eval (with side-by-side vs continuous DDPM, if present)
echo "[$(date)] Embedding-level eval (D3PM)..."
COMPARE_FLAG=""
if [ -f eval_results/ddpm_full_report.json ]; then
    COMPARE_FLAG="--compare-against eval_results/ddpm_full_report.json"
fi
python -m mdiff.evaluate \
    --variant d3pm \
    --families "$FAMILY" \
    $COMPARE_FLAG

# ── 6. Sequence-level eval (n-gram, edit distance, freq-KL) ────────────────
echo "[$(date)] Sequence-level eval..."
python -m mdiff.seq_evaluate --family "$FAMILY"

echo "=================================================="
echo "Finished at:   $(date)"
echo "=================================================="
