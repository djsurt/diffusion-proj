#!/bin/bash
#SBATCH --job-name=d3pm-zeroaccess
#SBATCH --partition=gpu                 # SJSU HPC GPU partition; change if your cluster uses a different name (e.g. `condo`, `gpuq`)
#SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00                 # walltime: 6 hr (~2x estimated; covers 30-50 epochs comfortably)
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

# ── 1. Load modules ─────────────────────────────────────────────────────────
# SJSU HPC: adjust these to whatever `module avail python` and `module avail cuda` show.
module purge
module load python/3.12 || module load python/3.11 || module load python/3.10
module load cuda || true                # only needed if torch wheels can't bring their own runtime

# ── 2. Activate venv (created once via HPC_SETUP.md) ────────────────────────
source .venv/bin/activate
python -c "import torch; print(f'torch={torch.__version__}, cuda available={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

mkdir -p logs checkpoints synthetic eval_results

# ── 3. Train D3PM (chunked, full data) ──────────────────────────────────────
FAMILY=${FAMILY:-zeroaccess}
EPOCHS=${EPOCHS:-30}

echo "[$(date)] Training D3PM on '$FAMILY' for $EPOCHS epochs..."
python src/d3pm_train.py \
    --family "$FAMILY" \
    --epochs "$EPOCHS" \
    --T 500 \
    --max-len 512 \
    --batch 32 \
    --d-model 128 \
    --nhead 4 \
    --layers 4 \
    --dim-ff 512 \
    --lambda-ce 0.1 \
    --lr 1e-3

# ── 4. Generate synthetic + build shared W2V real reference ────────────────
echo "[$(date)] Generating synthetic sequences..."
python src/d3pm_generate.py \
    --family "$FAMILY" \
    --n 200 \
    --T 500 \
    --max-len 512 \
    --batch 64

# ── 5. Side-by-side compare against existing continuous DDPM eval ──────────
echo "[$(date)] Running comparison..."
python src/d3pm_compare.py --family "$FAMILY"

# ── 6. Sequence-level eval (n-gram, edit distance, freq-KL) ────────────────
echo "[$(date)] Running sequence-level eval..."
python src/d3pm_evaluate.py --family "$FAMILY"

echo "=================================================="
echo "Finished at:   $(date)"
echo "=================================================="
