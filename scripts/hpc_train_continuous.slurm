#!/bin/bash
#SBATCH --job-name=ddpm-continuous
#SBATCH --partition=gpu                 # SJSU HPC GPU partition; change if your cluster uses a different name
#SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00                 # walltime: 2 hr (continuous DDPM is fast — usually <30 min)
#SBATCH --output=logs/ddpm_continuous_%j.out
#SBATCH --error=logs/ddpm_continuous_%j.err

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

# ── 2. Activate venv ────────────────────────────────────────────────────────
source .venv/bin/activate
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

mkdir -p logs checkpoints synthetic eval_results

# ── 3. Train continuous DDPM ────────────────────────────────────────────────
# FAMILIES: space-separated list (e.g. "zeroaccess winwebsec zbot")
#           default trains the three large Malicia families
# EPOCHS:   default 200 (paper config)
# N_GEN:    number of synthetic samples per family (default 500)
FAMILIES=${FAMILIES:-"zeroaccess winwebsec zbot"}
EPOCHS=${EPOCHS:-200}
N_GEN=${N_GEN:-500}

echo "[$(date)] Training continuous DDPM for: $FAMILIES (epochs=$EPOCHS)"
python src/train.py \
    --malicia malicia/ \
    --out checkpoints/ \
    --families $FAMILIES \
    --epochs "$EPOCHS" \
    --batch 64 \
    --T 1000 \
    --lr 2e-4 \
    --embed-dim 104

# ── 4. Generate synthetic for each family ──────────────────────────────────
for FAMILY in $FAMILIES; do
    echo "[$(date)] Generating $N_GEN synthetic samples for '$FAMILY'..."
    python src/generate.py \
        --checkpoints checkpoints/ \
        --family "$FAMILY" \
        --n "$N_GEN" \
        --T 1000 \
        --embed-dim 104
done

# ── 5. Run all 6 paper metrics ──────────────────────────────────────────────
echo "[$(date)] Evaluating (binary clf, t-SNE, synth-only train, threshold-aug, fidelity, cosine)..."
python src/evaluate.py \
    --families $FAMILIES \
    --checkpoints checkpoints/ \
    --synthetic synthetic/ \
    --out eval_results/

echo "=================================================="
echo "Finished at:   $(date)"
echo "Report:        eval_results/full_report.json"
echo "=================================================="
