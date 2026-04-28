# Malware Diffusion — Setup & Run Guide

Generates synthetic malware opcode sequences with two diffusion models: a
**continuous DDPM** (baseline, runs on Word2Vec embeddings) and a **discrete
D3PM** (absorbing-state, runs on tokenised opcodes). See `CLAUDE.md` for the
full design notes.

This README covers running the project on a **fresh Linux machine with an
NVIDIA GPU** (e.g. RTX 3070).

---

## Requirements

- Linux (any recent distro)
- NVIDIA GPU with CUDA 12.x driver — verify with `nvidia-smi`
- Python 3.10, 3.11, or 3.12 (3.12 recommended)
- ~10 GB free disk for venv + dataset + checkpoints

The 3070 has 8 GB VRAM, which is plenty for the default config (D3PM training
peaks around 3–4 GB at `max_len=2048`, `batch=16`).

## Setup

```bash
# 1. Clone
git clone <repo-url> diffusion-proj && cd diffusion-proj

# 2. Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip + install deps (default torch wheel ships CUDA 12.x runtime)
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify CUDA is visible to PyTorch
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
# Expected: CUDA: True | NVIDIA GeForce RTX 3070
```

If `torch.cuda.is_available()` is False, the installed torch wheel doesn't
match the host's CUDA driver. For older drivers, force the CUDA 11.8 wheel:
```bash
pip uninstall -y torch
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu118
```

## Dataset

Place the MALICIA opcode dataset at `malicia/<family>/<file>.txt`, one opcode
per line. The repo expects at least one family directory; `zeroaccess` is
the default target. A typical layout:
```
malicia/
  zeroaccess/    # 1311 files, mean ~5700 opcodes each
  zbot/
  cridex/
  winwebsec/
```

## Quick test (sanity check before a real run)

```bash
python tests/test_d3pm.py     # < 30 s, 30 unit tests
```

If all 30 pass, the install is good.

## Run the D3PM pipeline on zeroaccess

The 3070 sweet spot is `max_len=2048` + `batch=16`. The full 1311-file run
with 30 epochs takes roughly 30–60 min on a 3070.

```bash
# 1. Train (drop --max-files to use all 1311 zeroaccess files)
python src/d3pm_train.py \
    --family zeroaccess \
    --epochs 30 \
    --max-len 2048 \
    --batch 16 \
    --T 500 \
    --lambda-ce 0.01

# 2. Generate synthetic sequences + W2V-aligned embeddings
python src/d3pm_generate.py \
    --family zeroaccess \
    --n 200 \
    --max-len 2048 \
    --batch 64

# 3. Sequence-level eval (n-gram overlap, edit distance, opcode-frequency KL)
python src/d3pm_evaluate.py --family zeroaccess

# 4. Embedding-level eval — IMPORTANT: --variant d3pm
#    This reads {family}_d3pm_real_embeddings.npy and {family}_d3pm_synthetic.npy
#    so real and synthetic share the same W2V coordinate system.
python src/evaluate.py --families zeroaccess --variant d3pm
```

### Bumping max_len for better embedding quality

Real zeroaccess files average 5700 opcodes; only ~0.4 % fit in 2048 and ~36 %
fit in 4096. Going to `max_len=4096` covers more of each file in a single
synthesis pass (less stitching) and tends to improve embedding-level metrics:

```bash
# 4096 still fits on a 3070 with batch=4
python src/d3pm_train.py --family zeroaccess --epochs 30 \
    --max-len 4096 --batch 4 --T 500 --lambda-ce 0.01
python src/d3pm_generate.py --family zeroaccess --n 200 --max-len 4096 --batch 16
```

## Run the continuous DDPM baseline (optional)

```bash
python src/train.py --families zeroaccess --epochs 200
python src/generate.py --family zeroaccess --n 500
python src/evaluate.py --families zeroaccess --variant continuous
```

## Output layout

```
checkpoints/
  zeroaccess_d3pm.pt                 # trained D3PM weights
  zeroaccess_d3pm_vocab.pkl          # opcode↔idx map
  zeroaccess_d3pm_losses.json        # per-epoch loss
  zeroaccess_d3pm_real_embeddings.npy   # real refs in D3PM W2V space
  d3pm_w2v.pkl                       # cached W2V (per-family dict)

synthetic/
  zeroaccess_d3pm_synthetic.npy      # synth embeddings (same W2V space)
  zeroaccess_d3pm_sequences/         # synth opcode sequences as text

eval_results/
  full_report.json                   # all 6 metrics
  zeroaccess_tsne.png                # t-SNE plot
```

## Troubleshooting

- **OOM during training** → drop `--batch` (16 → 8 → 4), or `--max-len 1024`.
- **`Family 'X' not found`** → check `malicia/X/` exists and contains `.txt` files.
- **`Missing zeroaccess_d3pm.pt`** during generate → run training first; checkpoint name is derived from `--family`.
- **F1 ≈ 1.0 on embedding eval** → almost always means `--variant d3pm` was forgotten and `evaluate.py` is comparing across two different W2V models.
