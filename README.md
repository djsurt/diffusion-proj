# Malware Diffusion — Setup & Run Guide

Generates synthetic malware opcode sequences with two diffusion models: a
**continuous DDPM** (baseline, runs on Word2Vec embeddings) and a **discrete
D3PM** (absorbing-state, runs on tokenised opcodes).

- `COMMANDS.md` — quick command reference, every common task in one place.
- `ARCHITECTURE.md` — design notes and file structure.

This README covers running the project on a **fresh Linux machine with an
NVIDIA GPU** (e.g. RTX 3070).

## What ships in the repo (vs. what you provide / what gets created)

The repo is intentionally lean — code only, no data and no artifacts.

**In the repo (after `git clone`):**
```
mdiff/  tests/  scripts/  requirements.txt  ARCHITECTURE.md  README.md  .gitignore
```

**You provide (gitignored, must be on disk before running):**
- `malicia/` — the MALICIA opcode dataset, one folder per family (see "Dataset" below).

**Auto-created on first run (gitignored, do not pre-create):**
- `checkpoints/<family>/` — model weights, vocab, cached W2V, real-reference embeddings
- `synthetic/<family>/`   — generated opcode sequences and embeddings
- `eval_results/<family>/` — JSON reports and t-SNE PNGs

**Not needed to run:** `docs/` (paper PDFs) and `CLAUDE.md` are gitignored — they live on the original author's machine only.

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

# 2. Create venv (use .venv on Linux; the local macOS dev venv is .venv312)
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

The MALICIA dataset is **not** in the repo. Copy it (or symlink it) into the
project root before running anything:

```bash
cp -r /path/to/your/malicia ./malicia
# or:  ln -s /path/to/your/malicia malicia
```

Expected layout — one folder per family, one file per sample, one opcode per line:
```
malicia/
  zeroaccess/    # 1311 files, mean ~5700 opcodes each
  zbot/
  cridex/
  winwebsec/
  ...
```

Verify:
```bash
python -c "from mdiff.data import load_family_opcodes; \
  d = load_family_opcodes('malicia', families=['zeroaccess']); \
  print('files:', len(d['zeroaccess']), 'mean opcodes:', sum(map(len,d['zeroaccess']))//len(d['zeroaccess']))"
# Expected: files: 1311 mean opcodes: ~5700
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
# 1. Train
python -m mdiff.train --variant d3pm --families zeroaccess \
    --epochs 30 --max-len 2048 --batch 16

# 2. Generate synthetic sequences + W2V-aligned embeddings
python -m mdiff.generate --variant d3pm --family zeroaccess \
    --n 200 --max-len 2048 --batch 64

# 3. Sequence-level eval (n-gram overlap, edit distance, opcode-frequency KL)
python -m mdiff.seq_evaluate --family zeroaccess

# 4. Embedding-level eval — same W2V coordinate system for real & synth.
#    Optional --compare-against pulls in side-by-side numbers from a saved DDPM run.
python -m mdiff.evaluate --variant d3pm --families zeroaccess \
    --compare-against eval_results/ddpm_full_report.json
```

### Bumping max_len for better embedding quality

Real zeroaccess files average 5700 opcodes; only ~0.4 % fit in 2048 and ~36 %
fit in 4096. Going to `max_len=4096` covers more of each file in a single
synthesis pass (less stitching) and tends to improve embedding-level metrics:

```bash
python -m mdiff.train --variant d3pm --families zeroaccess \
    --epochs 30 --max-len 4096 --batch 4
python -m mdiff.generate --variant d3pm --family zeroaccess \
    --n 200 --max-len 4096 --batch 16
```

## Run the continuous DDPM baseline (optional)

```bash
python -m mdiff.train --variant ddpm --families zeroaccess --epochs 200
python -m mdiff.generate --variant ddpm --family zeroaccess --n 500
python -m mdiff.evaluate --variant ddpm --families zeroaccess
```

## Output layout

All artifacts for one family live under one directory in each of `checkpoints/`,
`synthetic/`, `eval_results/`:

```
checkpoints/
  zeroaccess/
    ddpm.pt                 ddpm_embeddings.npy        ddpm_losses.json
    d3pm.pt                 d3pm_vocab.pkl             d3pm_losses.json
    d3pm_w2v.pkl            d3pm_real_embeddings.npy

synthetic/
  zeroaccess/
    ddpm.npy                d3pm.npy                   d3pm_sequences/

eval_results/
  ddpm_full_report.json     d3pm_full_report.json      # multi-family rollups
  zeroaccess/
    ddpm_tsne.png           d3pm_tsne.png              report.json   seq_eval.json
```

## Adding a new diffusion variant

1. Add `mdiff/models/<variant>/model.py` with the `nn.Module`.
2. Add `mdiff/models/<variant>/runner.py` exposing
   `train_families(args, device)` and `generate_family(args, device)`.
   Save artifacts under `paths.family_ckpt_dir(...)` and `paths.family_synth_dir(...)`.
3. Wire `<variant>` into the `--variant` choices in `mdiff/train.py`,
   `mdiff/generate.py`, and `mdiff/evaluate.py`.

## Troubleshooting

- **OOM during training** → drop `--batch` (16 → 8 → 4), or `--max-len 1024`.
- **`Family 'X' not found`** → check `malicia/X/` exists and contains `.txt` files.
- **`Missing checkpoints/<family>/d3pm.pt`** during generate → run training first.
- **F1 ≈ 1.0 on embedding eval** → almost always means `--variant d3pm` was forgotten and `evaluate.py` is comparing across two different W2V models.
