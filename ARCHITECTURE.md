# Malware Diffusion Pipeline — Architecture

## Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         MALICIA DATASET                                    │
│                                                                            │
│  malicia/zeroaccess/*.asm.txt   malicia/zbot/*.asm.txt   malicia/...       │
│  [push, mov, sub, call, ...]    [xor, mov, push, ...]                      │
└──────────────────┬─────────────────────────────┬──────────────────────────┘
                   │                             │
         (one corpus per family)       (one corpus per family)
                   │                             │
                   ▼                             ▼
┌──────────────────────────────┐   ┌─────────────────────────────┐
│  Word2Vec (zeroaccess)       │   │  Word2Vec (zbot)            │
│  Trained on all opcode seqs  │   │  Trained on all opcode seqs │
│  in this family only         │   │  in this family only        │
│  vector_size = 104           │   │  vector_size = 104          │
└──────────────┬───────────────┘   └──────────────┬──────────────┘
               │                                   │
        ┌──────▼──────┐                     ┌──────▼──────┐
        │ File embed  │  for each .asm.txt  │ File embed  │
        │ = mean of   │ ──────────────────▶ │ = mean of   │
        │ opcode vecs │                     │ opcode vecs │
        │ shape: (104)│                     │ shape: (104)│
        └──────┬──────┘                     └──────┬──────┘
               │                                   │
        scale to [-1, 1]                    scale to [-1, 1]
               │                                   │
               ▼                                   ▼
        ┌─────────────┐                     ┌─────────────┐
        │  Embeddings │                     │  Embeddings │
        │  (N, 104)   │                     │  (M, 104)   │
        └──────┬──────┘                     └──────┬──────┘
               │                                   │
               └──────────────┬────────────────────┘
                              │  (train one diffusion model per family)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODIFIED DDPM (per family)                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     FORWARD DIFFUSION                               │   │
│  │                                                                     │   │
│  │  x₀ (real embed)  →  q(x_t | x₀)  →  x_{T/2}  (noisy embed)       │   │
│  │                                                                     │   │
│  │  x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε,   ε ~ N(0,I)                    │   │
│  │                                                                     │   │
│  │  KEY: only diffuses to T/2 (not full T) — paper modification        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    1D U-NET (noise predictor)                       │   │
│  │                                                                     │   │
│  │  Input: x_t (1, 104) + timestep t                                   │   │
│  │                                                                     │   │
│  │  ┌──────────┐  Conv1D(ch)  ┌──────────────┐  Conv1D(ch)  ┌───────┐ │   │
│  │  │ in_conv  │─────────────▶│  ResBlock1D  │─────────────▶│  enc1 │ │   │
│  │  │ (1→ch)   │              │  + time_emb  │              │ (ch)  │ │   │
│  │  └──────────┘              └──────────────┘              └───┬───┘ │   │
│  │                                                              │     │   │
│  │                                                  stride-2 down     │   │
│  │                                                              ▼     │   │
│  │                                                          ┌───────┐ │   │
│  │                                                          │  enc2 │ │   │
│  │                                                          │ (2ch) │ │   │
│  │                                                          └───┬───┘ │   │
│  │                                                              │     │   │
│  │                                                          ┌───▼───┐ │   │
│  │                                                          │  mid  │ │   │
│  │                                                          │ (2ch) │ │   │
│  │                                                          └───┬───┘ │   │
│  │                                                              │     │   │
│  │                                                 ConvTranspose1D up │   │
│  │                                                              ▼     │   │
│  │                ┌──────────────┐  skip conn (+ enc1)     ┌───────┐ │   │
│  │                │   out_conv   │◀────────────────────────│  dec1 │ │   │
│  │                │   (ch→1)     │                          │ (ch)  │ │   │
│  │                └──────┬───────┘                          └───────┘ │   │
│  │                       │                                             │   │
│  │               predicted noise ε̂(x_t, t)                           │   │
│  │                                                                     │   │
│  │  Loss = MSE(ε̂, ε)                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    REVERSE DIFFUSION (generation)                   │   │
│  │                                                                     │   │
│  │  x_{T/2} = forward_diffusion(real_sample, T/2)  // Algorithm 1      │   │
│  │  for t = T/2, T/2-1, ..., 1:                                        │   │
│  │      ε̂ = U-Net(x_t, t)                                              │   │
│  │      x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) · ε̂) + σ_t · z   │   │
│  │  return x_0  (synthetic embedding)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │  Synthetic Embeddings  │
                 │  shape: (N_synth, 104) │
                 └───────────┬────────────┘
                             │
               ┌─────────────┴────────────────┐
               ▼                              ▼
  ┌────────────────────────┐     ┌────────────────────────┐
  │  Binary Classification │     │   Cosine Similarity    │
  │  (real vs synthetic)   │     │  cross-comparison      │
  │  RF / SVM / MLP        │     │  goal: match baseline  │
  │  goal: F1 ≈ 0.5        │     │  (real vs real)        │
  └────────────────────────┘     └────────────────────────┘
               │
               ▼
  ┌────────────────────────┐
  │    t-SNE visualization │
  │  real (green) vs       │
  │  synthetic (orange)    │
  └────────────────────────┘
```

## Key Design Decisions (from paper)

| Decision | Value | Reason |
|---|---|---|
| Embedding dim | 104 | Best F1 vs 64/128 in ablation |
| Word2Vec per family | Separate model each | Opcodes have different co-occurrence patterns per family |
| Normalization range | [-1, 1] | Matches Tanh output range |
| Forward diffusion | T/2 only | Full T caused reconstruction failure on 1D data |
| U-Net layers | Conv1D, Dropout1d | Data is 1D (embedding vector), not 2D image |
| Betas | linear 1e-4 → 0.02 | Standard DDPM schedule |

## File Structure

```
diffusion-proj/
├── malicia/                       # Dataset (opcodes per family, gitignored)
│   ├── zeroaccess/*.asm.txt
│   ├── winwebsec/*.asm.txt
│   └── ...
├── mdiff/                         # Python package
│   ├── data/                      # Data loading + Word2Vec + vocab
│   │   ├── loader.py              #   load_family_opcodes
│   │   ├── embeddings.py          #   Word2Vec, file_embedding, scale_to_range
│   │   ├── vocab.py               #   Vocabulary, OpcodeDataset(Chunked)
│   │   └── preprocess.py          #   binaries → opcode .txt files
│   ├── models/
│   │   ├── ddpm/                  # Continuous DDPM variant
│   │   │   ├── model.py           #   MalwareDiffusion + 1D U-Net
│   │   │   └── runner.py          #   train_families / generate_family
│   │   └── d3pm/                  # Discrete D3PM variant
│   │       ├── model.py           #   AbsorbingD3PM
│   │       └── runner.py          #   train_families / generate_family
│   ├── paths.py                   # Per-family output layout (DDPM_*, D3PM_*)
│   ├── train.py                   # Unified CLI: --variant {ddpm,d3pm}
│   ├── generate.py                # Unified CLI
│   ├── evaluate.py                # All 6 paper metrics (embedding-level)
│   └── seq_evaluate.py            # n-gram, edit distance, freq-KL (token-level)
├── tests/
│   ├── test_d3pm.py               # < 30 s (no malicia data needed)
│   └── test_pipeline.py           # ~5 min, hits a few small families
├── scripts/
│   ├── HPC.md
│   ├── hpc_train_continuous.slurm
│   └── hpc_train_d3pm.slurm
├── checkpoints/<family>/          # Per-family artifacts (see paths.py)
├── synthetic/<family>/
└── eval_results/<family>/
```

### Adding a new diffusion variant

1. Create `mdiff/models/<variant>/model.py` (the `nn.Module`).
2. Create `mdiff/models/<variant>/runner.py` exposing `train_families(args, device)` and `generate_family(args, device)`. Save artifacts under `paths.family_ckpt_dir(...)` and `paths.family_synth_dir(...)`.
3. Add `<variant>` to the `--variant` choices in `mdiff/train.py`, `mdiff/generate.py`, and `mdiff/evaluate.py`, and dispatch to the new runner.

## Training a Full Run

```bash
# Activate venv
source .venv312/bin/activate

# Train continuous DDPM on multiple families
python -m mdiff.train --variant ddpm --families zeroaccess zbot winwebsec --epochs 200

# Train discrete D3PM on a single family
python -m mdiff.train --variant d3pm --families zeroaccess --epochs 100

# Generate synthetic samples
python -m mdiff.generate --variant ddpm --family zeroaccess --n 500
python -m mdiff.generate --variant d3pm --family zeroaccess --n 200
```
