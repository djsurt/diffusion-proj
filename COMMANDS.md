# Commands — quick reference

Every common task, with the exact command. For setup, see `README.md`. For HPC,
see `scripts/HPC.md`. For architecture/design, see `ARCHITECTURE.md`.

All commands assume:
- cwd = repo root
- venv active: `source .venv312/bin/activate` (macOS dev) or `source .venv/bin/activate` (HPC)

## Sanity checks

```bash
# Fast unit tests (~1 s, no malicia data needed)
python tests/test_d3pm.py

# End-to-end pipeline test (~5 min, hits a few small malicia families)
python tests/test_pipeline.py

# Verify malicia is wired up
python -c "from mdiff.data import load_family_opcodes; \
  d = load_family_opcodes('malicia', families=['zeroaccess']); \
  print('files:', len(d['zeroaccess']))"

# Verify CLIs parse
python -m mdiff.train --help
python -m mdiff.generate --help
python -m mdiff.evaluate --help
python -m mdiff.seq_evaluate --help
```

## Continuous DDPM (paper baseline)

```bash
# Train one family
python -m mdiff.train --variant ddpm --families zeroaccess --epochs 200

# Train multiple families in one go
python -m mdiff.train --variant ddpm --families zeroaccess winwebsec zbot --epochs 200

# Generate (one family at a time)
python -m mdiff.generate --variant ddpm --family zeroaccess --n 500

# Evaluate (embedding-level: all 6 paper metrics)
python -m mdiff.evaluate --variant ddpm --families zeroaccess winwebsec zbot
# → eval_results/ddpm_full_report.json + eval_results/<family>/ddpm_tsne.png
```

## Discrete D3PM

```bash
# Train one family (defaults: epochs=100, T=500, max-len=2048, batch=8, lambda-ce=0.01)
python -m mdiff.train --variant d3pm --families zeroaccess

# Train multiple families (loops internally, one at a time)
python -m mdiff.train --variant d3pm --families zeroaccess winwebsec --epochs 30

# Generate (one family)
python -m mdiff.generate --variant d3pm --family zeroaccess --n 200

# Embedding-level evaluation
python -m mdiff.evaluate --variant d3pm --families zeroaccess

# Embedding-level eval, with side-by-side numbers from a saved DDPM run
python -m mdiff.evaluate --variant d3pm --families zeroaccess \
    --compare-against eval_results/ddpm_full_report.json

# Sequence-level evaluation (n-gram, edit distance, opcode-frequency KL)
python -m mdiff.seq_evaluate --family zeroaccess
```

## Common knobs

```bash
# Cap files per family (fast smoke test on real data)
python -m mdiff.train --variant ddpm --families zeroaccess --max-files 50 --epochs 10
python -m mdiff.train --variant d3pm --families zeroaccess --max-files 50 --epochs 5

# OOM during D3PM training? Drop batch or max-len
python -m mdiff.train --variant d3pm --families zeroaccess --batch 4 --max-len 1024

# Bigger context (better embedding metrics, more VRAM)
python -m mdiff.train --variant d3pm --families zeroaccess --max-len 4096 --batch 4
python -m mdiff.generate --variant d3pm --family zeroaccess --max-len 4096 --batch 16

# Force CPU (skip MPS/CUDA — useful for debugging)
CUDA_VISIBLE_DEVICES="" python -m mdiff.train --variant d3pm --families zeroaccess --epochs 1
```

## Adding a new malware family

1. Drop opcodes into `malicia/<family>/*.asm.txt` (one opcode per line). If you
   only have raw binaries, run preprocessing:
   ```bash
   python -m mdiff.data.preprocess --input samples/ --output malicia/
   ```
2. Train + generate + evaluate as above — pass `--families <family>` (or `--family` for generate / seq_evaluate).

## Adding a new diffusion variant

1. Create `mdiff/models/<variant>/model.py` with the `nn.Module`.
2. Create `mdiff/models/<variant>/runner.py` exposing:
   ```python
   def train_families(args, device): ...
   def generate_family(args, device): ...
   ```
   Save artifacts under `paths.family_ckpt_dir(args.checkpoints, family)` and `paths.family_synth_dir(args.synthetic, family)` using new constants in `mdiff/paths.py` (e.g. `<VARIANT>_MODEL = "<variant>.pt"`).
3. Add `<variant>` to the `--variant choices=[...]` list in:
   - `mdiff/train.py` (also add a defaults entry in `_apply_variant_defaults`)
   - `mdiff/generate.py` (same)
   - `mdiff/evaluate.py` (`_variant_paths`)

## HPC (SJSU SLURM)

```bash
mkdir -p logs

# Continuous DDPM (defaults: 3 families, 200 epochs, 500 synth)
sbatch scripts/hpc_train_continuous.slurm
FAMILIES="zeroaccess" EPOCHS=300 sbatch scripts/hpc_train_continuous.slurm

# Discrete D3PM (defaults: zeroaccess, 30 epochs, 200 synth)
sbatch scripts/hpc_train_d3pm.slurm
FAMILY=winwebsec EPOCHS=50 sbatch scripts/hpc_train_d3pm.slurm

# Watch
squeue -u $USER
tail -f logs/d3pm_train_<jobid>.out

# Pull results back to laptop
rsync -av <user>@coe-hpc.sjsu.edu:~/diffusion-proj/{checkpoints,synthetic,eval_results}/ ./
```

Full HPC setup, troubleshooting, and partition knobs: `scripts/HPC.md`.

## Where things land

```
checkpoints/<family>/
  ddpm.pt                  ddpm_embeddings.npy        ddpm_losses.json
  d3pm.pt                  d3pm_vocab.pkl             d3pm_losses.json
  d3pm_w2v.pkl             d3pm_real_embeddings.npy

synthetic/<family>/
  ddpm.npy                 d3pm.npy                   d3pm_sequences/

eval_results/
  ddpm_full_report.json    d3pm_full_report.json      # multi-family rollups
  <family>/
    ddpm_tsne.png          d3pm_tsne.png              report.json   seq_eval.json
```

Filenames and the per-family layout are defined in `mdiff/paths.py` — touch
that file if you want to rename or reshape outputs.
