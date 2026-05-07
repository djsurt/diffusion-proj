# HPC runbook (SJSU HPC, SLURM)

Everything HPC-specific you need to do for this project, in one place. Two SLURM jobs:

| Script | What it trains | Typical wall time on 1 GPU |
|---|---|---|
| `scripts/hpc_train_continuous.slurm` | Continuous DDPM (paper baseline) on Word2Vec embeddings | 15–30 min for 3 families |
| `scripts/hpc_train_d3pm.slurm`       | Discrete D3PM (absorbing-state) on opcode tokens | 30–60 min for 1 family |

Both jobs are end-to-end: train → generate → evaluate. You should not need to manually
chain steps after submitting.

---

## 0. Quick reference

```bash
# Submit
sbatch scripts/hpc_train_continuous.slurm
sbatch scripts/hpc_train_d3pm.slurm

# Override params via env vars
EPOCHS=50 sbatch scripts/hpc_train_d3pm.slurm
FAMILIES="zeroaccess winwebsec" EPOCHS=300 sbatch scripts/hpc_train_continuous.slurm

# Watch
squeue -u $USER
tail -f logs/d3pm_train_<jobid>.out

# Kill
scancel <jobid>
```

---

## 1. One-time setup

### 1a. Transfer the project to HPC

From your **laptop**:

```bash
# Code (fast — excludes virtual envs and generated artifacts)
rsync -av --progress \
  --exclude='.venv*' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='checkpoints/' --exclude='synthetic/' --exclude='eval_results/' \
  --exclude='memory/' --exclude='.claude/' --exclude='logs/' \
  /Users/dhananjaysurti/CS271/diffusion-proj/ \
  <user>@coe-hpc.sjsu.edu:~/diffusion-proj/

# Dataset (large — only needed once, gitignored locally)
rsync -av --progress \
  /Users/dhananjaysurti/CS271/diffusion-proj/malicia/ \
  <user>@coe-hpc.sjsu.edu:~/diffusion-proj/malicia/
```

### 1b. Bootstrap a Python venv on the HPC

```bash
ssh <user>@coe-hpc.sjsu.edu
cd ~/diffusion-proj

module purge
module load python/3.12 || module load python/3.11 || module load python/3.10

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1c. Verify torch sees the GPU

Login nodes typically don't have GPUs, so grab an interactive GPU shell to test:

```bash
srun -p gpu --gres=gpu:1 --pty bash
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit
```

If `cuda.is_available()` is `False`, the bundled CUDA runtime in the pip wheel doesn't
match the cluster's driver. Reinstall a wheel that does:

```bash
# example: CUDA 11.8 (check the cluster's driver via `nvidia-smi`)
pip install --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## 2. Submitting jobs

### 2a. Continuous DDPM (paper baseline)

```bash
mkdir -p logs
sbatch scripts/hpc_train_continuous.slurm
```

Defaults: 3 families (zeroaccess, winwebsec, zbot), 200 epochs, 500 synth samples each.
Override with env vars:

```bash
EPOCHS=300 N_GEN=1000 sbatch scripts/hpc_train_continuous.slurm
FAMILIES="zeroaccess" sbatch scripts/hpc_train_continuous.slurm
FAMILIES="cridex zbot zeroaccess winwebsec" sbatch scripts/hpc_train_continuous.slurm
```

What lands in your repo when it finishes (per-family layout):

```
checkpoints/<family>/
  ddpm.pt                  # trained DDPM weights
  ddpm_embeddings.npy      # real W2V embeddings (DDPM coordinate system)
  ddpm_losses.json
synthetic/<family>/
  ddpm.npy                 # generated embeddings
eval_results/
  ddpm_full_report.json    # all 6 metrics, multi-family
  <family>/ddpm_tsne.png   # per-family t-SNE
```

### 2b. Discrete D3PM

```bash
sbatch scripts/hpc_train_d3pm.slurm
```

Defaults: zeroaccess, 30 epochs, 200 synth files. Override:

```bash
EPOCHS=50 sbatch scripts/hpc_train_d3pm.slurm
FAMILY=winwebsec EPOCHS=30 sbatch scripts/hpc_train_d3pm.slurm
```

What lands in your repo:

```
checkpoints/<family>/
  d3pm.pt                          # trained D3PM weights
  d3pm_vocab.pkl                   # opcode↔idx
  d3pm_losses.json
  d3pm_w2v.pkl                     # W2V used by D3PM (separate from continuous one)
  d3pm_real_embeddings.npy         # real ref in D3PM W2V space
synthetic/<family>/
  d3pm.npy                         # synth embeddings (same W2V space as real ref)
  d3pm_sequences/seq_*.txt         # raw opcode token sequences
eval_results/
  d3pm_full_report.json            # all 6 metrics
  <family>/
    d3pm_tsne.png
    seq_eval.json                  # n-gram, edit distance, freq-KL
```

The D3PM slurm script also runs `mdiff.evaluate --compare-against` against a
previously-saved DDPM report (if `eval_results/ddpm_full_report.json` exists),
giving you side-by-side per-family numbers in the SLURM log.

### 2c. What if I want continuous DDPM AND D3PM in the SAME W2V space?

The two scripts maintain separate W2V models on purpose (the DDPM W2V is the
paper baseline; the D3PM W2V is freshly trained from the generated sequences).
For a true side-by-side eval against one shared real reference, run
`hpc_train_d3pm.slurm` first (it builds `checkpoints/<family>/d3pm_w2v.pkl`),
then write a small wrapper that reuses that W2V to recompute the continuous
DDPM's embeddings and retrain it on those. Not currently automated.

---

## 3. Monitoring jobs

```bash
squeue -u $USER                     # all your queued/running jobs
squeue -j <jobid>                   # one job
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS

tail -f logs/d3pm_train_<jobid>.out
tail -f logs/d3pm_train_<jobid>.err

scancel <jobid>                     # kill
scancel -u $USER                    # kill ALL your jobs (careful)
```

GPU utilisation on the running node:

```bash
ssh <node-name>                     # node name from squeue's NODELIST column
nvidia-smi
```

---

## 4. Pulling results back to your laptop

After a job finishes, from your **laptop**:

```bash
rsync -av --progress \
  <user>@coe-hpc.sjsu.edu:~/diffusion-proj/checkpoints/ ./checkpoints/
rsync -av --progress \
  <user>@coe-hpc.sjsu.edu:~/diffusion-proj/synthetic/ ./synthetic/
rsync -av --progress \
  <user>@coe-hpc.sjsu.edu:~/diffusion-proj/eval_results/ ./eval_results/

# (Optional) the SLURM logs
rsync -av --progress \
  <user>@coe-hpc.sjsu.edu:~/diffusion-proj/logs/ ./logs/
```

If you only want one family:

```bash
rsync -av --progress \
  <user>@coe-hpc.sjsu.edu:~/diffusion-proj/checkpoints/zeroaccess/ \
  ./checkpoints/zeroaccess/
```

---

## 5. Cluster-specific knobs

These are placeholders in both `.slurm` files — confirm them for your cluster before
the first submission:

| `#SBATCH` directive | Default | How to verify |
|---|---|---|
| `--partition=gpu` | `gpu` | `sinfo -s` — pick whichever lists GPU nodes |
| `--gres=gpu:1` | 1 GPU | `sinfo -o "%P %G"` shows what GRES are available |
| `--time=06:00:00` (D3PM) | 6 hr | bump if running >50 epochs or full vocab |
| `--time=02:00:00` (continuous) | 2 hr | usually finishes in <30 min, padding is generous |
| `--mem=32G` (D3PM) / `16G` (continuous) | — | bump if you OOM |
| `--cpus-per-task=8` (D3PM) / `4` (continuous) | — | gensim W2V parallelises across these |
| `module load python/3.12` | py 3.12 | `module avail python` — drop to 3.11/3.10 if 3.12 absent |
| `module load cuda` | optional | `module avail cuda` — safe to drop the line if torch's bundled CUDA works |

---

## 6. Troubleshooting

**Job pending forever** → `squeue -u $USER` shows `(Resources)` or `(Priority)`.
The GPU partition is busy. Check queue depth: `squeue -p gpu | wc -l`.

**`torch.cuda.is_available() == False` inside the job** → the wheel's CUDA runtime
doesn't match the node's driver. See §1c — install a matching wheel.

**Out-of-memory (OOM)** in the SLURM err log → bump `--mem`. For D3PM on a family
with very long files, also consider lowering `--batch` (override via `BATCH=8 sbatch ...`
once we wire it through, or edit the slurm file).

**Walltime exceeded** (`TIMEOUT` state in `sacct`) → bump `--time` and resubmit.
Training is checkpoint-free right now (no resume from partial epoch); a timeout
loses the run.

**`module: command not found`** → some clusters use `Lmod`, others `Environment
Modules`, both expose `module`. If your shell strictly doesn't, add
`source /etc/profile.d/modules.sh` before the `module purge` line.

**Permission denied on `.slurm` script** → `chmod +x scripts/*.slurm` once after
the first rsync.

**Logs go to `slurm-<jobid>.out` instead of `logs/`** → the `logs/` dir didn't
exist when sbatch submitted. Run `mkdir -p logs` in your project root before the
first sbatch.

---

## 7. Things I am NOT doing on HPC (intentional)

- Running unit tests (`tests/test_d3pm.py`, `tests/test_pipeline.py`) — fast on
  laptop, no need to burn GPU hours
- Pre-processing (`mdiff.data.preprocess`) — already run; only re-run if Malicia changes
- The interactive `t-SNE` viewer — generates static `.png`s in `eval_results/<family>/`
  instead, fetched via rsync
