# Running experiments on the school GPU cluster (Slurm)

What we learned while getting the AObench layer sweep (`scripts/aobench_layer_sweep.py`) to run on
the cluster. Follow it for the next experiment and it should work on the first try.

## The cluster in short

- **Login node** (where VS Code Remote-SSH puts you): shared, no GPU. Edit, install, download,
  submit jobs there. Never run a model on it.
- **Our resources**: Slurm account `brain`, partition `Brain_GPU`, QOS `high` or `low`. The only
  working node is `sl-mee-br-501`, with **4 × A100 80 GB** (driver 570, fine for our CUDA 12.6
  torch). The H100/H200 of `Mee_Global_GPU` need a `mee_*` or `global_externe` QOS that we do not
  have ("Invalid account or account/partition combination").
- **No time limit** on `Brain_GPU`. Jobs can be preempted, and Slurm then puts them back in the
  queue by itself (`PreemptMode=requeue`): scripts must be able to resume (see below).
- We submit with `--qos=low` so as not to take priority over the other users of the node.
- The node has internet access (Hugging Face works from a job).

## The two storage traps

1. **The home folder (`/homes/$USER`) has a 20 GB quota.** Model weights (Qwen3-8B 16 GB,
   Qwen3-32B 65 GB) and packages (~6 GB) do not fit.
2. **Jobs cannot read the home folder at all** ("Permission denied": it is a Kerberos-protected
   network disk and a job carries no ticket). So `uv`, the Python it installs, and anything a job
   reads or writes must not be in `~`. Before we fixed this, the job failed with `uv: command not
   found`, and `.venv/bin/python` pointed into `~/.local/share/uv/python`, which would have failed
   next.

Everything therefore lives on the shared `/Brain` disk, which both the login node and the jobs see
(it is 99% full overall, ~400 GB free: delete big files when the project ends):

```
/Brain/private/ProCom-AO/
  tools/bin/        uv and uvx
  tools/python/     the Python that .venv points to
  cache/uv/         uv package cache
  cache/huggingface/  model weights (HF_HOME)
  jobhome/$USER/    HOME given to jobs (matplotlib, torch... write there)
  LLM-Activation-.../   the repo, with its .venv/
```

[`cluster_env.sh`](cluster_env.sh) sets these paths. It is sourced by `~/.bashrc` on the login
node **and** by every job, so both always agree.

## First-time setup (once per person)

```bash
cd /Brain/private/ProCom-AO/LLM-Activation-Sparsity-Analysis-with-Oracle-Activations-A-Research-and-Experimental-Project
# 1. uv, on /Brain (skip if tools/bin/uv already exists: it is shared)
mkdir -p /Brain/private/ProCom-AO/tools/bin
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/Brain/private/ProCom-AO/tools/bin UV_NO_MODIFY_PATH=1 sh
# 2. add this line at the end of ~/.bashrc, then `source ~/.bashrc`
source /Brain/private/ProCom-AO/LLM-Activation-Sparsity-Analysis-with-Oracle-Activations-A-Research-and-Experimental-Project/scripts/slurm/cluster_env.sh
# 3. the environment
uv sync
ls -l .venv/bin/python      # must point to /Brain/..., never to /homes/...
mkdir -p logs               # Slurm does not create the log folder; without it the job dies silently
```

If `.venv/bin/python` points to `/homes`, run `rm -rf .venv && uv sync` with `cluster_env.sh` loaded.

## Checklist for a new experiment

1. **Write a script, not a notebook.** Slurm runs jobs unattended. Take
   `scripts/aobench_layer_sweep.py` as a model: command-line options (`argparse`), a small-size
   option for testing (`--limit`), and **save each unit of work as soon as it is done and skip it
   on restart**, so a preempted or crashed job resumes instead of starting over. Set
   `MPLBACKEND=Agg` (no display on the nodes) and print results to the log.
2. **Copy the job file** [`aobench_layer_sweep.sbatch`](aobench_layer_sweep.sbatch) and change
   only the job name and the command. Keep `source scripts/slurm/cluster_env.sh`, the `HOME`
   override and `python -u` (unbuffered, so the log shows progress live). Ask for one GPU unless
   the code really uses more (a GPU you hold is unavailable to everyone else).
3. **Download models on the login node first** (the AObench script has `--prepare` for this), so
   GPU time is not spent downloading.
4. **Test the environment on the node without a GPU** (starts at once, even when all GPUs are busy):
   ```bash
   srun -A brain --qos=low -p Brain_GPU --time=00:05:00 bash -c \
     'source scripts/slurm/cluster_env.sh && export HOME=/Brain/private/ProCom-AO/jobhome/$USER && mkdir -p $HOME && uv run python scripts/check_env.py'
   ```
5. **Run a tiny version first** (a few examples, a few configurations), with a short time limit so
   Slurm can fit it in sooner:
   ```bash
   sbatch --qos=low --time=04:00:00 scripts/slurm/<job>.sbatch <small-size options>
   ```
6. **Use its timings to size the real run**, then submit without the small-size options. Options
   given to `sbatch` on the command line override the `#SBATCH` lines of the file.

## Following a job

```bash
squeue -u $USER                       # R = running, PD = waiting (last column says why)
tail -f logs/<job-name>-<jobid>.out   # live log
sacct -j <jobid> -D --format=JobID,State%25,Start,Elapsed,ExitCode   # what happened, restarts included
scontrol show job <jobid> | grep -E "JobState|Reason|Restarts"       # Restarts>0: preempted and requeued
scancel <jobid>
```
`CANCELLED by <uid>` in `sacct` tells who cancelled it (`getent passwd <uid>`).

## What to expect in the logs

- **Loading weights is slow** because `/Brain` is a network disk: ~3 min for Qwen3-8B, ~12 min for
  Qwen3-32B, at every job start.
- **Harmless messages**: `Failed to load ... torchao/_C_cutlass_90a / _C_mxfp8` (optional kernels
  for other GPUs), `unauthenticated requests to the HF Hub`, `torch_dtype is deprecated`,
  `register_constant() ... deprecated`, `generation flags are not valid ... temperature`.

## Useful facts

- Find GPUs and their use: `sinfo -N -p Brain_GPU -O "NodeList,Gres,GresUsed"`. GPU memory is not
  listed by Slurm: the first lines of our job log print `nvidia-smi` output.
- Disk: `df -h /Brain` (shared disk), `du -sh /Brain/private/ProCom-AO/cache/*`, `quota -s` (home).
- AObench timings on one A100 80 GB, 10 examples per eval: ~3 min of generation and ~2.5 min of
  judging (Qwen3-32B, bf16, 66 GB) per window.
