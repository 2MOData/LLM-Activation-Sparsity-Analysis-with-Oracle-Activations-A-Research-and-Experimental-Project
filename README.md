# LLM Activation Sparsity Analysis with Oracle Activations: A Research and Experimental Project

> What does an LLM lose when its activations are made sparse? We ask an *Activation Oracle*,
> an LLM that can read another LLM's activations, to tell us in natural language.

PROCOM project by Momo, Raph, Max and Léo.

## The idea

**Activation sparsity** methods (TEAL, Deja Vu-style predictors, post-computation cutoff, quantization...)
speed up inference by zeroing out part of the LLM's activations. We know how to measure their overall
cost (perplexity, accuracy), but not **which information** disappears, nor **in which layers**.

**Activation Oracles** (Karvonen et al., 2025, [arXiv:2512.15674](https://arxiv.org/abs/2512.15674),
following LatentQA) are LLMs trained to read a model's activations and answer questions about
what they contain ("What topic is the text about?", "What is the answer to the calculation?"...).

Our idea: **query the oracle on the dense activations, then on the sparse activations**, and look at
what changes. If the oracle can no longer recover a piece of information after sparsification, the
method has erased it.

## How we go about it

```
prompt ──► target LLM ──► dense activations ─────────────► oracle ──► dense answer
                     └──► sparsity method ──► sparse activations ──► oracle ──► sparse answer
                                                                     comparison ◄──┘
```

1. **Dataset**: prompts grouped by theme (facts, context, calculation, predictable text...) and, for
   each theme, questions to ask the oracle.
2. **Extraction**: PyTorch hooks that capture the target LLM's activations, layer by layer.
3. **Sparsification**: several families of methods so as not to depend on a single one
   (predictors, TEAL-style post-computation cutoff, quantization...).
4. **Querying the oracle** on the dense and sparse versions, and checking its reliability on sparse
   activations, which it never saw during training.
5. **Analysis**: layer × theme heatmaps of the information loss, compared with standard measures
   (KL divergence between dense and sparse outputs, activation patching).

**Research questions**
- Do all sparsity methods erase the same kind of information?
- Which layers can be made sparse without the oracle noticing any difference?
- Does the loss seen by the oracle follow the standard metrics (KL, perplexity), or does it reveal something else?

A lack of correlation between the oracle and the standard metrics would also be a result.


## Installation (once)

The environment is managed by **[uv](https://docs.astral.sh/uv/)**: `pyproject.toml` lists the dependencies,
`uv.lock` pins every version (including indirect dependencies) and `.python-version` sets Python 3.11.
So everyone has exactly the same environment.

1. Install uv:
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Clone and install:
   ```bash
   git clone <repo-url>
   cd <project-name>
   uv sync          # creates .venv/ with the right Python version and the exact versions from the lock
   uv run python scripts/check_env.py
   uv run pre-commit install   # automatic formatting before each commit
   ```

uv downloads Python 3.11 itself if needed. No need to activate the venv: prefix commands with `uv run`
(or `source .venv/bin/activate` if you prefer).

## torch and CUDA

- **Linux / Windows**: torch is installed from the PyTorch **CUDA 12.6** index (see `[tool.uv.sources]`
  in `pyproject.toml`). Works with a recent NVIDIA driver (≥ 560); without a GPU, torch still runs on CPU.
- **macOS**: standard torch (CPU + MPS on Apple silicon).
- To change the CUDA version (e.g. `cu128` on a school server), change the index URL **in the
  repo**, rerun `uv lock`, and commit: never run `pip install torch` by hand.

## Team rules

- Adding a library: `uv add <package>` (or `uv add --dev <package>` for a tool), then **commit
  `pyproject.toml` and `uv.lock` together**.
- After a `git pull` that changes `uv.lock`: `uv sync`.
- Never use `pip install` in the project (except on Colab).
- Data, activations and model weights do not go into Git (see `.gitignore` and `data/README.md`).
- One branch per task, then a pull request to `main`.

## Structure

```
src/procom/
  oracle/        # loading and querying the Activation Oracle (Max, Léo)
  activations/   # activation extraction via PyTorch hooks (Max, Léo)
  dataset/       # prompt themes and questions asked to the oracle (Raph, Momo)
  sparsity/      # sparsity methods: predictors, post-computation cutoff, quantization (Raph, Momo)
scripts/         # runnable scripts (uv run python scripts/...)
notebooks/       # explorations (outputs stripped automatically by nbstripout)
configs/         # YAML config files for experiments
tests/           # uv run pytest
data/            # local only, not versioned
```

## Useful commands

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # formatting
uv run jupyter lab       # notebooks (after `uv add --dev jupyterlab`)
```
