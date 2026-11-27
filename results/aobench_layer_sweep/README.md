# AObench layer-window sweep (Experiment 1)

Results of the setup experiment that checks at which depths the Activation Oracle
(`ceselder/qwen3-8b-ao-v3-best`, trained on layers 21-25 of Qwen3-8B) can still be trusted.
The oracle was fed 19 windows of 5 consecutive layers, from 13-17 to 31-35, and scored on
the 9 AObench evals that need no extra model. The full description and discussion are in
Section 4 of the paper.

**Outcome.** The score is flat within about one layer of the trained window, then decreases
gradually (about 0.01 per layer), and drops sharply only once layers 33-35 are in the window.
The rest of the project reads the oracle at three windows that share no layer:

| Window | AObench aggregate | Change vs 21-25 |
|---|---|---|
| 16-20 | 0.516 | -0.054 (-9%) |
| 21-25 (trained) | 0.570 | - |
| 26-30 | 0.532 | -0.038 (-7%) |

Selection rule: windows that share no layer with each other and lose at most 10% of the
trained window's aggregate score.

## How it was run

- Script: `scripts/aobench_layer_sweep.py` at the repository root (Slurm job in
  `scripts/slurm/`), also available as `notebooks/aobench_layer_window_sweep.ipynb`.
- One NVIDIA A100 80 GB: 6 h 21 min to generate the oracle's answers, 10 h 24 min to grade them.
- Settings are in `sweep_settings.json`. They differ from the original AObench in four ways:
  - judge: Qwen3-32B in bf16, run locally, instead of Claude Sonnet 4.6;
  - hallucination rubric: one added sentence (quoted in `sweep_settings.json`);
  - layer labels: the oracle's prompt names the layers it really receives (`prompt_layers="actual"`);
  - target activations: read with the oracle's LoRA disabled (`target_activations="base"`).

  Absolute scores are therefore **not** comparable with the AObench paper; only the
  comparison between windows is meaningful.
- Decoding is greedy everywhere, so each window was run once.

## Files

| File | Content |
|---|---|
| `results.csv` | Raw and normalized score of every (window, eval) pair, plus run time |
| `sweep_settings.json` | Settings of the run (oracle, AObench commit, judge, prompt change) |
| `stats_per_window.csv` | For every eval and sub-aggregate x window: score, 95% CI, change vs 21-25 with paired 95% CI, bootstrap p-value |
| `judge_audit_items.csv` | 237 judged answers from windows 13-17, 21-25 and 31-35, re-graded blind by a second LLM (Claude Opus 5.5) to check the Qwen3-32B judge: both labels side by side, with the oracle's answer and the judge's reasoning |
| `hallucination_correction.csv` | Hallucination scores corrected for the judge's wrong "obviously wrong" verdicts (24 re-graded per window) |
| `figures/exp1_aggregate.png` | Aggregate, judge-free and judged scores per window, with 95% CIs |
| `figures/exp1_per_eval.png` | Score of each eval per window (drawn by `plot_sweep` in `src/procom/oracle/aobench.py`) |
| `figures/exp1_hallucination_correction.png` | Hallucination: judge scores and corrected scores |
| `scripts/` | Analysis scripts (below) |

Sub-aggregates in `stats_per_window.csv`: `objective_only` = the 4 judge-free evals,
`judged_only` = the 5 judged evals, `aggregate_no_actsens_no_vague` = the 7 evals left
without the two that stay near their ceiling.

**Known issues with two judged evals**, see the paper for details:
- `activation_sensitivity` does not discriminate between windows: the judge labels almost every
  pair as meaningfully different, and AObench pairs 18 of its 45 answer pairs with the wrong text.
- `hallucination`: the judge over-calls "obviously wrong", more often in shallow windows.

## Raw outputs

The per-example outputs of every window (oracle answers, judge verdicts, AObench logs,
about 114 MB) are not versioned in Git. They are available at: TODO (add the link).

## Reproducing the statistics and figures

The scripts need `pandas`, `numpy` and `matplotlib`. Download the raw outputs so that the
`window_13-17/` ... `window_31-35/` folders sit in `raw/` next to this README, then run:

```bash
cd results/aobench_layer_sweep/scripts
python load.py           # reads ../raw/ (or a folder given as argument), writes *.pkl here
python stats.py          # cluster bootstrap over problems; rewrites ../stats_per_window.csv
python figures.py        # redraws ../figures/exp1_aggregate.png and exp1_hallucination_correction.png
python sample.py <eval> <n>   # draws a blind re-grading sample, as used for the judge audit
```

`figures.py` only needs the CSV files in this folder, so it runs without the raw outputs.
`hallucination_correction.csv` was computed from the re-graded verdicts in
`judge_audit_items.csv`, not by a script.
