"""AObench layer-window sweep, as a script.

Runs the same experiment as `notebooks/aobench_layer_window_sweep.ipynb`: the oracle was trained
on layers 21-25 of Qwen3-8B, and we measure its AObench score when it is fed another window of 5
consecutive layers. The two phases of the notebook are kept:

- generate: Qwen3-8B and the oracle answer every question, for every window (needs >= 24 GB);
- judge: an open-weights judge grades the five judged evals from the saved answers.

Every (window, eval) is saved as soon as it finishes, so rerunning the same command after a
crash or a Slurm time limit resumes where it stopped.

Examples:
    # check the pipeline: 2 windows, 10 examples per eval
    uv run python scripts/aobench_layer_sweep.py --starts 21 13 --limit 10
    # the full experiment, generation and judging in one job
    uv run python scripts/aobench_layer_sweep.py
    # only one phase (e.g. two jobs on different GPUs)
    uv run python scripts/aobench_layer_sweep.py --phase generate
    uv run python scripts/aobench_layer_sweep.py --phase judge
"""

from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # no display on the compute nodes

from procom.oracle import aobench  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--phase",
        choices=["all", "generate", "judge"],
        default="all",
        help="generate: Qwen3-8B + oracle; judge: grade the saved answers; all: both, in order",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="examples per eval (the notebook's QUICK mode is 10). Default: the full benchmark",
    )
    parser.add_argument(
        "--starts",
        type=int,
        nargs="+",
        default=None,
        help="explicit start layers of the windows (e.g. 21 13). Overrides --first/--last/--stride",
    )
    parser.add_argument("--first", type=int, default=13, help="first window start (default 13)")
    parser.add_argument("--last", type=int, default=31, help="last window start (default 31)")
    parser.add_argument("--stride", type=int, default=1, help="step between windows (default 1)")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--judge-4bit",
        action="store_true",
        help="load the judge in 4-bit (Qwen3-32B in bf16 needs about 65 GB)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="results folder. Default: outputs/aobench_window_sweep, with _quick<N> if --limit",
    )
    parser.add_argument("--aobench-dir", default="outputs/aobench", help="where AObench is cloned")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="only clone AObench and download the model weights, then exit (for a login node "
        "with internet, when the compute nodes have none)",
    )
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    for repo in (aobench.MODEL_NAME, aobench.ORACLE_LORA, args.judge_model):
        print("Downloading", repo)
        snapshot_download(repo)
    print("Done. Downloads are in the Hugging Face cache (HF_HOME).")


def main() -> None:
    args = parse_args()
    aobench.install_aobench(args.aobench_dir)
    if args.prepare:
        prepare(args)
        return

    import torch

    assert torch.cuda.is_available(), "No GPU visible: did the job request one (--gres=gpu:1)?"
    print("GPU:", torch.cuda.get_device_name(0))

    output_dir = args.output_dir or "outputs/aobench_window_sweep"
    if args.output_dir is None and args.limit is not None:
        output_dir += f"_quick{args.limit}"
    sample_limits = (
        {name: args.limit for name in aobench.QUICK_SAMPLE_LIMITS} if args.limit else None
    )
    trained_start = aobench.TRAINED_WINDOW[0]
    starts = (
        sorted(args.starts, key=lambda s: (abs(s - trained_start), s))
        if args.starts
        else aobench.window_schedule(args.first, args.last, args.stride)
    )
    print("Results in:", output_dir)
    print(f"{len(starts)} windows, in this order:", [f"{s}-{s + 4}" for s in starts])

    if args.phase in ("all", "generate"):
        model, tokenizer = aobench.load_model_and_tokenizer()
        print(f"{aobench.MODEL_NAME}: {model.config.num_hidden_layers} layers")
        print("Oracle:", aobench.ORACLE_LORA)
        aobench.run_sweep(
            model,
            tokenizer,
            starts=starts,
            output_dir=output_dir,
            sample_limits=sample_limits,
            # The oracle's prompt names the layers it is really given
            prompt_layers="actual",
            # Activations read on Qwen3-8B with every LoRA disabled, as during training
            target_activations="base",
            judge_now=False,
        )
        del model, tokenizer  # frees Qwen3-8B for the judge
        gc.collect()
        torch.cuda.empty_cache()

    if args.phase in ("all", "judge"):
        aobench.use_local_judge(args.judge_model, load_in_4bit=args.judge_4bit)
        print(f"GPU memory used by the judge: {torch.cuda.memory_allocated() / 1e9:.0f} GB")
        aobench.judge_sweep(output_dir, starts=starts)

    results = aobench.collect_results(output_dir, starts=starts)
    results.to_csv(f"{output_dir}/results.csv", index=False)
    table = aobench.aggregate_scores(results)
    print("\nNormalized scores per window (aggregate = mean over the evals):")
    print(table.round(3).to_string())
    if trained_start in table.index:
        reference = table.loc[trained_start, "aggregate"]
        loss = (table[["aggregate"]] - reference).rename(columns={"aggregate": "change"})
        loss["relative"] = (loss["change"] / reference).map("{:+.0%}".format)
        print("\nChange relative to the trained window:")
        print(loss.round(3).to_string())
    if len(table):
        aobench.plot_sweep(results).savefig(f"{output_dir}/sweep.png", dpi=150)
        print(f"\nSaved {output_dir}/results.csv and {output_dir}/sweep.png")


if __name__ == "__main__":
    main()
