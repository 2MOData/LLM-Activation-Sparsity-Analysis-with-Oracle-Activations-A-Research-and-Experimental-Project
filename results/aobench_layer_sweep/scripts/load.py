"""Loads every per-example result of the AObench window sweep into tidy pandas tables."""

import json
import sys
from pathlib import Path

import pandas as pd

# Folder holding the window_AA-BB/ folders of the raw outputs (see ../README.md)
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "raw"
V = "qwen3-8b-ao-v3-best"
TRAINED = 21


def windows():
    out = []
    for d in sorted(ROOT.glob("window_*")):
        start = int(d.name.split("_")[1].split("-")[0])
        out.append((start, d))
    return sorted(out)


def load_judged():
    """One row per judged item: window, eval, key, cluster, category/grades, texts."""
    rows = []
    for start, d in windows():
        # hallucination
        h = json.loads((d / "hallucination" / f"hallucination_{V}.json").read_text())
        for r in h["scored_results"]:
            rows.append(
                dict(
                    start=start,
                    eval="hallucination",
                    key=f"ri{r['result_index']}",
                    cluster=r["problem_id"],
                    category=r.get("category"),
                    reasoning=r.get("reasoning"),
                    response=r["ao_response"],
                    problem=r["problem"],
                    question=r["prompt_name"],
                    domain=r.get("domain"),
                )
            )
        dc = json.loads((d / "domain_confusion" / f"domain_confusion_{V}.json").read_text())
        for r in dc["scored_results"]:
            rows.append(
                dict(
                    start=start,
                    eval="domain_confusion",
                    key=f"ri{r['result_index']}",
                    cluster=r["problem_id"],
                    category=r.get("category"),
                    reasoning=r.get("reasoning"),
                    response=r["ao_response"],
                    problem=r["problem"],
                    question=r["prompt_name"],
                    domain=r.get("domain"),
                )
            )
        vg = json.loads((d / "vagueness" / f"vagueness_{V}.json").read_text())
        for r in vg["scored_results"]:
            rows.append(
                dict(
                    start=start,
                    eval="vagueness",
                    key=f"ri{r['result_index']}",
                    cluster=r["problem_id"],
                    category=r.get("category"),
                    reasoning=r.get("reasoning"),
                    response=r["ao_response"],
                    problem=r["problem"],
                    question=r["prompt"],
                    domain=r.get("domain"),
                )
            )
        bt = json.loads((d / "backtracking" / f"backtracking_{V}.json").read_text())
        # backtracking scored results carry no problem_id: recover from verbalizer order
        for r in bt["scored_results"]:
            rows.append(
                dict(
                    start=start,
                    eval="backtracking",
                    key=f"bt|{r['result_index']}",
                    cluster=f"bt{r['result_index']}",
                    specificity=r.get("specificity"),
                    correctness=r.get("correctness"),
                    reasoning=r.get("reasoning"),
                    response=r["ao_response"],
                    ground_truth=r["ground_truth"],
                    backtrack_rate=r.get("backtrack_rate"),
                )
            )
        a = json.loads(
            (d / "activation_sensitivity" / f"activation_sensitivity_{V}.json").read_text()
        )
        for i, r in enumerate(a["scored_results"]):
            rows.append(
                dict(
                    start=start,
                    eval="activation_sensitivity",
                    key=f"pair{i}",
                    cluster=r["problem_id"],
                    pair_index=i,
                    category=r.get("category"),
                    reasoning=r.get("reasoning"),
                    response_a=r["response_a"],
                    response_c=r["response_c"],
                    problem=r["problem_text"],
                    missing=r["missing_info_description"],
                    question=r["prompt_name"],
                )
            )
    return pd.DataFrame(rows)


def load_binary():
    """One row per yes/no item of the ROC AUC evals (mode = sub-eval averaged by AObench)."""
    rows = []
    specs = {
        "mmlu_prediction": ["mmlu_prediction/pre_answer", "mmlu_prediction/post_answer"],
        "missing_info": ["missing_info"],
        "sycophancy": ["sycophancy/cot", "sycophancy/no_cot"],
    }
    for start, d in windows():
        for name, subdirs in specs.items():
            for sub in subdirs:
                files = list((d / sub).glob(f"*_binary_{V}.json"))
                assert len(files) == 1, (d, sub, files)
                data = json.loads(files[0].read_text())
                mode = sub.split("/")[-1] if "/" in sub else "main"
                for i, r in enumerate(data["binary_scored_results"]):
                    cluster = r.get("problem_id") or r["id"]
                    if name == "sycophancy":
                        cluster = "_".join(r["id"].split("_")[:2])  # scruples_154
                    rows.append(
                        dict(
                            start=start,
                            eval=name,
                            mode=mode,
                            key=f"{mode}|{i}|{r['id']}|{r.get('prompt_name')}",
                            cluster=cluster,
                            label=r["binary_label"],
                            margin=r["margin_yes_minus_no"],
                            condition=r.get("condition"),
                        )
                    )
    return pd.DataFrame(rows)


def load_number():
    rows = []
    for start, d in windows():
        data = json.loads((d / "number_prediction" / f"number_prediction_{V}.json").read_text())
        for r in data["scored_results"]:
            rows.append(
                dict(
                    start=start,
                    eval="number_prediction",
                    key=f"{r['id']}|{r['prompt_name']}",
                    cluster=r["id"],
                    match=int(bool(r["matches_model_answer"])),
                    match_true=int(bool(r["matches_true_answer"])),
                    category=r["category"],
                    prompt=r["prompt_name"],
                    ao_response=r["ao_response"],
                    model_answer=r["model_answer"],
                )
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    j = load_judged()
    b = load_binary()
    n = load_number()
    print(j.groupby(["eval"]).size() / 19)
    print(b.groupby(["eval", "mode"]).size() / 19)
    print(len(n) / 19)
    # key uniqueness per window
    for name, df in (("judged", j), ("binary", b), ("number", n)):
        dup = df.groupby(["start", "eval", "key"]).size()
        print(name, "duplicate keys:", int((dup > 1).sum()))
    out = Path(__file__).parent
    j.to_pickle(out / "judged.pkl")
    b.to_pickle(out / "binary.pkl")
    n.to_pickle(out / "number.pkl")
