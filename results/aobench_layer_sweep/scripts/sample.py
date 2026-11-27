"""Prints a blind sample (no judge label) of one judged eval for blind re-grading.

usage: python sample.py EVAL N_PER_WINDOW [max_problem_chars]
Writes audit_<eval>_items.csv with the hidden labels, prints the items without them.
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
WINDOWS = [13, 21, 31]
ev, k = sys.argv[1], int(sys.argv[2])
maxc = int(sys.argv[3]) if len(sys.argv) > 3 else 700
j = pd.read_pickle(HERE / "judged.pkl")
d = j[(j["eval"] == ev) & (j["start"].isin(WINDOWS))]
parts = [d[d["start"] == s].sample(k, random_state=1000 + s) for s in WINDOWS]
s = pd.concat(parts).sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle windows
s["audit_id"] = [f"{ev[:2]}{i:02d}" for i in range(len(s))]
s.to_pickle(HERE / f"audit_{ev}_items.pkl")


def cut(x, n):
    x = str(x).replace("\n", " ")
    return x if len(x) <= n else "…" + x[-n:] if ev == "backtracking" else x[:n] + "…"


for _, r in s.iterrows():
    print(f"### {r['audit_id']}")
    if ev == "activation_sensitivity":
        print("PROBLEM (C version):", cut(r["problem"], maxc))
        print("MISSING IN C:", r["missing"])
        print("A:", r["response_a"])
        print("C:", r["response_c"])
    elif ev == "backtracking":
        print("GROUND TRUTH:", r["ground_truth"])
        print("AO:", r["response"])
    else:
        print("PROBLEM:", cut(r["problem"], maxc))
        if ev == "vagueness":
            print("QUESTION:", r["question"])
        print("AO:", r["response"])
    print()
