"""Point estimates, cluster-bootstrap CIs and paired differences vs the trained window."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
B = 2000
TRAINED = 21
rng = np.random.default_rng(0)

j = pd.read_pickle(HERE / "judged.pkl")
b = pd.read_pickle(HERE / "binary.pkl")
n = pd.read_pickle(HERE / "number.pkl")
STARTS = sorted(j["start"].unique())

# Backtracking rows have no problem_id: same order as the vagueness entries (one per dataset entry)
vag = j[(j["eval"] == "vagueness") & (j["start"] == TRAINED)].copy()
vag["idx"] = vag["key"].str[2:].astype(int)
bt_cluster = dict(zip(vag["idx"], vag["cluster"], strict=False))
mask = j["eval"] == "backtracking"
j.loc[mask, "cluster"] = j.loc[mask, "key"].str.split("|").str[1].astype(int).map(bt_cluster)
assert j["cluster"].notna().all()

# Per-item value for the rate-type evals
VALUE = {
    "hallucination": lambda d: (d["category"] != "obviously_wrong").astype(float),
    "domain_confusion": lambda d: (d["category"] == "domain_correct_specific").astype(float),
    "vagueness": lambda d: (d["category"] == "specific_correct").astype(float),
    "activation_sensitivity": lambda d: (d["category"] == "divergent_meaningful").astype(float),
    "backtracking": lambda d: (d["correctness"].astype(float) - 1) / 4,
}
GROUP = {  # evals that share problems are resampled together
    "hallucination": "bt",
    "domain_confusion": "bt",
    "vagueness": "bt",
    "backtracking": "bt",
    "missing_info": "mi",
    "activation_sensitivity": "mi",
    "mmlu_prediction": "mmlu",
    "sycophancy": "syc",
    "number_prediction": "num",
}


def cluster_weights(clusters_by_group):
    """Multinomial bootstrap weights per cluster, shared by all evals of a group."""
    out = {}
    for g, clusters in clusters_by_group.items():
        k = len(clusters)
        w = rng.multinomial(k, np.full(k, 1 / k), size=B).astype(float)
        out[g] = (dict(zip(clusters, range(k), strict=False)), w)
    return out


clusters = {
    "bt": sorted(
        j[j["eval"].isin(["hallucination", "domain_confusion", "vagueness", "backtracking"])][
            "cluster"
        ].unique()
    ),
    "mi": sorted(
        set(j[j["eval"] == "activation_sensitivity"]["cluster"])
        | set(b[b["eval"] == "missing_info"]["cluster"])
    ),
    "mmlu": sorted(b[b["eval"] == "mmlu_prediction"]["cluster"].unique()),
    "syc": sorted(b[b["eval"] == "sycophancy"]["cluster"].unique()),
    "num": sorted(n["cluster"].unique()),
}
W = cluster_weights(clusters)


def item_weights(group, item_clusters):
    index, w = W[group]
    cols = np.array([index[c] for c in item_clusters])
    return w[:, cols]  # [B, n_items]


def auc_point(labels, scores):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    cmp = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return cmp.mean()


def auc_boot(labels, scores, wts):
    pos = labels == 1
    cmp = (scores[pos][:, None] > scores[~pos][None, :]) + 0.5 * (
        scores[pos][:, None] == scores[~pos][None, :]
    )
    wp, wn = wts[:, pos], wts[:, ~pos]
    num = np.einsum("bp,pn,bn->b", wp, cmp, wn)
    return num / (wp.sum(1) * wn.sum(1))


point = {}  # (eval, start) -> normalized score
boot = {}  # (eval, start) -> [B] normalized
for ev, f in VALUE.items():
    d = j[j["eval"] == ev]
    keys = sorted(d["key"].unique())
    for s in STARTS:
        ds = d[d["start"] == s].set_index("key").loc[keys]
        x = f(ds).to_numpy()
        w = item_weights(GROUP[ev], ds["cluster"].to_numpy())
        point[ev, s] = x.mean()
        boot[ev, s] = (w * x).sum(1) / w.sum(1)

for ev in ["mmlu_prediction", "missing_info", "sycophancy"]:
    d = b[b["eval"] == ev]
    for s in STARTS:
        ps, bs = [], []
        for _mode, dm in d[d["start"] == s].groupby("mode"):
            dm = dm.sort_values("key")
            lab, sc = dm["label"].to_numpy(), dm["margin"].to_numpy()
            ps.append(auc_point(lab, sc))
            bs.append(auc_boot(lab, sc, item_weights(GROUP[ev], dm["cluster"].to_numpy())))
        point[ev, s] = (np.mean(ps) - 0.5) / 0.5
        boot[ev, s] = (np.mean(bs, axis=0) - 0.5) / 0.5

for s in STARTS:
    d = n[n["start"] == s].sort_values("key")
    x = d["match"].to_numpy(float)
    w = item_weights("num", d["cluster"].to_numpy())
    point["number_prediction", s] = x.mean()
    boot["number_prediction", s] = (w * x).sum(1) / w.sum(1)

EVALS = sorted({ev for ev, _ in point})
OBJECTIVE = ["mmlu_prediction", "missing_info", "number_prediction", "sycophancy"]
JUDGED = [
    "activation_sensitivity",
    "backtracking",
    "domain_confusion",
    "hallucination",
    "vagueness",
]
SUBSETS = {
    "aggregate": EVALS,
    "objective_only": OBJECTIVE,
    "judged_only": JUDGED,
    "aggregate_no_actsens_no_vague": [
        e for e in EVALS if e not in ("activation_sensitivity", "vagueness")
    ],
}
for name, evs in SUBSETS.items():
    for s in STARTS:
        point[name, s] = np.mean([point[e, s] for e in evs])
        boot[name, s] = np.mean([boot[e, s] for e in evs], axis=0)

# Check against results.csv
ref = pd.read_csv(HERE.parent / "results.csv")
maxdiff = max(abs(point[r["eval"], r["start"]] - r["normalized"]) for _, r in ref.iterrows())
print("max |recomputed - results.csv| =", maxdiff)

rows = []
for key in EVALS + list(SUBSETS):
    for s in STARTS:
        diff = boot[key, s] - boot[key, TRAINED]
        rows.append(
            dict(
                metric=key,
                start=s,
                layers=f"{s}-{s + 4}",
                score=point[key, s],
                lo=np.percentile(boot[key, s], 2.5),
                hi=np.percentile(boot[key, s], 97.5),
                diff=point[key, s] - point[key, TRAINED],
                diff_lo=np.percentile(diff, 2.5),
                diff_hi=np.percentile(diff, 97.5),
                p_two_sided=min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
                if s != TRAINED
                else np.nan,
            )
        )
out = pd.DataFrame(rows)
out.to_csv(HERE.parent / "stats_per_window.csv", index=False)
pd.set_option("display.width", 250)
for key in list(SUBSETS) + EVALS:
    t = out[out["metric"] == key].set_index("layers")
    t["sig"] = np.where((t["diff_lo"] > 0) | (t["diff_hi"] < 0), "*", "")
    print(f"\n=== {key}")
    print(
        t[["score", "lo", "hi", "diff", "diff_lo", "diff_hi", "p_two_sided", "sig"]]
        .round(3)
        .to_string()
    )

np.save(HERE / "boot.npy", {f"{k[0]}@{k[1]}": v for k, v in boot.items()}, allow_pickle=True)
