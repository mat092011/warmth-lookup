"""Recompute paired estimates and figures from a complete evaluation log."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import CONFIG as C, ROOT, save_json
from experiment import MODES

METRICS = [
    ("needed_lookup", "choice", True, "lookup"),
    ("redundant_lookup", "choice", False, "lookup"),
    ("choice_accuracy", "choice", None, "correct"),
    ("provided_accuracy", "provided", None, "correct"),
    ("choice_format_error", "choice", None, "malformed"),
    ("provided_format_error", "provided", None, "malformed"),
]


def paired(values, reference, strata):
    diff = np.asarray(values, dtype=float) - reference
    rng = np.random.default_rng(C["seed"])
    sums = np.zeros(C["bootstrap_draws"])
    for group in sorted(set(strata)):
        cell = diff[np.array(strata) == group]
        sums += rng.choice(cell, (len(sums), len(cell))).sum(axis=1)
    return dict(delta=float(diff.mean()), ci95=np.quantile(sums/len(diff), [.025, .975]).tolist())


def summarize(out):
    rows = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
    tasks = [t for t in json.loads((ROOT / "tasks.json").read_text()) if t["split"] == "evaluation"]
    index = {(r["mode"], r["phase"], r["id"]): r for r in rows}
    expected = {(m, p, t["id"]) for m in MODES for p in ("choice", "provided") for t in tasks}
    if len(rows) != len(expected) or set(index) != expected:
        raise ValueError("Incomplete or duplicate evaluation trials")
    metrics = {}
    for name, phase, need, field in METRICS:
        group = [t for t in tasks if need is None or need == t["necessary"]]
        strata = [f"{t['necessary']}-{t['gold']}" for t in group]
        values = {m: np.array([index[m, phase, t["id"]][field] for t in group], dtype=int) for m in MODES}
        metrics[name] = {m: dict(count=int(arr.sum()), n=len(arr), rate=float(arr.mean()),
                                 vs_neutral=paired(arr, values["neutral"], strata),
                                 vs_random=paired(arr, values["random"], strata)) for m, arr in values.items()}
    save_json(out / "metrics.json", metrics)
    for filename, keys, labels in [
        ("lookup", ["needed_lookup", "redundant_lookup"], ["Tag needed", "Override present"]),
        ("accuracy", ["choice_accuracy", "provided_accuracy"], ["Optional lookup", "Fact supplied"]),
    ]:
        fig, ax = plt.subplots(figsize=(7, 3.7), constrained_layout=True)
        x = np.arange(len(MODES))
        for i, (key, label) in enumerate(zip(keys, labels)):
            bars = ax.bar(x+(i-.5)*.38, [metrics[key][m]["rate"]*100 for m in MODES], .38,
                          label=label, color=["#326888", "#e2a457"][i])
            ax.bar_label(bars, fmt="%.0f", fontsize=8)
        ax.set(xticks=x, xticklabels=["Neutral", "Warm prompt", "CAA", "Random"],
               ylabel="Trials (%)", ylim=(0, 114), title=f"Held-out {filename} rates")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
        fig.savefig(out / f"{filename}.png", dpi=160)
        plt.close(fig)
    changes = [dict(id=t["id"], mode=m, neutral=index["neutral","choice",t["id"]]["first"],
                    reply=index[m,"choice",t["id"]]["first"])
               for t in tasks for m in MODES[1:]
               if index[m,"choice",t["id"]]["lookup"] != index["neutral","choice",t["id"]]["lookup"]]
    save_json(out / "lookup-changes.json", changes)


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
