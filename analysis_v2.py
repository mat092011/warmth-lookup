"""Paired analysis for the locked 60-card evaluation."""
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import paired
from experiment import MODES, parse


def summarize(rows, folder):
    idx = {(r["mode"], r["view"], r["id"]): r for r in rows}
    ids = sorted({r["id"] for r in rows})
    if len(idx) != 480 or len(rows) != 480 or len(ids) != 60:
        raise ValueError("Incomplete or duplicate trials")
    strata = []
    for i in ids:
        meta = {(idx[m,v,i]["template"], idx[m,v,i]["gold"])
                for m in MODES for v in ("hidden", "visible")}
        if len(meta) != 1:
            raise ValueError("Inconsistent paired stimuli")
        strata.append("-".join(meta.pop()))
    if sorted(Counter(strata).values()) != [10]*6:
        raise ValueError("Unbalanced strata")
    for r in rows:
        first = parse(r["first"])
        requested = first == "LOOKUP"
        final = parse(r["followup"]) if requested else first
        if (requested != r["lookup"] or requested != (r["issued_fact"] is not None)
                or requested != (r["followup"] is not None)
                or r["correct"] != (final == "ANSWER " + r["gold"])
                or r["malformed"] != (final not in ("ANSWER A", "ANSWER B"))):
            raise ValueError("Inconsistent trial derived from raw reply")
    rates, select = {}, {}
    for mode in MODES:
        rates[mode] = {}
        look = {}
        for view in ("visible", "hidden"):
            group = [idx[mode,view,i] for i in ids]
            look[view] = np.array([int(r["lookup"]) for r in group])
            rates[mode][view] = dict(n=len(group), lookups=int(look[view].sum()),
                                     correct=sum(r["correct"] for r in group),
                                     malformed=sum(r["malformed"] for r in group),
                                     correct_after_lookup=sum(r["correct"] and r["lookup"] for r in group))
        select[mode] = look["hidden"] - look["visible"]
        rates[mode]["selectivity_pp"] = 100 * select[mode].mean()
    diff = paired(select["caa"], select["neutral"], strata)
    changes = [dict(id=i, view=v, neutral=idx["neutral",v,i]["first"], caa=idx["caa",v,i]["first"])
               for i in ids for v in ("visible", "hidden") if idx["neutral",v,i]["lookup"] != idx["caa",v,i]["lookup"]]
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    x = np.arange(4)
    for offset, view, color in ((-.2, "visible", "#6c939d"), (.2, "hidden", "#c6763e")):
        bars = ax.bar(x+offset, [100*rates[m][view]["lookups"]/60 for m in MODES], .4,
                      color=color, label=view)
        ax.bar_label(bars, fmt="%.1f")
    ax.set(xticks=x, xticklabels=["Neutral", "Warm prompt", "CAA", "Random"],
           ylabel="LOOKUP (%)", ylim=(0, 112))
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(.5, -.12))
    fig.savefig(folder / "lookup.png", dpi=160)
    plt.close(fig)
    return dict(n_cards=60, n_trials=480, rates=rates, primary_pp=100*diff["delta"],
                ci95_pp=[100*v for v in diff["ci95"]], changed_cards=len({r["id"] for r in changes}),
                changed_decisions=changes)
