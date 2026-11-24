"""Compare two independent full runs on the same inputs and check their raw trials."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from analysis import METRICS, paired
from dataset import ROOT, save_json
from experiment import BASE, CHOICE, FINAL, MODES, WARM, chat, parse


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(reference, repeat, report=None):
    tasks = {t["id"]: t for t in read(ROOT / "tasks.json") if t["split"] == "evaluation"}
    expected = {(mode, phase, key) for mode in MODES for phase in ("choice", "provided")
                for key in tasks}
    for folder in (reference, repeat):
        manifest = read(folder / "manifest.json")
        require(manifest.get("completed") and manifest["smoke"] is False, f"Incomplete run: {folder}")
        require(manifest["config"] == read(ROOT / "config.json"), "Config mismatch")
        for name, digest in manifest["inputs_sha256"].items():
            require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name)
    logs = [[json.loads(line) for line in (p / "responses.jsonl").read_text().splitlines()]
            for p in (reference, repeat)]
    require(logs[0] == logs[1], "Raw responses differ")
    rows = logs[0]
    keys = {(r["mode"], r["phase"], r["id"]) for r in rows}
    require(keys == expected and len(rows) == len(expected), "Missing or duplicate trials")
    for row in rows:
        task = tasks[row["id"]]
        first = parse(row["first"])
        lookup = first == "LOOKUP" and row["phase"] == "choice"
        instruction = CHOICE if row["phase"] == "choice" else task["fact"] + "\n" + FINAL
        system = WARM if row["mode"] == "warm_prompt" else BASE
        turns = chat(system, task["prompt"] + "\n" + instruction)
        if lookup:
            turns.extend([dict(role="assistant", content=row["first"]),
                          dict(role="user", content=task["fact"] + "\n" + FINAL)])
        require(row["messages"] == turns, f"Unexpected transcript for {row['id']}")
        require(row["lookup"] == lookup == (row["followup"] is not None), row["id"])
        final = parse(row["followup"]) if lookup else first
        require(row["correct"] == (final == "ANSWER " + task["gold"]), row["id"])
        require(row["malformed"] == (final not in ("ANSWER A", "ANSWER B")), row["id"])
        require(row["necessary"] == task["necessary"] and row["gold"] == task["gold"], row["id"])
    for name in ("calibration.json", "style-check.json", "metrics.json", "lookup-changes.json"):
        require(read(reference / name) == read(repeat / name), f"Runs differ: {name}")
    a, b = (torch.load(folder / "vectors.pt", weights_only=True) for folder in (reference, repeat))
    for name in ("caa", "random"):
        require(torch.equal(a[name], b[name]), f"Runs differ: {name} vector")
    require(torch.isclose(a["caa"].norm(), a["random"].norm()).item(), "Control norm mismatch")
    # Recount the published rates and compare all six pairs of conditions.
    index = {(r["mode"], r["phase"], r["id"]): r for r in rows}
    metrics = read(reference / "metrics.json")
    contrasts = {}
    for metric, phase, need, field in METRICS:
        group = [t for t in tasks.values() if need is None or t["necessary"] == need]
        strata = [f"{t['necessary']}-{t['gold']}" for t in group]
        values = {}
        for mode in MODES:
            arr = np.array([index[mode, phase, t["id"]][field] for t in group], dtype=int)
            published = metrics[metric][mode]
            require((int(arr.sum()), len(arr), arr.mean()) ==
                    (published["count"], published["n"], published["rate"]), metric)
            values[mode] = arr
        contrasts[metric] = {f"{other}_minus_{base}": paired(values[other], values[base], strata)
                             for base, other in combinations(MODES, 2)}
    if report:
        save_json(report, dict(matched_trials=len(rows), repeat_manifest=read(repeat / "manifest.json"),
                               pairwise=contrasts))
    print(f"Matched {len(rows)} paired trial records across runs, calibration, style checks, "
          "metrics, lookup changes and vectors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    verify(args.reference, args.repeat, args.report)
