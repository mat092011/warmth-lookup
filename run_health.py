"""Audit a finished v2 run offline, using only Python's standard library."""
import argparse
import hashlib
import json
import sys
from itertools import product
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = ("neutral", "warm_prompt", "caa", "random")
VIEWS = ("visible", "hidden")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def audit(run):
    manifest, cards, summary = [read(run / name) for name in
                                ("manifest.json", "cards.json", "summary.json")]
    lock = read(ROOT / "v2/frozen.json")
    require(manifest["frozen_sha256"] == lock, "Manifest disagrees with frozen inputs")
    for name, expected in lock.items():
        data = (ROOT / name).read_bytes()
        if not name.endswith(".pt"):
            data = data.replace(b"\r\n", b"\n")
        require(hashlib.sha256(data).hexdigest() == expected, f"Frozen input changed: {name}")
    cfg = read(ROOT / "config.json")
    require(manifest["config"] == cfg, "Model configuration mismatch")
    require(cards == read(ROOT / "v2/main-cards.json"), "Cards differ from frozen inputs")
    gold = {card["id"]: card["gold"] for card in cards}
    expected = set(product(gold, MODES, VIEWS))
    seen = set()
    cells = {k: dict(n=0, lookups=0, correct=0, malformed=0, correct_after_lookup=0)
             for k in product(MODES, VIEWS)}
    log = run / "responses.jsonl"
    for line in log.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        i, mode, view = key = row["id"], row["mode"], row["view"]
        require(key in expected and key not in seen, f"Unexpected or duplicate trial: {key}")
        seen.add(key)
        lookup = row["first"].strip() == "LOOKUP"
        require(lookup == (row["followup"] is not None) == row["lookup"], f"Tool mismatch: {key}")
        response = (row["followup"] if lookup else row["first"]).strip()
        correct, malformed = response == "ANSWER " + gold[i], response not in ("ANSWER A", "ANSWER B")
        require((gold[i], correct, malformed) == (row["gold"], row["correct"], row["malformed"]),
                f"Outcome mismatch: {key}")
        cell = cells[mode, view]
        for name, value in zip(cell, (1, lookup, correct, malformed, lookup and correct)):
            cell[name] += value
    require(seen == expected, f"Incomplete run: {len(seen)} of {len(expected)} trials")
    require((len(gold), len(seen)) == (summary["n_cards"], summary["n_trials"]), "Summary size mismatch")
    selectivity = {}
    for mode in MODES:
        for view in VIEWS:
            require(cells[mode, view] == summary["rates"][mode][view], f"Summary mismatch: {mode}/{view}")
        selectivity[mode] = 100 * (cells[mode, "hidden"]["lookups"] - cells[mode, "visible"]["lookups"]) / len(gold)
        require(isclose(selectivity[mode], summary["rates"][mode]["selectivity_pp"], abs_tol=1e-9),
                f"Selectivity mismatch: {mode}")
    require(isclose(selectivity["caa"] - selectivity["neutral"], summary["primary_pp"], abs_tol=1e-9),
            "Primary comparison mismatch")
    return dict(status="ok", model=cfg["model"], revision=cfg["revision"], run_commit=manifest["commit"],
                python=manifest["python"], torch=manifest["torch"], seed=cfg["seed"], decode=manifest["decode"],
                frozen_files=len(lock), cards=len(gold), trials=len(seen), primary_pp=summary["primary_pp"],
                cells={f"{mode}/{view}": counts for (mode, view), counts in cells.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=ROOT / "v2/full")
    args = parser.parse_args()
    try:
        report = audit(args.run)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        print(json.dumps(dict(status="error", reason=str(error))), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
