"""Verify gate records or compare two full runs, including their published metrics."""
import argparse
import json
import tempfile
from pathlib import Path

from analysis_v2 import summarize
from experiment import parse
from experiment_v2 import gate_result, prompt


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(out):
    rows = [json.loads(line) for line in (out/"responses.jsonl").read_text().splitlines()]
    items = {c["id"]: c for c in read(out/"cards.json")}
    for r in rows:
        c = items[r["id"]]
        tag = c["colors"][c["bays"].index(r["gold"])]
        gold = c["gold"] if r["view"] != "flipped" else ("B" if c["gold"] == "A" else "A")
        assert r["gold"] == gold and r["template"] == c["template"]
        action = parse(r["first"])
        lookup = action == "LOOKUP"
        assert r["lookup"] == lookup == (r["followup"] is not None)
        if lookup:
            assert tag in r["issued_fact"] and r["issued_fact"] in r["messages"][-1]["content"]
        else:
            assert r["issued_fact"] is None
        last = parse(r["followup"]) if lookup else action
        assert (action, last) == (r["parsed_first"], r["parsed_final"])
        assert r["correct"] == (last == "ANSWER " + gold)
        assert r["malformed"] == (last not in ("ANSWER A", "ANSWER B"))
    return rows


def check(out, replica=None):
    rows = records(out)
    if replica is None:
        stats = gate_result(rows)
        assert stats == read(out/"summary.json")
        return stats
    def keyed(data):
        return {(r["mode"],r["view"],r["id"]):r for r in data}
    assert keyed(rows) == keyed(records(replica))
    assert read(out/"cards.json") == read(replica/"cards.json")
    items = {c["id"]: c for c in read(out/"cards.json")}
    for r in rows:
        body, fact, _ = prompt(items[r["id"]], r["view"])
        assert r["messages"][7]["content"] == body
        if r["lookup"]:
            assert r["issued_fact"] == fact
    with tempfile.TemporaryDirectory() as tmp:
        metrics = summarize(rows, Path(tmp))
    assert metrics == read(out/"summary.json") == read(replica/"summary.json")
    return dict(matched_trials=len(rows), matched_metrics=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("replica", nargs="?", type=Path)
    args = p.parse_args()
    print(json.dumps(check(args.out, args.replica)))
