"""Freeze the prospective run once the independent gate has passed."""
from collections import Counter

from dataset import ROOT, save_json
from experiment_v2 import VECTORS, cards, digest
from verify_v2 import check, read


def freeze():
    target = ROOT / "v2/frozen.json"
    if target.exists():
        raise FileExistsError(target)
    if not check(ROOT / "v2/gate21")["passed"]:
        raise ValueError("Both gates of version 2.1 must pass")
    held = cards("main")
    assert len(held) == len({c["id"] for c in held}) == 60
    assert sorted(Counter((c["template"], c["gold"]) for c in held).values()) == [10]*6
    old_ids = {c["id"] for path in ("v2/gate/cards.json", "v2/gate21/cards.json") for c in read(ROOT/path)}
    assert not old_ids.intersection(c["id"] for c in held)
    save_json(ROOT / "v2/main-cards.json", held)
    names = ["experiment_v2.py", "analysis_v2.py", "experiment.py", "analysis.py", "dataset.py",
             "config.json", "uv.lock", "PLAN_V2.md", "v2/AMENDMENT.md", "v2/main-cards.json",
             "v2/gate21/summary.json", "v2/gate21/responses.jsonl", "test_v2.py",
             VECTORS.relative_to(ROOT).as_posix()]
    save_json(target, {n:digest(ROOT/n) for n in names})


if __name__ == "__main__":
    freeze()
