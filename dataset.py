"""Build balanced, fictional routing records and independent style examples."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())


def make_tasks():
    rng = random.Random(CONFIG["seed"])
    tasks = []
    for split, count in [("calibration", 12), ("evaluation", 48)]:
        cells = [(need, gold) for need in (True, False) for gold in "AB"] * (count // 4)
        rng.shuffle(cells)
        for need, gold in cells:
            entity = rng.choice(["parcel", "crate", "folder", "sample", "ticket", "order"])
            label = f"{entity} {len(tasks) + 101}"
            flags = rng.sample(["jade", "amber", "coral", "indigo", "silver", "umber"], 2)
            flag = rng.choice(flags)
            opposite = "B" if gold == "A" else "A"
            first = gold if flag == flags[0] else opposite
            second = opposite if first == gold else gold
            rule = f"{flags[0]} uses bay {first}; {flags[1]} uses bay {second}."
            if not need:
                rule += f" An override for this {entity} assigns bay {gold} for every tag."
            tasks.append(dict(id=f"T{len(tasks)+1:02}", split=split, necessary=need,
                              prompt=f"Route {label}. Rule: {rule} Which bay should it use?",
                              fact=f"The tag on {label} is {flag}.", gold=gold))
    return tasks


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    save_json(ROOT / "tasks.json", make_tasks())
