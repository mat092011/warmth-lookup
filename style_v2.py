"""Create independent blind ballots and score two completed ballots."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from dataset import save_json
from experiment import BASE, chat

QUESTIONS = [
    "I spilled rice on the floor. Where should I start?",
    "My first loaf came out flat. What could I try next time?",
    "I'm nervous about joining a book club. How do I introduce myself?",
    "I forgot a friend's birthday. What could I say?",
    "How can I keep two similar keys apart?",
    "My shirt has a loose button. How do I secure it?",
    "I have ten minutes to tidy my desk. What should I do first?",
    "I keep losing my shopping list. Any ideas?",
    "How do I fold a blanket so it fits on a narrow shelf?",
    "I promised to water a neighbor's plants. What should I ask them?",
    "I want to teach my grandmother a card game. Where should we begin?",
    "I'm shy about sharing my drawing. How can I ask for feedback?",
    "What can I bring to a picnic if I can't cook?",
    "How do I help a new roommate feel at home?",
    "My mug left a ring on the table. What should I try first?",
    "I have a spare notebook. How could I use it?",
    "How should I label boxes before a move?",
    "I mixed up two appointments. What should I tell the person waiting?",
    "What is a simple way to start learning to knit?",
    "I feel awkward asking a librarian for help. What could I say?",
]


def prepare(model, spec, out):
    turns = [chat(BASE, q) for q in QUESTIONS]
    answers = {mode: model.generate(turns, s, limit=96) for mode, s in (("neutral", None), ("caa", spec))}
    save_json(out / "raw.json", dict(questions=QUESTIONS, answers=answers, limit=96))
    rng = random.SystemRandom()
    keys = {}
    for judge in ("r1", "r2"):
        order = list(range(len(QUESTIONS)))
        rng.shuffle(order)
        keys[judge] = {}
        with (out / f"{judge}.csv").open("x", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["item", "question", "left", "right", "vote"])
            for num, i in enumerate(order, 1):
                left, right = rng.sample(["neutral", "caa"], 2)
                writer.writerow([num, QUESTIONS[i], answers[left][i], answers[right][i], ""])
                keys[judge][str(num)] = dict(question=i, left=left, right=right)
    save_json(out / "key.json", keys)


def score(folder, files):
    key = json.loads((folder / "key.json").read_text())
    votes = {}
    counts = {}
    for judge, path in zip(("r1", "r2"), files, strict=True):
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        with (folder / f"{judge}.csv").open(encoding="utf-8-sig", newline="") as f:
            original = {r["item"]: r for r in csv.DictReader(f)}
        if len(rows) != 20 or {r["item"] for r in rows} != set(original):
            raise ValueError("Incomplete ballot")
        votes[judge] = {}
        for r in rows:
            k = key[judge][r["item"]]
            if any(r[col] != original[r["item"]][col] for col in ("question", "left", "right")):
                raise ValueError("Ballot content changed")
            v = r["vote"].strip().lower()
            if v not in ("left", "right", "tie"):
                raise ValueError("Expected left, right or tie")
            votes[judge][k["question"]] = "tie" if v == "tie" else k[v]
        counts[judge] = dict(Counter(votes[judge].values()))
    agreement = sum(votes["r1"][q] == votes["r2"][q] for q in range(20))
    return dict(counts=counts, agreement=agreement/20, matched=agreement, questions=20)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("folder", "r1", "r2", "out"):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    save_json(args.out, score(args.folder, [args.r1, args.r2]))
