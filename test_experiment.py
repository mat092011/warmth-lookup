import json
import re
import unittest
from collections import Counter

import numpy as np

from analysis import paired
from dataset import ROOT, make_tasks
from experiment import parse, trials


class DatasetTest(unittest.TestCase):
    def test_tasks_and_gold(self):
        tasks = json.loads((ROOT / "tasks.json").read_text())
        self.assertEqual(make_tasks(), tasks)
        self.assertEqual(len({t["id"] for t in tasks}), 60)
        cells = Counter((t["split"], t["necessary"], t["gold"]) for t in tasks)
        for split, n in (("calibration", 3), ("evaluation", 12)):
            for need in (False, True):
                for gold in "AB":
                    self.assertEqual(cells[split, need, gold], n)
        for t in tasks:
            rules = dict(re.findall(r"(\w+) uses bay ([AB])", t["prompt"]))
            tag = t["fact"].removesuffix(".").split()[-1]
            self.assertEqual(rules[tag], t["gold"])
            self.assertEqual(len(set(rules.values())), 2)
            self.assertEqual("override" in t["prompt"], not t["necessary"])
            if not t["necessary"]:
                self.assertIn("assigns bay " + t["gold"], t["prompt"])


class AnalysisTest(unittest.TestCase):
    def test_stratified_paired_interval(self):
        # Within each stratum all paired differences agree, so every
        # stratified resample retains the observed weighted mean.
        result = paired([1, 1, 1, 0], np.array([0, 0, 0, 1]), ["x", "x", "x", "y"])
        self.assertEqual(result, {"delta": 0.5, "ci95": [0.5, 0.5]})


class ScriptedModel:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.calls = []

    def generate(self, turns, spec):
        self.calls.append([tuple(m["content"] for m in turn) for turn in turns])
        return [next(self.replies) for _ in turns]


class ProtocolTest(unittest.TestCase):
    def test_only_one_lookup_and_exact_commands(self):
        tasks = make_tasks()[:3]
        model = ScriptedModel("LOOKUP", "LOOKUP please", "ANSWER " + tasks[2]["gold"], "LOOKUP")
        rows = list(trials(model, tasks, "neutral", "choice"))
        self.assertEqual([r["lookup"] for r in rows], [True, False, False])
        self.assertEqual([r["malformed"] for r in rows], [True, True, False])
        self.assertEqual([r["correct"] for r in rows], [False, False, True])
        self.assertNotIn(tasks[0]["fact"], model.calls[0][0][-1])
        self.assertIn(tasks[0]["fact"], model.calls[1][0][-1])
        self.assertIsNone(parse("ANSWER A\nANSWER B"))

    def test_provided_fact_and_illegal_lookup(self):
        task = make_tasks()[0]
        model = ScriptedModel("LOOKUP")
        row, = trials(model, [task], "neutral", "provided")
        self.assertIn(task["fact"], model.calls[0][0][-1])
        self.assertEqual(model.calls[1], [])
        self.assertTrue(row["malformed"])
        self.assertFalse(row["lookup"])


if __name__ == "__main__":
    unittest.main()
