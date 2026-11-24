import unittest
from collections import Counter

from experiment_v2 import cards, gate_result, prompt, trials
from test_experiment import ScriptedModel


class CardTests(unittest.TestCase):
    def test_switch(self):
        items = cards("gate")
        self.assertEqual(len({c["id"] for c in items}), 12)
        self.assertEqual(set(Counter((c["template"], c["gold"]) for c in items).values()), {2})
        for c in items:
            shown, fact, gold = prompt(c, "visible")
            hidden, same_fact, same_gold = prompt(c, "hidden")
            self.assertEqual((fact, gold), (same_fact, same_gold))
            self.assertNotEqual(shown, hidden)
            _, other_fact, other_gold = prompt(c, "flipped")
            self.assertNotEqual(gold, other_gold)
            self.assertNotEqual(fact, other_fact)

    def test_count_request_even_when_final_is_wrong(self):
        c = cards("gate")[0]
        row, = trials(ScriptedModel("LOOKUP", "LOOKUP"), [c], "neutral", "hidden", None)
        self.assertTrue(row["lookup"])
        self.assertTrue(row["malformed"])
        self.assertFalse(row["correct"])
        self.assertIn(row["issued_fact"], row["messages"][-1]["content"])

    def test_gate_uses_both_correct_in_same_pair(self):
        rows = [dict(id=i, view=v, lookup=v=="hidden",
                     correct=(i>=2 if v=="visible" else i<10 if v=="flipped" else True))
                for i in range(12) for v in ("visible", "flipped", "hidden")]
        result = gate_result(rows)
        self.assertEqual(result["direct_pairs"], 8)
        self.assertFalse(result["passed"])
