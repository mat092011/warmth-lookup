"""Exercise the blind scoring code on artificial ballots in a temporary folder."""
import csv
import tempfile
import unittest
from pathlib import Path

from dataset import ROOT
from style_v2 import score
from verify_v2 import read


class BlindScoringTest(unittest.TestCase):
    def test_question_alignment_after_side_and_row_randomization(self):
        folder = ROOT / "v2/style"
        keys = read(folder/"key.json")
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for judge in ("r1", "r2"):
                with (folder/f"{judge}.csv").open(encoding="utf-8-sig", newline="") as f:
                    ballots = list(csv.DictReader(f))
                for row in ballots:
                    key = keys[judge][row["item"]]
                    row["vote"] = ("tie" if judge == "r2" and key["question"] < 5
                                   else next(side for side in ("left", "right") if key[side] == "caa"))
                path = Path(tmp)/f"{judge}.csv"
                with path.open("w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(ballots[0]))
                    w.writeheader()
                    w.writerows(reversed(ballots))
                files.append(path)
            result = score(folder, files)
        self.assertEqual(result["counts"]["r1"], dict(caa=20, neutral=0, tie=0))
        self.assertEqual(result["counts"]["r2"], dict(caa=15, neutral=0, tie=5))
        self.assertEqual(result["agreement"], .75)
