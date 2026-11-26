import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class HealthTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "run_health.py"), *args],
                              capture_output=True, text=True)

    def test_published_archive(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["trials"], 480)

    def test_duplicate_with_unchanged_total(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            for name in ("manifest.json", "cards.json", "summary.json", "responses.jsonl"):
                shutil.copyfile(ROOT / "v2/full" / name, run / name)
            log = run / "responses.jsonl"
            lines = log.read_text(encoding="utf-8").splitlines()
            lines[-1] = lines[0]
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = self.cli(run)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate", json.loads(result.stderr)["reason"])
