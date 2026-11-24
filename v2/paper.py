"""Build the final LaTeX article at the historical output path."""
import subprocess
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    (root / "output/pdf").mkdir(exist_ok=True, parents=True)
    subprocess.run(["tectonic", "--outdir", "output/pdf", "paper/warmth-lookup.tex"],
                   cwd=root, check=True)
