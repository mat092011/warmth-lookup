# Does warmth steering change when a model asks for evidence?

Qwen2.5-1.5B-Instruct solved all 60 routing cards when the required tag was visible. When the tag was hidden, the neutral model executed 40 lookups; adding a warmth-derived activation vector produced 38. On both changed cards, the steered response was `LOOKUP hidden`, which the tool rejected. This is evidence about **successful tool calls**, not an independent measure of willingness to seek information.

[Read the short article](docs/article.md) · [Full paper (PDF)](output/pdf/warmth-lookup.pdf) · [LaTeX source](paper/warmth-lookup.tex) · [Trial transcripts](v2/full/responses.jsonl) · [Paired estimates](v2/full/summary.json)

## Inspect the result in seconds

From the repository root, with Python 3.12, no model download, API key, GPU or third-party packages:

```sh
python run_health.py
python -m unittest -v test_run_health
```

The command emits JSON: model revision, original run commit, seed, software versions, checked frozen inputs, cell-level lookups, errors, accuracy and the primary contrast. It exits nonzero on missing trials, duplicates or inconsistent outcomes. It recounts a finished run; it does not generate model responses or independently recompute the bootstrap interval. Pass another finished run directory as its positional argument for the same checks.

## Reproduce the model experiment

Requires Python 3.12, uv, Git and sufficient disk/RAM for the pinned model and CPU Torch. Initial dependency and model downloads need network access unless already cached. No paid endpoint. Run from the repository root; output directories must not exist yet.

```sh
uv sync --locked --python 3.12
uv run --no-sync python -X utf8 -m unittest -v
uv run --no-sync python -X utf8 verify_v2.py v2/gate21
uv run --no-sync python -X utf8 experiment_v2.py main runs/main-replica
uv run --no-sync python -X utf8 verify_v2.py v2/full runs/main-replica
python run_health.py runs/main-replica
```

The last verifier compares exact records/cards and recomputes metrics, including the paired interval. Cross-platform or different hardware results need not be bit-identical. Do not repurpose the committed evidence directories for new runs.

## Repository guide

| Question | Where to look |
|---|---|
| What changed after the pilot, and what was frozen? | [Original plan](PLAN_V2.md), [interface amendment](v2/AMENDMENT.md), [frozen hashes](v2/frozen.json), [human-rating closeout](v2/CLOSEOUT.md) |
| What exactly did the model see? | [Card generator and loop](experiment_v2.py), [frozen cards](v2/main-cards.json), [all main observations](v2/full/responses.jsonl) |
| Which estimate was primary? | [Analysis](analysis_v2.py), [main summary](v2/full/summary.json) and [run manifest](v2/full/manifest.json) |
| Which outputs are preliminary? | [First pilot](results/evaluation/), [failed gate](v2/gate/), [revised gate](v2/gate21/). Neither gate is part of the main 60-card estimate. |
| Where is the full article and history? | [Paper](paper/warmth-lookup.tex), [compiled PDF](output/pdf/warmth-lookup.pdf), [changelog](CHANGELOG.md) |

**Provenance:** routing records use fictional labels from seeded generators. Transcripts are outputs of the named model on these tasks. The repository contains derived steering vectors, not the model weights; check the upstream model's terms for reuse. The original protocol and amendment are preserved in their original language because the freeze records their exact contents.
