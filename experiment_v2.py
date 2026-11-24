"""Run one stage of PLAN_V2 in a new output directory."""
import argparse
import hashlib
import json
import random
import platform
import subprocess
from pathlib import Path

import torch

from dataset import CONFIG, ROOT, save_json
from experiment import BASE, WARM, FINAL, CHOICE, MODES, Experiment, chat, parse

VECTORS = ROOT / "results/evaluation/vectors.pt"
TEMPLATES = ("prose", "table", "form")
COLORS = "red blue green yellow white black orange purple pink brown".split()
DEMO = [
    ("user", "Copy the command for the selected tag. If the tag is missing you can request it with LOOKUP.\ngray: ANSWER B\ngold: ANSWER A\nSelected tag: gold"),
    ("assistant", "ANSWER A"),
    ("user", "Copy the command for the selected tag. If the tag is missing you can request it with LOOKUP.\ngray: ANSWER B\ngold: ANSWER A\nSelected tag: hidden"),
    ("assistant", "LOOKUP"),
    ("user", "Copy the command for the selected tag.\ngray: ANSWER B\ngold: ANSWER A\nSelected tag: gray"),
    ("assistant", "ANSWER B"),
]


def digest(path):
    data = path.read_bytes()
    if path.suffix != ".pt":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def cards(split):
    rng = random.Random(f"{CONFIG['seed']}:command-card:{split}")
    n = 4 if split == "gate" else 20
    ids = rng.sample(range(2000 if split == "gate" else 4000,
                           3000 if split == "gate" else 5000), 3*n)
    out = []
    for template in TEMPLATES:
        golds = list("AB" * (n // 2))
        rng.shuffle(golds)
        for gold in golds:
            colors = rng.sample(COLORS, 2)
            bays = rng.sample(["A", "B"], 2)
            out.append(dict(id=ids.pop(), template=template, colors=colors, bays=bays, gold=gold))
    rng.shuffle(out)
    return out


def prompt(card, view, gate=False):
    c, d = card["colors"]
    a, b = card["bays"]
    gold = card["gold"]
    if view == "flipped":
        gold = "B" if gold == "A" else "A"
    tag = c if gold == a else d
    shown = "hidden" if view == "hidden" else tag
    body = {
        "prose": f'For tag "{c}", reply "ANSWER {a}". For tag "{d}", reply "ANSWER {b}".',
        "table": f"Tag | Command\n{c} | ANSWER {a}\n{d} | ANSWER {b}",
        "form": f"Tag: {c}\nCommand: ANSWER {a}\nTag: {d}\nCommand: ANSWER {b}",
    }[card["template"]]
    instruction = "Copy the command for the selected tag. If the tag is missing you can request it with LOOKUP. Output one command only."
    if gate:
        instruction = "Your first command must be LOOKUP. Output that command only." if view == "hidden" else "Copy the command for the selected tag. Output one command only."
    return f"Parcel {card['id']}\n{instruction}\n{body}\nSelected tag: {shown}", f"Selected tag: {tag}", gold


def trials(model, items, mode, view, spec, gate=False):
    info = [prompt(c, view, gate) for c in items]
    system = WARM if mode == "warm_prompt" else BASE
    turns = [[dict(role="system", content=system)] +
             [dict(role=r, content=s) for r, s in DEMO] +
             [dict(role="user", content=body)] for body, _, _ in info]
    first = model.generate(turns, spec)
    requested = [i for i, s in enumerate(first) if parse(s) == "LOOKUP"]
    for i in requested:
        turns[i].extend([dict(role="assistant", content=first[i]),
                         dict(role="user", content=prompt(items[i], "visible", gate=True)[0])])
    follow = dict(zip(requested, model.generate([turns[i] for i in requested], spec), strict=True))
    for i, (c, s) in enumerate(zip(items, first, strict=True)):
        last = parse(follow.get(i, s))
        yield dict(id=c["id"], template=c["template"], view=view, mode=mode, gold=info[i][2],
                   messages=turns[i], first=s, followup=follow.get(i), parsed_first=parse(s),
                   parsed_final=last, lookup=i in follow, issued_fact=info[i][1] if i in follow else None,
                   correct=last == "ANSWER " + info[i][2], malformed=last not in ("ANSWER A", "ANSWER B"))


def gate_result(rows):
    cells = {(r["id"], r["view"]):r for r in rows}
    ids = {r["id"] for r in rows}
    assert len(rows) == len(cells) == 36 and len(ids) == 12
    direct = sum(all(cells[i, v]["correct"] and not cells[i, v]["lookup"]
                     for v in ("visible", "flipped")) for i in ids)
    forced = sum(cells[i, "hidden"]["correct"] and cells[i, "hidden"]["lookup"] for i in ids)
    return dict(direct_pairs=direct, forced_lookup=forced, passed=direct >= 10 and forced >= 10)


@torch.inference_mode()
def run(stage, out):
    if stage == "main":
        lock = json.loads((ROOT / "v2/frozen.json").read_text())
        for name, sha in lock.items():
            if digest(ROOT / name) != sha:
                raise ValueError(f"Frozen input changed: {name}")
        if not json.loads((ROOT / "v2/gate21/summary.json").read_text())["passed"]:
            raise ValueError("Gate closed")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(CONFIG["threads"])
    torch.manual_seed(CONFIG["seed"])
    model = Experiment()
    vectors = torch.load(VECTORS, weights_only=True)
    assert torch.isclose(vectors["caa"].norm(), vectors["random"].norm())
    specs = dict(neutral=None, warm_prompt=None, caa=(18, vectors["caa"]), random=(18, vectors["random"]))
    if stage == "style":
        from style_v2 import prepare
        prepare(model, specs["caa"], out)
        return
    items = cards(stage)
    if stage == "main":
        if items != json.loads((ROOT / "v2/main-cards.json").read_text()):
            raise ValueError("Evaluation cards differ from frozen inputs")
        save_json(out / "manifest.json", dict(commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(), config=CONFIG, python=platform.python_version(),
            torch=torch.__version__, dtype="bfloat16", decode="greedy", frozen_sha256=lock))
    save_json(out / "cards.json", items)
    rows = []
    with (out / "responses.jsonl").open("x", encoding="utf-8") as f:
        for mode in (("neutral",) if stage == "gate" else MODES):
            for view in (("visible", "flipped", "hidden") if stage == "gate" else ("visible", "hidden")):
                block = list(trials(model, items, mode, view, specs[mode], stage == "gate"))
                rows.extend(block)
                for row in block:
                    f.write(json.dumps(row) + "\n")
                f.flush()
                print(mode, view, len(block), flush=True)
    if stage == "gate":
        summary = gate_result(rows)
    else:
        from analysis_v2 import summarize
        summary = summarize(rows, out)
    save_json(out / "summary.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("gate", "main", "style"))
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.stage, a.out)
