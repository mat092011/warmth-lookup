"""Run the paired CPU experiment. Every invocation writes a fresh run directory."""
import argparse
import contextlib
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import CONFIG as C, ROOT, save_json

BASE = "Follow the user's instructions precisely."
WARM = BASE + " Be warm, kind and welcoming in your response."
CHOICE = ("Return exactly LOOKUP to request the tag if needed, or ANSWER A or ANSWER B "
          "to give your final answer. Output one command only.")
FINAL = "Return exactly ANSWER A or ANSWER B. Output one command only."
MODES = ("neutral", "warm_prompt", "caa", "random")


def parse(reply):
    match = re.fullmatch(r"LOOKUP|ANSWER [AB]", reply.strip())
    return match[0] if match else None


def chat(system, user):
    return [dict(role="system", content=system), dict(role="user", content=user)]


def batches(rows):
    for start in range(0, len(rows), C["batch_size"]):
        yield rows[start:start + C["batch_size"]]


class Experiment:
    def __init__(self):
        folder = ROOT / ".weights"
        if not (folder / "model.safetensors").exists():
            folder = Path(snapshot_download(C["model"], revision=C["revision"],
                                            allow_patterns=["*.json", "*.txt", "*.safetensors"]))
        with (folder / "model.safetensors").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != C["weights_sha256"]:
                raise ValueError("Model weight checksum mismatch")
        self.tokenizer = AutoTokenizer.from_pretrained(folder, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=torch.bfloat16,
                                                          attn_implementation="sdpa").eval()

    def prefix(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def encode(self, sequences):
        return self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")

    def hidden(self, encoded, **kwargs):
        positions = (encoded.attention_mask.cumsum(1) - 1).clamp_min(0)
        return self.model.model(**encoded, position_ids=positions, use_cache=False, **kwargs)

    def directions(self, pairs):
        deltas = {layer: [] for layer in C["layers"]}
        for group in batches(pairs):
            seqs = [self.prefix(chat(BASE, q)) + self.tokenizer.encode(a, add_special_tokens=False)
                    for q, warm, neutral in group for a in (warm, neutral)]
            states = self.hidden(self.encode(seqs), output_hidden_states=True).hidden_states
            for layer, values in deltas.items():
                ends = states[layer + 1][:, -1].float()
                values.extend(ends[::2] - ends[1::2])
        return {k: torch.stack(v).mean(0) for k, v in deltas.items()}

    @contextlib.contextmanager
    def steer(self, spec, boundary):
        if spec is None:
            yield
            return
        layer, vector = spec

        def add(_module, _args, output):
            h, *rest = output
            # During cached decoding only the new token is present.
            mask = True if h.shape[1] == 1 else torch.arange(h.shape[1])[None, :] >= boundary[:, None]
            return (h + vector.to(h.dtype) * torch.as_tensor(mask)[..., None], *rest)

        handle = self.model.model.layers[layer].register_forward_hook(add)
        try:
            yield
        finally:
            handle.remove()

    def style_scores(self, rows, spec=None, system=BASE):
        result = []
        for group in batches(rows):
            seqs, lengths = [], []
            for q, warm, neutral in group:
                for answer in (warm, neutral):
                    tokens = self.tokenizer.encode(answer, add_special_tokens=False)
                    seqs.append(self.prefix(chat(system, q)) + tokens)
                    lengths.append(len(tokens))
            encoded = self.encode(seqs)
            boundary = encoded.input_ids.shape[1] - torch.tensor(lengths) - 1
            with self.steer(spec, boundary):
                states = self.hidden(encoded).last_hidden_state
                for i in range(len(group)):
                    scores = []
                    for j in (2*i, 2*i+1):
                        start = boundary[j].item()
                        logits = self.model.lm_head(states[j, start:-1]).float()
                        gold = encoded.input_ids[j, start+1:]
                        scores.append(-torch.nn.functional.cross_entropy(logits, gold).item())
                    result.append(dict(question=group[i][0], warm=scores[0], neutral=scores[1],
                                       margin=scores[0] - scores[1]))
        return result

    def generate(self, conversations, spec=None, limit=None):
        replies = []
        for group in batches(conversations):
            encoded = self.encode([self.prefix(c) for c in group])
            width = encoded.input_ids.shape[1]
            with self.steer(spec, torch.full((len(group),), width-1)):
                out = self.model.generate(**encoded, do_sample=False, max_new_tokens=limit or C["generation_tokens"],
                                          pad_token_id=self.tokenizer.pad_token_id)
            replies.extend(self.tokenizer.batch_decode(out[:, width:], skip_special_tokens=True))
        return replies


def trials(model, tasks, mode, phase, spec=None):
    system = WARM if mode == "warm_prompt" else BASE
    forced = phase == "provided"
    turns = [chat(system, t["prompt"] + "\n" + (t["fact"] + "\n" + FINAL if forced else CHOICE))
             for t in tasks]
    first = model.generate(turns, spec)
    lookups = [i for i, r in enumerate(first) if not forced and parse(r) == "LOOKUP"]
    for i in lookups:
        turns[i].extend([dict(role="assistant", content=first[i]),
                         dict(role="user", content=tasks[i]["fact"] + "\n" + FINAL)])
    follow = dict(zip(lookups, model.generate([turns[i] for i in lookups], spec), strict=True))
    for i, (t, r) in enumerate(zip(tasks, first, strict=True)):
        action = parse(r)
        final = parse(follow.get(i, r))
        yield dict(id=t["id"], necessary=t["necessary"], gold=t["gold"], mode=mode, phase=phase,
                   messages=turns[i], first=r, followup=follow.get(i), lookup=i in follow,
                   malformed=action is None or final not in ("ANSWER A", "ANSWER B"),
                   correct=final == "ANSWER " + t["gold"])


@torch.inference_mode()
def run(out, smoke):
    from analysis import summarize
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(C["threads"])
    torch.manual_seed(C["seed"])
    tasks = json.loads((ROOT / "tasks.json").read_text())
    styles = json.loads((ROOT / "style.json").read_text())
    dev = [t for t in tasks if t["split"] == "calibration"]
    test = [t for t in tasks if t["split"] == "evaluation"]
    meta = dict(config=C, smoke=smoke, date_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                platform=platform.platform(), cpu=platform.processor(), dtype="bfloat16",
                inputs_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                               for p in ("tasks.json", "style.json", "config.json","experiment.py","analysis.py")})
    save_json(out / "manifest.json", meta)
    model = Experiment()
    if smoke:
        rows = list(trials(model, dev[:10], "neutral", "choice"))
        save_json(out / "smoke.json", rows)
    else:
        vectors = model.directions(styles["train"])
        search = []
        for layer, v in vectors.items():
            for alpha in C["strengths"]:
                spec = (layer, v * alpha)
                scores = model.style_scores(styles["calibration"], spec)
                check = list(trials(model, dev, "caa", "choice", spec))
                search.append(dict(layer=layer, alpha=alpha, scores=scores, trials=check,
                                   valid=sum(not r["malformed"] for r in check),
                                   margin=sum(s["margin"] for s in scores)/len(scores)))
        best = max(search, key=lambda s: (s["valid"], s["margin"]))
        layer, alpha = best["layer"], best["alpha"]
        v = vectors[layer] * alpha
        noise = torch.randn(v.shape, generator=torch.Generator().manual_seed(C["seed"]))
        noise = noise * (v.norm() / noise.norm())
        specs = dict(neutral=None, warm_prompt=None, caa=(layer, v), random=(layer, noise))
        save_json(out / "calibration.json", dict(candidates=search, layer=layer, alpha=alpha,
                                                vector_norm=v.norm().item(), random_norm=noise.norm().item()))
        torch.save(dict(caa=v, random=noise), out / "vectors.pt")
        style_checks = {}
        with (out / "responses.jsonl").open("x", encoding="utf-8") as log:
            for mode, spec in specs.items():
                system = WARM if mode == "warm_prompt" else BASE
                probes = styles["evaluation"]
                style_checks[mode] = model.style_scores(probes, spec, system)
                examples = model.generate([chat(system, q) for q, _, _ in probes], spec, limit=32)
                for score, reply in zip(style_checks[mode], examples, strict=True):
                    score["free_reply"] = reply
                for phase in ("choice", "provided"):
                    for row in trials(model, test, mode, phase, spec):
                        log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(mode, phase, len(test), flush=True)
        save_json(out / "style-check.json", style_checks)
        summarize(out)
    meta.update(completed=True, seconds=round(time.monotonic()-started, 2))
    save_json(out / "manifest.json", meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.out or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f"), args.smoke)
