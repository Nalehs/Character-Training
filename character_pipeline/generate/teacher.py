"""Stage 2: teacher generation against a local llama.cpp server.

Request body, stop strings, finish-reason mapping and mechanical flags are
ported from hdmurder/golem-15 gen/g6/upload/kernel.py (`dispatch`, `perform`,
`mechanical`), the G6 run against Qwen3-30B-A3B Q4_K_M. The Kaggle bootstrap,
artifact download and deadline machinery are left behind; start the server
yourself (see scripts/serve_teacher.sh) and point `--url` at it.

Partner turns come from a scenario file (scripted caller turns, as in G6).
Every reply is stored raw with its flags. Nothing here accepts or repairs a
turn: acceptance happens later, in the checker and human review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path

from character_pipeline.character import load_character
from character_pipeline.generate.controller import hygiene, render

STOP = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def request_body(prompt: str, seed: int, cap: int = 384, extractor: bool = False) -> dict:
    """G6 sampling settings: character turns at T=0.7, extractor calls greedy."""
    return {
        "prompt": prompt, "n_predict": cap,
        "temperature": 0.0 if extractor else 0.7, "top_p": 0.8,
        "top_k": 1 if extractor else 20, "min_p": 0.0,
        "presence_penalty": 0.0 if extractor else 1.5, "repeat_penalty": 1.0,
        "seed": seed, "stop": STOP, "cache_prompt": True, "stream": False,
    }


def finish_reason(response: dict) -> str:
    stop = response.get("stop_type")
    if response.get("stopped_limit") or stop == "limit":
        return "length"
    return "stop" if stop in ("eos", "word") else "unknown"


def mechanical(text: str, finish: str) -> list[str]:
    import re
    flags = []
    if not text.strip():
        flags.append("empty")
    if finish != "stop":
        flags.append("incomplete_or_unknown_finish")
    if re.search(r"<\|(?:im_start|im_end)|</?think>", text):
        flags.append("role_or_think_marker")
    return flags


def complete(url: str, body: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_scenario(scenario: dict, system: str, url: str, seed: int, name: str,
                 call=complete) -> dict:
    """Play the scripted caller turns against the character; keep every reply raw."""
    hist, turns = [], []
    for i, caller in enumerate(scenario["caller_turns"]):
        hist.append({"speaker": "partner", "text": caller})
        prompt = render(system, hist, "golem")
        body = request_body(prompt, seed + i)
        started = time.monotonic()
        response = call(url, body)
        text = response.get("content", "").strip()
        finish = finish_reason(response)
        flags = mechanical(text, finish) + [f for f in hygiene(text, finish, name=name)
                                            if f not in ("empty", "incomplete")]
        row = {"speaker": "golem", "text": text, "finish_reason": finish, "flags": flags,
               "request_sha256": canonical_hash(body), "response_sha256": canonical_hash(response),
               "tokens_predicted": response.get("tokens_predicted"),
               "elapsed_seconds": time.monotonic() - started}
        hist.append(row)
        turns.append(row)
    return {"scenario_id": scenario["id"], "seed": seed, "history": hist,
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "training_eligible": False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("character", type=Path)
    ap.add_argument("scenarios", type=Path, help="JSONL with id and caller_turns")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", type=Path, default=Path("runs/teacher.jsonl"))
    ap.add_argument("--seed", type=int, default=720300)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    character = load_character(args.character)
    system = character.system_prompt()
    scenarios = [json.loads(x) for x in args.scenarios.read_text().splitlines() if x.strip()]
    if args.limit:
        scenarios = scenarios[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with args.out.open("x") as f:  # never overwrite raw generations
        for scenario in scenarios:
            row = run_scenario(scenario, system, args.url, rng.randrange(1 << 30), character.name)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(scenario["id"], [t["flags"] for t in row["history"] if t["speaker"] == "golem"], flush=True)


if __name__ == "__main__":
    main()
