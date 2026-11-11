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
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from character_pipeline.character import load_character
from character_pipeline.data.io import read_jsonl, resumable_jsonl
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
    flags = []
    if not text.strip():
        flags.append("empty")
    if finish != "stop":
        flags.append("incomplete_or_unknown_finish")
    if re.search(r"<\|(?:im_start|im_end)|</?think>", text):
        flags.append("role_or_think_marker")
    return flags


def complete(url: str, body: dict, timeout: float = 180.0, retries: int = 2, backoff: float = 2.0) -> dict:
    """POST one completion. Connection errors, timeouts and HTTP 5xx are retried with the
    identical body (same seed); HTTP 4xx and a non-JSON body raise at once."""
    req = urllib.request.Request(url.rstrip("/") + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                response = json.load(r)
            if not isinstance(response, dict):
                raise ValueError(f"teacher server returned {type(response).__name__}, expected a JSON object")
            return response
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            client_error = isinstance(e, urllib.error.HTTPError) and e.code < 500
            if client_error or attempt == retries:
                raise
            time.sleep(backoff * (attempt + 1))


def run_scenario(scenario: dict, system: str, url: str, seed: int, name: str,
                 call=complete) -> dict:
    """Play the scripted caller turns against the character; keep every reply raw."""
    hist, turns = [], []
    for i, caller in enumerate(scenario["caller_turns"]):
        hist.append({"speaker": "partner", "text": caller})
        prompt = render(system, hist, "character")
        body = request_body(prompt, seed + i)
        started = time.monotonic()
        response = call(url, body)
        content = response.get("content")
        text = content.strip() if isinstance(content, str) else ""  # missing content is flagged "empty"
        finish = finish_reason(response)
        flags = mechanical(text, finish) + [f for f in hygiene(text, finish, name=name)
                                            if f not in ("empty", "incomplete")]
        row = {"speaker": "character", "text": text, "finish_reason": finish, "flags": flags,
               "request_sha256": canonical_hash(body), "response_sha256": canonical_hash(response),
               "tokens_predicted": response.get("tokens_predicted"),
               "elapsed_seconds": time.monotonic() - started}
        hist.append(row)
        turns.append(row)
    return {"scenario_id": scenario["id"], "seed": seed, "history": hist,
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "training_eligible": False}


def scenario_seeds(scenarios: list[dict], seed: int) -> list[int]:
    """One seed per scenario, drawn in file order, so a resumed run gets the seeds it would have had."""
    rng = random.Random(seed)
    return [rng.randrange(1 << 30) for _ in scenarios]


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
    scenarios = read_jsonl(args.scenarios)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    ids = [s["id"] for s in scenarios]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{args.scenarios}: duplicate scenario ids")
    seeds = scenario_seeds(scenarios, args.seed)
    system_sha = hashlib.sha256(system.encode()).hexdigest()
    # Raw generations are never overwritten. Rows go to <out>.partial and the file is renamed
    # when every scenario is done, so an interrupted run is resumed, not mistaken for a full one.
    with resumable_jsonl(args.out) as (done, write):
        expected = [(i, s, system_sha) for i, s in zip(ids, seeds)][: len(done)]
        if len(done) > len(ids) or [(r["scenario_id"], r["seed"], r["system_sha256"]) for r in done] != expected:
            raise SystemExit(f"{args.out}.partial was written with other scenarios, seed or system prompt; "
                             "delete it to start again")
        for scenario, seed in list(zip(scenarios, seeds))[len(done):]:
            row = run_scenario(scenario, system, args.url, seed, character.name)
            write(row)
            print(scenario["id"], [t["flags"] for t in row["history"] if t["speaker"] != "partner"], flush=True)


if __name__ == "__main__":
    main()
