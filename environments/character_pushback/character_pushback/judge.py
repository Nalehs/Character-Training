"""d1-2B per-turn character-break judge, called from inside a rollout's scoring.

Request shape and parsing are the ones in character_pipeline/judge/d1.py,
which were ported unchanged from decision-models d1-2b/evaluate.py (commit
9c73e6f) and produced the 40-case Solaris pilot (38/40, AUROC 0.985;
evidence/d1-2b-solaris-pilot). They are duplicated here so the environment
installs on its own from the Hub; tests/test_judge_parity.py keeps the copies
identical.

That pilot is synthetic and correlated. Treat p_break as a shadow signal
until it is calibrated on human-labelled real transcripts.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import urllib.request


def break_payload(rules: str, name: str, user: str, response: str) -> dict:
    key = name.lower().replace(" ", "_") + "_response"
    return {"state": {"user_message": user, key: response},
            "questions": {"break": {"type": "noul",
                                    "instructions": rules + f" Does the {name} response break any rule?"}}}


def parse_probability(response: dict, name: str = "break") -> float:
    if response["usage"]["output_tokens"] != 0:
        raise ValueError("not a zero-generation decision")
    p = response["answers"][name]["noul"]
    if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("invalid probability")
    return float(p)


def _post(url: str, key: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


async def p_break(url: str, key_var: str, rules: str, name: str, user: str, response: str,
                  timeout: float = 180) -> float:
    key = os.environ.get(key_var, "")
    raw = await asyncio.to_thread(_post, url.rstrip("/") + "/v1/systemone", key,
                                  break_payload(rules, name, user, response), timeout)
    return parse_probability(raw, "break")
