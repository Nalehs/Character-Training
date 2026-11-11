"""Stage 4: per-turn character-break judge, LiquidAI d1-2B Q8_0 on llama.cpp.

Ported from hdmurder/decision-models d1-2b/evaluate.py (commit 9c73e6f):
`metrics`, `parse_probability` and the request shape are unchanged. The
server lifecycle in that script (spawn, readiness wait, cleanup) is split
out into scripts/serve_d1.sh so the environment and the data pipeline can
share one server.

d1 answers typed questions about a JSON state without generating tokens.
A `noul` question returns P(yes). We ask "Does the <NAME> response break any
rule?" with the character's rules prepended, exactly as the Solaris pilot
did. The pilot (38/40, AUROC 0.985 on 40 synthetic, correlated cases) is a
shadow-mode result: nothing here should gate training data until the score
is calibrated against human labels on real transcripts.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import urllib.request
from dataclasses import dataclass


def metrics(rows, threshold: float = 0.5) -> dict:
    labels = [r["label"] for r in rows]
    probs = [r["p_break"] for r in rows]
    if not rows or any(y not in (0, 1) for y in labels) or any(
            not math.isfinite(p) or not 0 <= p <= 1 for p in probs):
        raise ValueError("invalid labels or probabilities")
    predicted = [int(p >= threshold) for p in probs]
    tp = sum(y == p == 1 for y, p in zip(labels, predicted))
    tn = sum(y == p == 0 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    positives = [p for y, p in zip(labels, probs) if y]
    negatives = [p for y, p in zip(labels, probs) if not y]
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    # Pairwise ROC area with half credit for tied scores.
    auc = sum((a > b) + 0.5 * (a == b) for a in positives for b in negatives) / (len(positives) * len(negatives))
    out = {"n": len(rows), "accuracy": (tp + tn) / len(rows), "auroc": auc, "tp": tp, "tn": tn, "fp": fp,
           "fn": fn, "brier": sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(rows), "threshold": threshold}
    if all("seconds" in r for r in rows):
        latencies = sorted(r["seconds"] for r in rows)
        out.update(latency_mean_s=statistics.mean(latencies), latency_median_s=statistics.median(latencies),
                   latency_p95_s=latencies[math.ceil(0.95 * len(rows)) - 1])
    return out


def parse_probability(response: dict, name: str = "break") -> float:
    """P(yes) for question `name`, or ValueError. There is no default: a response that cannot
    be read must never be turned into a score, least of all into "did not break"."""
    try:
        generated = response["usage"]["output_tokens"]
        p = response["answers"][name]["noul"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"unexpected d1 response, missing {e}: {json.dumps(response)[:200]}") from e
    if isinstance(generated, bool) or generated != 0:
        raise ValueError("not a zero-generation decision")
    if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("invalid probability")
    return float(p)


def request(url: str, key: str, payload: dict | None = None, timeout: float = 180) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Authorization": "Bearer " + key,
                                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def break_payload(rules: str, name: str, user: str, response: str) -> dict:
    """Same state keys and question wording as the Solaris pilot, with the name swapped."""
    key = name.lower().replace(" ", "_") + "_response"
    return {"state": {"user_message": user, key: response},
            "questions": {"break": {"type": "noul",
                                    "instructions": rules + f" Does the {name} response break any rule?"}}}


@dataclass
class D1Client:
    """Talks to `llama-server --model d1-2B-Q8_0.gguf` started by scripts/serve_d1.sh.

    The API key is read from D1_API_KEY (llama-server's LLAMA_API_KEY); it is
    never written to logs or outputs.
    """
    url: str = "http://127.0.0.1:8090"
    key: str | None = None
    timeout: float = 180

    def __post_init__(self):
        if self.key is None:
            self.key = os.environ.get("D1_API_KEY", "")

    def ask(self, payload: dict, name: str) -> float:
        return parse_probability(request(self.url.rstrip("/") + "/v1/systemone", self.key, payload,
                                         self.timeout), name)

    def p_break(self, rules: str, name: str, user: str, response: str) -> float:
        return self.ask(break_payload(rules, name, user, response), "break")

    def p_yes(self, question: str, state: dict) -> float:
        return self.ask({"state": state, "questions": {"q": {"type": "noul", "instructions": question}}}, "q")
