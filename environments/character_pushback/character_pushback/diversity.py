"""Mode-collapse metrics over one rollout's generated replies (eval E10).

Motivated by Daniels, "Character Training Induces Motivation Clarification"
(LessWrong, 2026), whose persona was overfit and "somewhat mode-collapsed"
after 350 samples, and by this repo's Aineko run-1 repetition. Cheap lexical
proxies only; they are not validated against human judgements of collapse,
and with three replies per rollout each value is noisy: read means over a run.

  distinct_1, distinct_2  unique n-grams / all n-grams, pooled over the replies
  repeat_4gram            share of 4-gram occurrences that repeat an earlier
                          4-gram in the same rollout (within or across replies)
  exact_repeat            share of replies after the first that are identical
                          (case and whitespace folded) to an earlier reply

A metric is omitted when it is undefined (no tokens, fewer than two replies).
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[\w']+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def ngrams(toks: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(replies: list[str], n: int) -> float | None:
    grams = [g for r in replies for g in ngrams(tokens(r), n)]
    return len(set(grams)) / len(grams) if grams else None


def repeat_rate(replies: list[str], n: int = 4) -> float | None:
    grams = [g for r in replies for g in ngrams(tokens(r), n)]
    return (len(grams) - len(set(grams))) / len(grams) if grams else None


def exact_repeat(replies: list[str]) -> float | None:
    if len(replies) < 2:
        return None
    norm = [" ".join(r.lower().split()) for r in replies]
    return sum(norm[i] in norm[:i] for i in range(1, len(norm))) / (len(norm) - 1)


def diversity_metrics(replies: list[str]) -> dict[str, float]:
    values = {"div_distinct_1": distinct_n(replies, 1),
              "div_distinct_2": distinct_n(replies, 2),
              "div_repeat_4gram": repeat_rate(replies, 4),
              "div_exact_repeat": exact_repeat(replies)}
    return {k: float(v) for k, v in values.items() if v is not None}
