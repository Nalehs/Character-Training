"""Turn a vf-eval run directory into hold / drift / cave / update rates.

vf-eval averages every metric over all rollouts, so its printed `hold` is
divided by evidence probes too. This script uses the right denominators:

  insistence rates are over eligible rollouts (correct opening answer)
  update rates are over completed evidence rollouts

and prints a Wilson 95% interval beside each rate. Rollouts of one probe are
not independent samples, so treat the intervals as rough.

    python -m character_pushback.summarize outputs/<run-dir> [--json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rollouts(run_dir: Path):
    for line in (run_dir / "traces.jsonl").read_text().splitlines():
        episode = json.loads(line)
        data = episode["task"]["data"]
        for trace in episode["traces"]:
            yield data, trace.get("metrics") or {}, bool(trace.get("ok"))


def summarize(run_dir: Path) -> dict:
    count = {"insistence": {}, "evidence": {}}
    totals = {"rollouts": 0, "failed": 0, "incomplete": 0}
    for data, m, ok in rollouts(run_dir):
        totals["rollouts"] += 1
        if not ok:
            totals["failed"] += 1
            continue
        if m.get("complete") != 1.0:
            totals["incomplete"] += 1
            continue
        c = count[data["kind"]]
        c["n"] = c.get("n", 0) + 1
        for key, value in m.items():
            if value is not None and not key.startswith("p_break"):
                c[key] = c.get(key, 0) + int(value == 1.0)

    def rate(c, key, denom_key):
        k, n = c.get(key, 0), c.get(denom_key, 0)
        lo, hi = wilson(k, n)
        return {"k": k, "n": n, "rate": k / n if n else math.nan, "ci95": [lo, hi]}

    ins, ev = count["insistence"], count["evidence"]
    out = {"totals": totals,
           "insistence": {"probes_completed": ins.get("n", 0),
                          "eligible": ins.get("eligible", 0),
                          "initial_error": ins.get("initial_error", 0),
                          "initial_unclear": ins.get("initial_unclear", 0)},
           "evidence": {"probes_completed": ev.get("n", 0)}}
    for key in ("hold", "drift", "cave", "unclear", "cave_push1", "cave_push2"):
        if key in ins or key in ("hold", "cave", "unclear"):
            out["insistence"][key] = rate(ins, key, "eligible")
    for key in ("updated", "not_updated", "update_unclear"):
        out["evidence"][key] = rate(ev, key, "n")
    out["judge_used"] = "drift" in ins
    return out


def render(s: dict) -> str:
    def fmt(r):
        lo, hi = r["ci95"]
        return f'{r["k"]}/{r["n"]} = {r["rate"]:.2f}  (95% CI {lo:.2f}-{hi:.2f})' if r["n"] else "n/a"
    i, e, t = s["insistence"], s["evidence"], s["totals"]
    lines = [f'rollouts {t["rollouts"]}, failed {t["failed"]}, incomplete {t["incomplete"]}',
             f'insistence: {i["probes_completed"]} completed, {i["eligible"]} eligible, '
             f'{i["initial_error"]} opening errors, {i["initial_unclear"]} unreadable openings']
    for key in ("hold", "drift", "cave", "unclear", "cave_push1", "cave_push2"):
        if key in i:
            lines.append(f"  {key:<11}{fmt(i[key])}")
    if not s["judge_used"]:
        lines.append("  (judge off: hold = kept the answer; drift not measured)")
    lines.append(f'evidence: {e["probes_completed"]} completed')
    for key in ("updated", "not_updated", "update_unclear"):
        lines.append(f"  {key:<15}{fmt(e[key])}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    s = summarize(args.run_dir)
    print(json.dumps(s, indent=2) if args.json else render(s))


if __name__ == "__main__":
    main()
