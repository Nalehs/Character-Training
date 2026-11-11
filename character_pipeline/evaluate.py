"""Step 8: score one arm in the pushback environment, start to finish.

Starts `character_pipeline.serve` for the arm, runs the `character-pushback`
verifiers environment against it, stops the server and writes summary.json
(hold / drift / cave / update rates with denominators). The prompted base gets
the character's bible as its system prompt (prompted_base; prompted_instruct
runs the post-trained sibling the same way); trained arms get none, because
the character should be in the weights.

The verifiers command line is `vf-eval` in verifiers 0.4.0 and `vf-run` in
0.4.1; whichever is installed is used.

Decoding. verifiers sends no temperature, so the server's default applies:
greedy unless --temperature is given. Under greedy decoding every rollout of a
probe is the same reply, so --rollouts above 1 repeats one sample; the summary
records this as "rollouts_independent": false and its intervals are too narrow
by that factor. With --temperature above 0 rollouts differ; they are seeded
once per server (--seed), which is reproducible only at fixed concurrency.

summary.json is written only when no rollout errored. If any did (for
example the judge was unreachable) the numbers go to summary.incomplete.json
and the command exits 1, unless --allow-failed.

    python -m character_pipeline.evaluate --arm dpo --adapter runs/golem-15/dpo/seed0/adapter \
        --character characters/golem-15 --out runs/golem-15/eval/dpo-seed0

Needs the d1 judge (scripts/serve_d1.sh) unless --judge-url none, in which case
hold means "kept the answer" and drift is not measured.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from character_pipeline.train.common import STUDENT, STUDENT_REVISION


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(url: str, proc, timeout: float = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("policy server exited before it was ready")
        try:
            urllib.request.urlopen(url + "/models", timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError("policy server did not come up")


def vf_cli() -> list[str]:
    """The installed verifiers runner: `vf-eval --no-push` (0.4.0) or `vf-run` (0.4.1, which has no push)."""
    for name, extra in (("vf-eval", ["--no-push"]), ("vf-run", [])):
        if shutil.which(name):
            return [name, *extra]
    raise SystemExit("neither vf-eval nor vf-run is on PATH; pip install -e environments/character_pushback")


def eval_command(arm: str, character: Path, base_url: str, judge_url: str, rollouts: int, out: Path,
                 limit: int | None = None, cli: list[str] | None = None) -> list[str]:
    name, *extra = cli or vf_cli()
    cmd = [name, "character-pushback", "--model", arm, "--client.base-url", base_url,
           "--client.api-key-var", "POLICY_KEY", "--env.assistant.runtime.type", "subprocess",
           "--env.taskset.character-dir", str(Path(character).resolve()),
           "--env.taskset.system-prompt-mode", "character" if arm.startswith("prompted") else "none",
           "-r", str(rollouts), *extra, "--output-dir", str(out)]
    if judge_url == "none":
        cmd += ["--env.taskset.task.use-judge", "false"]
    else:
        cmd += ["--env.taskset.task.judge-url", judge_url]
    if limit:
        cmd += ["-n", str(limit)]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--character", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:8090")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--limit", type=int, help="first n probes only (smoke tests)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature for the arm; 0 = greedy, and then rollouts are repeats")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for the policy server")
    ap.add_argument("--allow-failed", action="store_true", help="write summary.json even if rollouts failed")
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--revision", default=STUDENT_REVISION)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args(argv)
    if args.rollouts < 1 or args.temperature < 0:
        ap.error("--rollouts must be at least 1 and --temperature must not be negative")
    if args.arm.startswith("prompted") and args.adapter:
        ap.error("a prompted arm is the model without an adapter; drop --adapter or rename the arm")
    if args.adapter and not (args.adapter / "adapter_config.json").is_file():
        raise SystemExit(f"{args.adapter} is not a LoRA adapter folder (no adapter_config.json)")
    from character_pushback.summarize import summarize

    cli = vf_cli()
    raw = args.out / "vf"
    if raw.exists() and any(raw.iterdir()):
        # An earlier attempt's traces must never be summarised as this one's.
        raise SystemExit(f"{raw} already holds a run; move it away before evaluating into {args.out} again")
    port = free_port()
    serve = [sys.executable, "-m", "character_pipeline.serve", "--name", args.arm, "--port", str(port),
             "--model", args.model, "--revision", args.revision, "--device", args.device,
             "--temperature", str(args.temperature), "--seed", str(args.seed)]
    if args.adapter:
        serve += ["--adapter", str(args.adapter)]
    if args.local_files_only:
        serve += ["--local-files-only"]
    args.out.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}/v1"
    env = os.environ | {"POLICY_KEY": os.environ.get("POLICY_KEY", "local")}
    with (args.out / "server.log").open("w") as server_log, (args.out / "vf-eval.log").open("w") as vf_log:
        proc = subprocess.Popen(serve, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            try:
                wait_ready(base_url, proc)
            except (RuntimeError, TimeoutError) as e:
                raise SystemExit(f"{e}; see {args.out / 'server.log'}") from None
            done = subprocess.run(eval_command(args.arm, args.character, base_url, args.judge_url, args.rollouts,
                                               raw, args.limit, cli), check=False, env=env,
                                  stdout=vf_log, stderr=subprocess.STDOUT)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    if done.returncode != 0:
        raise SystemExit(f"{cli[0]} exited with status {done.returncode}; see {args.out / 'vf-eval.log'}")
    runs = sorted(raw.glob("character-pushback--*"))
    if len(runs) != 1:
        raise SystemExit(f"expected one run directory under {raw}, found {len(runs)}; see {args.out / 'vf-eval.log'}")
    run_dir = runs[0]
    greedy = args.temperature == 0
    summary = summarize(run_dir) | {
        "arm": args.arm, "adapter": str(args.adapter or ""), "run_dir": str(run_dir),
        "model": args.model, "revision": args.revision,
        "decoding": {"temperature": args.temperature, "seed": args.seed, "rollouts": args.rollouts,
                     "rollouts_independent": not (greedy and args.rollouts > 1)}}
    totals = summary["totals"]
    broken = totals["rollouts"] == 0 or totals["failed"]  # incomplete rollouts stay counted in the summary
    name = "summary.incomplete.json" if broken and not args.allow_failed else "summary.json"
    (args.out / name).write_text(json.dumps(summary, indent=2) + "\n")
    if greedy and args.rollouts > 1:
        print(f"note: greedy decoding, so the {args.rollouts} rollouts of each probe are repeats of one reply; "
              "counts and intervals overstate the number of independent samples", file=sys.stderr)
    if name != "summary.json":
        raise SystemExit(f"{totals['failed']} of {totals['rollouts']} rollouts failed: wrote {args.out / name}, not summary.json. See {args.out / 'vf-eval.log'}; "
                         "pass --allow-failed to accept a partial run.")
    print(json.dumps({k: summary["insistence"].get(k, {}).get("rate") for k in ("hold", "drift", "cave")}
                     | {"updated": summary["evidence"]["updated"]["rate"]}))


if __name__ == "__main__":
    main()
