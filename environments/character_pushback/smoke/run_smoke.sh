#!/usr/bin/env bash
# Plumbing smoke test: runs the real environment through vf-eval against the
# canned stand-ins in stub_servers.py (no model, no judge, no GPU), then checks
# that the summary matches what the stand-ins are scripted to do:
# 10 eligible insistence probes -> 4 hold, 1 drift, 5 cave; 10/10 updates.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp -d)
pp=${SMOKE_POLICY_PORT:-8701}
jp=${SMOKE_D1_PORT:-8702}
python3 "$here/stub_servers.py" policy "$pp" & p1=$!
python3 "$here/stub_servers.py" d1 "$jp" & p2=$!
trap 'kill $p1 $p2 2>/dev/null; rm -rf "$out"' EXIT
sleep 1
STUB_KEY=stub D1_API_KEY=stub vf-eval character-pushback --model stub \
  --client.base-url http://127.0.0.1:$pp/v1 --client.api-key-var STUB_KEY \
  --env.taskset.task.judge-url http://127.0.0.1:$jp \
  --env.assistant.runtime.type subprocess -n 20 -r 1 --no-push --output-dir "$out" >/dev/null
run=$(ls -d "$out"/character-pushback--*)
python3 -m character_pushback.summarize "$run"
python3 - "$run" <<'PY'
import sys
from pathlib import Path
from character_pushback.summarize import summarize
s = summarize(Path(sys.argv[1]))
got = {k: s["insistence"][k]["k"] for k in ("hold", "drift", "cave")}
assert s["insistence"]["eligible"] == 10, s
assert got == {"hold": 4, "drift": 1, "cave": 5}, got
assert s["evidence"]["updated"]["k"] == 10, s
print("smoke OK: environment, scripted caller, judge calls and summary all ran")
PY
