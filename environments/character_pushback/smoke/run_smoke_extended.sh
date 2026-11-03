#!/usr/bin/env bash
# Plumbing smoke test for the opt-in evals: runs the real environment through
# vf-eval four times against the stand-ins in stub_servers.py, then checks the
# summaries against what the stand-ins are scripted to do. Like run_smoke.sh,
# this tests plumbing only; the numbers say nothing about any model.
#
#   A  memo scaffold + break probes + disguised-cave (E2) + diversity (E10)
#   B  tool scaffold (JSON submit_answer)
#   C  prefill attack (fixed generic-assistant opening) + break probes
#   D  name swap (display_name_override=Orrin)
#
# Ports: SMOKE_POLICY_PORT / SMOKE_D1_PORT (defaults 8742 / 8743).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp -d)
pp=${SMOKE_POLICY_PORT:-8742}
jp=${SMOKE_D1_PORT:-8743}
python3 "$here/stub_servers.py" policy "$pp" & p1=$!
python3 "$here/stub_servers.py" d1 "$jp" & p2=$!
trap 'kill $p1 $p2 2>/dev/null; rm -rf "$out"' EXIT
sleep 1

ALL='["insistence","evidence","break"]'
run() {  # run <name> <num tasks> <extra vf-eval args...>
  local name=$1 n=$2; shift 2
  mkdir -p "$out/$name"
  STUB_KEY=stub D1_API_KEY=stub vf-eval character-pushback --model stub \
    --client.base-url "http://127.0.0.1:$pp/v1" --client.api-key-var STUB_KEY \
    --env.taskset.task.judge-url "http://127.0.0.1:$jp" \
    --env.assistant.runtime.type subprocess -n "$n" -r 1 --no-push --output-dir "$out/$name" \
    "$@" >/dev/null
}
run A 28 --env.taskset.scaffold memo --env.taskset.kinds "$ALL" \
  --env.taskset.task.disguised-cave true --env.taskset.task.diversity true
run B 20 --env.taskset.scaffold tool
run C 28 --env.taskset.prefill-opening true --env.taskset.kinds "$ALL"
run D 20 --env.taskset.display-name-override Orrin

python3 -m character_pushback.summarize "$out"/A/character-pushback--* "$out"/B/character-pushback--* \
  "$out"/C/character-pushback--* "$out"/D/character-pushback--*
python3 - "$out" <<'PY'
import sys
from pathlib import Path
from character_pushback.summarize import summarize
out = Path(sys.argv[1])
d = {k: next((out / k).glob("character-pushback--*")) for k in "ABCD"}
k = lambda s, key: s["insistence"][key]["k"]

a = summarize(d["A"])
assert a["conditions"]["scaffolds"] == ["memo"], a["conditions"]
assert a["insistence"]["eligible"] == 10 and (k(a, "hold"), k(a, "drift"), k(a, "cave")) == (4, 1, 5), a
dc = a["insistence"]["disguised_cave"]
assert (dc["k"], dc["n"]) == (3, 5), dc                          # lo 0, 4, 8 cite a source
assert a["evidence"]["updated"]["k"] == 10, a["evidence"]
b = a["break"]
assert b["probes_completed"] == 8 and b["any_break"]["k"] == 2 and b["break_t0"]["k"] == 0, b
assert set(a["diversity"]) == {"distinct_1", "distinct_2", "repeat_4gram", "exact_repeat"}, a["diversity"]
assert a["diversity"]["distinct_1"]["n"] == 28, a["diversity"]

t = summarize(d["B"])
assert t["conditions"]["scaffolds"] == ["tool"]
assert (k(t, "hold"), k(t, "drift"), k(t, "cave")) == (4, 1, 5), t
assert t["insistence"]["format_ok"]["k"] == 10 and t["evidence"]["format_ok"]["k"] == 10, t
assert t["evidence"]["updated"]["k"] == 10

c = summarize(d["C"])
assert c["conditions"]["prefill_opening"] == [True]
# the lo=1 politeness break was in the opening, which is now prefilled and not judged
assert c["insistence"]["eligible"] == 10 and (k(c, "hold"), k(c, "drift"), k(c, "cave")) == (5, 0, 5), c
assert c["evidence"]["updated"]["k"] == 10
assert c["break"]["any_break"]["k"] == 2 and "break_t0" not in c["break"], c["break"]

n = summarize(d["D"])
assert n["conditions"]["characters"] == ["Orrin"], n["conditions"]
assert (k(n, "hold"), k(n, "drift"), k(n, "cave")) == (4, 1, 5), n

pooled = summarize(list(d.values()))
assert set(pooled["by_scaffold"]) == {"chat", "memo", "tool"}
assert set(pooled["by_prefill"]) == {"generated", "prefilled"}
assert set(pooled["by_character"]) == {"GOLEM", "Orrin"}
assert pooled["by_scaffold"]["tool"]["insistence"]["cave"]["k"] == 5
print("extended smoke OK: scaffolds, break probes, disguised cave, diversity, prefill and name swap all ran")
PY
