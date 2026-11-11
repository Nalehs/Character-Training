"""The whole pipeline for one character, as an ordered list of resumable steps.

Every step names its inputs and outputs under runs/<character-id>/. A finished
step is skipped, so `character-pipeline run <character>` can be re-run after
any interruption. Two steps need people:

  review.jsonl      accept/reject for each character turn (make a sheet with
                    `character-pipeline review-sheet`)
  kto_review.jsonl  optional absolute good/bad labels; without it the KTO arm
                    is skipped, never faked from DPO pairs

The run stops at a missing human input and says what to write. Nothing here
starts the teacher or judge servers; scripts/serve_teacher.sh and
scripts/serve_d1.sh do, once per machine.

What "finished" means. A step is done when its outputs exist and no attempt at
it was left unfinished. Each step runs as its own process (`python -m <module>
<args>`), so its exit status is checked and its memory is returned before the
next step. Around each attempt the runner keeps three small files in
runs/<id>/.steps/:

  <step>.running  written before the attempt, removed after it succeeds. If it
                  is still there on the next run, the attempt crashed or was
                  killed: whatever it left is moved to runs/<id>/.interrupted/
                  (never deleted) and the step runs again.
  <step>.json     the receipt: the command and a SHA-256 of every input file
                  (and of the character's bible, rules and judge settings).
                  On a later run a finished step whose command or inputs no
                  longer match is reported as [stale] and the run stops; move
                  its outputs away to redo it, or pass --reuse-stale.
  <step>.lock     held while the step runs, so two runs of one character (for
                  example one seed per GPU) wait for each other instead of
                  writing the same files.

Outputs that were put there by hand, with no receipt, count as done and are
not checked for staleness.
"""
from __future__ import annotations

import contextlib
import difflib
import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from character_pipeline.train.common import STUDENT, STUDENT_REVISION

try:
    import fcntl
except ImportError:  # not POSIX: no locking between concurrent runs
    fcntl = None

DEFAULTS = {
    "scenarios": "scenarios.jsonl",
    "teacher_url": "http://127.0.0.1:8080",
    "judge_url": "http://127.0.0.1:8090",
    "seeds": [0, 1, 2],
    "sft": {"epochs": 2, "lr": 1e-4, "seq_len": 2048},
    "preference": {"lr": 1e-6, "beta": 0.1, "max_length": 2048},
    "continued_sft": {"lr": 1e-5, "epochs": 1},
    "eval": {"rollouts": 3},
}
ARMS = ("sft", "dpo", "kto", "continued_sft")
# Every key a recipe's or a character's "pipeline" block may set; anything else is a typo.
PIPELINE_KEYS = {
    "scenarios": None, "teacher_url": None, "judge_url": None, "seeds": None,
    "sft": {"epochs", "lr", "seq_len"},
    "preference": {"lr", "beta", "max_length"},
    "continued_sft": {"lr", "epochs"},
    "eval": {"rollouts", "limit", "temperature"},
    "lora": {"rank", "alpha"},
}
RECIPE_KEYS = {"name", "schema_version", "summary", "status", "doc", "cite", "arms", "pipeline", "sources",
               "untuned", "student", "teacher", "constitution", "stages", "evaluation", "deviations",
               "oct_internal_discrepancies", "eval_arms", "runner"}
STAGE_KEYS = {"id", "oct_stage", "module", "args", "implemented", "hyperparameters", "sources", "note",
              "deviation", "deviations"}
TEMPLATES = ("{runs}", "{character}", "{seed}", "{teacher_url}", "{judge_url}")
NOT_SETTINGS = {"--device": 1, "--local-files-only": 0}  # flag -> values it takes; left out of the receipt


_HASHES: dict[tuple, str] = {}


def _sha_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _HASHES:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _HASHES[key] = h.hexdigest()
    return _HASHES[key]


def character_digest(folder: Path) -> str:
    """What a stage can read from a character folder: bible, private core/style, judge rules and settings.

    Deliberately not the whole folder or the whole config: editing `status` or the `pipeline` block
    must not make the teacher chats look stale. Hyperparameters reach a step through its command.
    """
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    parts = {"id": config.get("id"), "display_name": config.get("display_name"), "judge": config.get("judge")}
    names = {config.get("bible"), (config.get("judge") or {}).get("rules_file"), "core.md", "style.md"}
    for name in sorted(n for n in names if n):
        path = folder / name
        parts[name] = _sha_file(path) if path.is_file() else None
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def _digest(path: Path) -> str | None:
    """Content hash of one input path, or None for paths that are not inputs we can pin."""
    if path.is_file():
        return _sha_file(path)
    if (path / "adapter_config.json").is_file():
        weights = [p for p in sorted(path.glob("adapter_model.*")) if p.is_file()]
        return hashlib.sha256(" ".join(
            _sha_file(p) for p in [path / "adapter_config.json", *weights]).encode()).hexdigest()
    return None


@dataclass
class Step:
    name: str
    module: str
    args: list
    outputs: list[Path]
    needs: list[Path] = field(default_factory=list)
    human: str | None = None      # what a person must produce if `needs` is missing
    optional: bool = False        # skip (not stop) when `needs` is missing
    blocked: str | None = None    # a reason this step cannot run yet that is not a missing file
    marker: Path | None = None    # written after success, for steps whose outputs are directories
    state: Path | None = None     # runs/<id>/.steps; None = no receipts, locks or interruption tracking
    character: Path | None = None  # the character folder, hashed by content when it is an argument
    run_dir: Path | None = None   # runs/<id>, so receipts do not depend on where runs/ lives

    # ---- completion -------------------------------------------------------------------------
    def _state(self, suffix: str) -> Path | None:
        return self.state / f"{self.name.replace('/', '--')}{suffix}" if self.state else None

    def interrupted(self) -> bool:
        running = self._state(".running")
        return bool(running and running.exists())

    def done(self) -> bool:
        return all(p.exists() for p in self.outputs) and not self.interrupted()

    def ready(self) -> bool:
        return self.blocked is None and all(p.exists() for p in self.needs)

    # ---- command ----------------------------------------------------------------------------
    def argv(self) -> list[str]:
        return [sys.executable, "-m", self.module, *map(str, self.args)]

    def command(self) -> str:
        return shlex.join(["python", "-m", self.module, *map(str, self.args)])

    def products(self) -> list[Path]:
        """Paths this step creates: every --out / --*-out value, the outputs and the marker."""
        args = [str(a) for a in self.args]
        outs = [Path(v) for flag, v in itertools.pairwise(args) if flag == "--out" or flag.endswith("-out")]
        return list(dict.fromkeys(outs + [Path(p) for p in self.outputs] + ([self.marker] if self.marker else [])))

    # ---- receipt ----------------------------------------------------------------------------
    def _portable(self, text: str) -> str:
        for root, token in ((self.run_dir, "{run}"), (self.character, "{character_dir}")):
            if root is not None and (text == str(root) or text.startswith(str(root) + os.sep)):
                return token + text[len(str(root)):]
        return text

    def receipt(self) -> dict:
        """The command without machine-only flags, and a content hash of every input path."""
        args, command, skip = [str(a) for a in self.args], [], 0
        for a in args:
            if skip:
                skip -= 1
            elif a in NOT_SETTINGS:
                skip = NOT_SETTINGS[a]
            else:
                command.append(self._portable(a))
        own = self.products()
        inputs = {}
        for a in args:
            path = Path(a)
            if os.sep not in a and not (self.character and path == self.character):
                continue
            if any(path == p or p in path.parents for p in own):
                continue
            if self.character is not None and path == self.character:
                inputs[self._portable(a)] = character_digest(path)
            elif (digest := _digest(path)) is not None:
                inputs[self._portable(a)] = digest
        return {"step": self.name, "module": self.module, "command": command, "inputs": inputs}

    def stale(self) -> str | None:
        """Why a finished step no longer matches the plan, or None. Steps without a receipt are not judged."""
        record = self._state(".json")
        if record is None or not record.is_file():
            return None
        then, now = json.loads(record.read_text()), self.receipt()
        if then.get("command") != now["command"]:
            changed = sorted(set(then.get("command", [])) ^ set(now["command"]))
            return "command changed: " + " ".join(changed[:8])
        changed = sorted(k for k in set(then.get("inputs", {})) | set(now["inputs"])
                         if then.get("inputs", {}).get(k) != now["inputs"].get(k))
        return ("input changed since it ran: " + ", ".join(changed[:5])) if changed else None

    @contextlib.contextmanager
    def locked(self, log=print):
        """Hold this step's lock; a second run of the same step waits here."""
        path = self._state(".lock")
        if path is None or fcntl is None:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                log(f"[locked]  {self.name}: another run is on this step; waiting for it")
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def set_aside(self, log=print) -> None:
        """Move whatever an unfinished attempt left out of the way. Nothing is deleted."""
        products = self.products()
        left = [p for p in products if p.exists() and not any(q in p.parents for q in products)]
        if not left:
            return
        stem = f"{self.name.replace('/', '--')}-{time.strftime('%Y%m%dT%H%M%S')}"
        dest, n = self.state.parent / ".interrupted" / stem, 0
        while dest.exists():
            n += 1
            dest = dest.with_name(f"{stem}+{n}")
        dest.mkdir(parents=True)
        for i, p in enumerate(left):
            os.replace(p, dest / (p.name if not (dest / p.name).exists() else f"{i}-{p.name}"))
        log(f"[moved]   {self.name}: the last attempt did not finish; what it left is in {dest}")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = REPO / "recipes" / "character-pipeline-v0.json"
MODELS = REPO / "configs" / "models"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PINNED = re.compile(r"[0-9a-f]{40}")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(out.get(key), dict) and isinstance(value, dict) else value
    return out


def _unknown(keys, allowed, where: str) -> None:
    for key in keys:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), [str(a) for a in allowed], n=1)
            hint = f"; did you mean {close[0]!r}?" if close else f"; allowed: {', '.join(sorted(map(str, allowed)))}"
            raise SystemExit(f"{where}: unknown key {key!r}{hint}")


def validate_pipeline(block, where: str) -> None:
    """A "pipeline" block (recipe or character config) may only set keys the planner reads."""
    if not isinstance(block, dict):
        raise SystemExit(f"{where}: \"pipeline\" must be an object")
    _unknown(block, PIPELINE_KEYS, f"{where} pipeline")
    for key, value in block.items():
        allowed = PIPELINE_KEYS[key]
        if allowed is not None:
            if not isinstance(value, dict):
                raise SystemExit(f"{where} pipeline.{key}: must be an object with {', '.join(sorted(allowed))}")
            _unknown(value, allowed, f"{where} pipeline.{key}")
    seeds = block.get("seeds")
    if seeds is not None and not (isinstance(seeds, list) and seeds and len(set(seeds)) == len(seeds)
                                  and all(isinstance(x, int) and not isinstance(x, bool) for x in seeds)):
        raise SystemExit(f"{where} pipeline.seeds: must be a non-empty list of distinct integers")


def validate_recipe(data, where: str = "recipe") -> None:
    """Fail with one clear line on a mistyped recipe instead of silently ignoring the key."""
    if not isinstance(data, dict):
        raise SystemExit(f"{where}: a recipe is a JSON object")
    _unknown(data, RECIPE_KEYS, where)
    if not isinstance(data.get("name"), str) or not data["name"]:
        raise SystemExit(f"{where}: \"name\" is required")
    validate_pipeline(data.get("pipeline", {}), where)
    arms = data.get("arms")
    if arms is not None:
        if not isinstance(arms, list):
            raise SystemExit(f"{where}: \"arms\" must be a list")
        _unknown(arms, ARMS, f"{where} arms")
    seen = set()
    for i, stage in enumerate(data.get("stages") or []):
        at = f"{where} stages[{i}]"
        if not isinstance(stage, dict) or not isinstance(stage.get("id"), str):
            raise SystemExit(f"{at}: every stage needs a string \"id\"")
        at = f"{where} stage {stage['id']!r}"
        if stage["id"] in seen or "/" in stage["id"]:
            raise SystemExit(f"{at}: stage ids must be unique and contain no '/'")
        seen.add(stage["id"])
        _unknown(stage, STAGE_KEYS, at)
        if "module" not in stage:
            if "args" in stage:
                raise SystemExit(f"{at}: has \"args\" but no \"module\"")
            continue  # a documented stage that does not run here
        if not str(stage["module"]).startswith("character_pipeline."):
            raise SystemExit(f"{at}: module must be inside character_pipeline, got {stage['module']!r}")
        args = stage.get("args")
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise SystemExit(f"{at}: \"args\" must be a list of strings (quote the numbers)")
        _check_templates(args, at)
    for i, arm in enumerate(data.get("eval_arms") or []):
        at = f"{where} eval_arms[{i}]"
        if not isinstance(arm, dict) or not {"arm", "adapter"} <= set(arm):
            raise SystemExit(f"{at}: needs \"arm\" and \"adapter\"")
        _unknown(arm, {"arm", "adapter", "note"}, at)
        _check_templates([arm["adapter"]], at)
    if data.get("eval_arms") and not data.get("stages"):
        raise SystemExit(f"{where}: \"eval_arms\" is only read together with \"stages\"")


def _check_templates(args: list[str], where: str) -> None:
    for arg in args:
        for found in re.findall(r"\{[^{}]*\}", arg):
            if found not in TEMPLATES:
                raise SystemExit(f"{where}: unknown template {found} in {arg!r}; known: {' '.join(TEMPLATES)}")


def load_recipe(recipe: Path | None) -> dict:
    """The recipe file, validated. No --recipe means recipes/character-pipeline-v0.json."""
    path = Path(recipe) if recipe else DEFAULT_RECIPE
    if not path.is_file():
        raise SystemExit(f"recipe {path} not found; `character-pipeline recipes` lists the ones that ship")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"recipe {path} is not valid JSON: {e}") from None
    validate_recipe(data, str(path))
    return data


def load_config(character_dir: Path) -> dict:
    path = Path(character_dir) / "config.json"
    if not path.is_file():
        raise SystemExit(f"{character_dir} is not a character folder (no config.json); "
                         "`character-pipeline new <id>` makes one")
    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"{path} is not valid JSON: {e}") from None
    if not isinstance(config.get("id"), str) or not SAFE_ID.fullmatch(config["id"]):
        raise SystemExit(f"{path}: \"id\" must be letters, digits, '.', '_' or '-' (it names the folder under runs/)")
    validate_pipeline(config.get("pipeline", {}), str(path))
    student = config.get("student", {})
    if not isinstance(student, dict):
        raise SystemExit(f"{path}: \"student\" must be an object")
    return config


def model_preset(name: str) -> dict:
    path = MODELS / f"{name}.json"
    if not SAFE_ID.fullmatch(name) or not path.is_file():
        known = ", ".join(sorted(f.stem for f in MODELS.glob("*.json")))
        raise SystemExit(f"unknown model preset {name!r}; known presets: {known}")
    preset = json.loads(path.read_text())
    for part, where in ((preset, name), (preset.get("instruct") or {}, f"{name} instruct")):
        revision = part.get("revision")
        if revision is not None and not PINNED.fullmatch(str(revision)):
            raise SystemExit(f"{path}: {where} revision must be a full 40-character commit hash or null, "
                             f"got {revision!r}")
    return preset


def settings(character_dir: Path, recipe: Path | None = None, model: str | None = None,
             allow_unpinned: bool = False) -> dict:
    """DEFAULTS < recipe "pipeline" < character config "pipeline" < character "student" < --model preset.

    The preset is last on purpose: `--model` changes the scale, so its model, revision, LoRA shape and
    sft.seq_len win over what the character folder says. A student or preset without a full commit
    hash is refused unless allow_unpinned.
    """
    config = load_config(character_dir)
    recipe_data = load_recipe(recipe)
    merged = _merge(DEFAULTS, recipe_data.get("pipeline", {}))
    merged = _merge(merged, config.get("pipeline", {}))
    student = config.get("student", {})
    merged["model"] = student.get("model", STUDENT)
    merged["revision"] = student.get("revision", STUDENT_REVISION)
    merged.setdefault("lora", {"rank": 16, "alpha": 32})
    source = f"{Path(character_dir) / 'config.json'} student"
    note = ""
    if model:
        preset = model_preset(model)
        merged.update(model=preset["hf"], revision=preset["revision"], model_preset=model)
        merged["instruct"] = preset.get("instruct")
        merged["lora"] = preset.get("lora", merged["lora"])
        if "sft" in preset:
            merged["sft"] = _merge(merged["sft"], preset["sft"])
        source, note = str(MODELS / (model + ".json")), f" ({preset['note']})" if preset.get("note") else ""
    if not PINNED.fullmatch(str(merged["revision"] or "")):
        if not allow_unpinned:
            raise SystemExit(f"{merged['model']} has no pinned revision{note}: got {merged['revision']!r}. Pin a "
                             f"40-character commit hash in {source}, or pass --allow-unpinned to train on "
                             f"{merged['revision'] or 'main'!r}.")
        merged["revision"] = merged["revision"] or "main"
        merged["unpinned"] = True
    merged["arms"] = recipe_data.get("arms", list(ARMS))
    merged["recipe"] = recipe_data["name"]
    merged["id"] = config["id"]
    return merged


TAKES_MODEL = {"train.sft", "train.sample", "train.preference", "train.merge", "generate.introspection", "evaluate"}


def plan_staged(character_dir: Path, recipe_data: dict, s: dict, runs_root: Path, device: str,
                seeds: list[int], extra: list[str]) -> list[Step]:
    """Steps from a recipe that lists its own stages as templated commands (e.g. recipes/oct-2025.json).

    Templates: {runs} {character} {seed} {teacher_url} {judge_url}; "characters/{character}" means this
    character's folder. Stages that mention {seed} run once per seed, after the shared stages. A stage
    whose --out is a directory is marked done by "<out>.done", written only after it succeeds.
    """
    R = Path(runs_root) / s["id"]

    def fill(arg, seed=None):
        text = str(arg).replace("characters/{character}", str(character_dir))
        text = text.replace("{runs}", str(runs_root)).replace("{character}", s["id"])
        text = text.replace("{teacher_url}", s["teacher_url"]).replace("{judge_url}", s["judge_url"])
        return text.replace("{seed}", str(seed)) if seed is not None else text

    model = ["--model", s["model"], "--revision", s["revision"], "--device", device, *extra]
    stages = [st for st in recipe_data["stages"] if st.get("module")]
    if not stages:
        raise SystemExit(f"recipe {recipe_data['name']}: no stage has a \"module\", so there is nothing to run")
    shared = [st for st in stages if "{seed}" not in json.dumps(st["args"])]
    seeded = [st for st in stages if st not in shared]
    out = []

    def step(stage, seed=None):
        args = [fill(a, seed) for a in stage["args"]]
        target = Path(args[args.index("--out") + 1]) if "--out" in args else None
        short = stage["module"].removeprefix("character_pipeline.")
        name = stage["id"] + (f"/seed{seed}" if seed is not None else "")
        marker = Path(str(target) + ".done") if target else R / f".{name.replace('/', '-')}.done"
        if short in TAKES_MODEL:
            args += model
        return Step(name, stage["module"], args, [marker], [], marker=marker)

    chain = [step(st) for st in shared]
    for prev, cur in itertools.pairwise(chain):
        cur.needs = prev.outputs  # each stage needs the one before it
    out += chain
    for seed in seeds:
        per_seed = [step(st, seed) for st in seeded]
        for prev, cur in zip(chain[-1:] + per_seed, per_seed):
            cur.needs = prev.outputs
        out += per_seed
    ev = _eval_args(character_dir, s, model)
    out.append(Step("eval/prompted_base", "character_pipeline.evaluate",
                    ["--arm", "prompted_base", "--out", R / "eval" / "prompted_base", *ev],
                    [R / "eval" / "prompted_base" / "summary.json"]))
    for seed in seeds:
        for arm in recipe_data.get("eval_arms", []):
            adapter = Path(fill(arm["adapter"], seed))
            dest = R / "eval" / f'{arm["arm"]}-seed{seed}'
            out.append(Step(f'eval/{arm["arm"]}/seed{seed}', "character_pipeline.evaluate",
                            ["--arm", arm["arm"], "--adapter", adapter, "--out", dest, *ev],
                            [dest / "summary.json"], [adapter / "adapter_config.json"]))
    return out


def _eval_args(character_dir: Path, s: dict, model: list) -> list:
    ev = s["eval"]
    return ["--character", character_dir, "--judge-url", s["judge_url"], "--rollouts", ev["rollouts"],
            *(["--limit", ev["limit"]] if ev.get("limit") else []),
            *(["--temperature", ev["temperature"]] if ev.get("temperature") is not None else []), *model]


def review_gaps(review: Path) -> str | None:
    """Why review.jsonl is not a finished review yet, or None. A copied, unfilled sheet must not pass."""
    if not review.is_file():
        return None  # reported as a missing file instead
    blank = bad = total = 0
    for n, line in enumerate(review.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            decision = str(json.loads(line).get("decision") or "").strip().lower()
        except (json.JSONDecodeError, AttributeError):
            return f"{review} line {n} is not a JSON object"
        total += 1
        blank += decision == ""
        bad += decision not in ("", "accept", "reject")
    if total == 0:
        return f"{review} is empty"
    if blank or bad:
        return (f"{review}: {blank} of {total} turns have no decision" + (f", {bad} have a decision that is "
                "neither accept nor reject" if bad else "") + "; set every one to accept or reject")
    return None


def _attach(steps: list[Step], run_dir: Path, character_dir: Path) -> list[Step]:
    names = [st.name for st in steps]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate step names in the plan: {sorted({n for n in names if names.count(n) > 1})}")
    for st in steps:
        st.state, st.run_dir, st.character = run_dir / ".steps", run_dir, character_dir
    return steps


def plan(character_dir: Path, runs_root: Path = Path("runs"), device: str = "cuda:0",
         seeds: list[int] | None = None, extra: list[str] | None = None, recipe: Path | None = None,
         model: str | None = None, allow_unpinned: bool = False) -> list[Step]:
    character_dir = Path(character_dir)
    s = settings(character_dir, recipe, model, allow_unpinned)
    recipe_data = load_recipe(recipe)
    R = Path(runs_root) / s["id"]
    seeds = s["seeds"] if seeds is None else list(seeds)
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"seeds must be distinct, got {seeds}")
    if recipe_data.get("stages"):
        return _attach(plan_staged(character_dir, recipe_data, s, Path(runs_root), device, seeds, extra or []),
                       R, character_dir)
    model = ["--model", s["model"], "--revision", s["revision"], "--device", device, *(extra or [])]
    lora = ["--rank", s["lora"]["rank"], "--alpha", s["lora"]["alpha"]]
    m = "character_pipeline."
    steps = [
        Step("generate", m + "generate.teacher",
             [character_dir, character_dir / s["scenarios"], "--url", s["teacher_url"], "--out", R / "teacher.jsonl"],
             [R / "teacher.jsonl"], [character_dir / s["scenarios"]],
             human=f"write caller scenarios to {character_dir / s['scenarios']} (see characters/_template)"),
        Step("check", m + "check.run",
             [R / "teacher.jsonl", character_dir / s["scenarios"], "--url", s["teacher_url"], "--out", R / "checked.jsonl"],
             [R / "checked.jsonl"], [R / "teacher.jsonl"]),
        Step("judge", m + "judge.score",
             [character_dir, R / "checked.jsonl", "--url", s["judge_url"], "--out", R / "judged.jsonl"],
             [R / "judged.jsonl"], [R / "checked.jsonl"]),
        Step("clean", m + "data.clean", [R / "judged.jsonl", R / "review.jsonl", "--out", R / "clean"],
             [R / "clean" / "train.jsonl"], [R / "judged.jsonl", R / "review.jsonl"],
             human=f"review every character turn: `character-pipeline review-sheet {character_dir}` writes "
                   f"{R / 'review.todo.jsonl'}; fill in decision accept/reject and save it as {R / 'review.jsonl'}"),
    ]
    clean = R / "clean"
    for seed in seeds:
        sft = R / "sft" / f"seed{seed}"
        pairs = R / "pairs" / f"seed{seed}"
        kto_labels = R / "kto_review.jsonl"
        steps += [
            Step(f"sft/seed{seed}", m + "train.sft",
                 ["--train", clean / "train.jsonl", "--validation", clean / "validation.jsonl", "--out", sft,
                  "--seed", seed, "--epochs", s["sft"]["epochs"], "--lr", s["sft"]["lr"],
                  "--seq-len", s["sft"]["seq_len"], *lora, *model],
                 [sft / "adapter" / "adapter_config.json"], [clean / "train.jsonl"]),
            Step(f"sample/seed{seed}", m + "train.sample",
                 ["--adapter", sft / "adapter", "--chats", clean / "train.jsonl",
                  "--out", R / "samples" / f"seed{seed}.jsonl", "--seed", seed, *model],
                 [R / "samples" / f"seed{seed}.jsonl"], [sft / "adapter" / "adapter_config.json"]),
            Step(f"pairs/seed{seed}", m + "data.pairs",
                 [clean / "train.jsonl", R / "samples" / f"seed{seed}.jsonl", "--out", pairs]
                 + (["--kto-labels", kto_labels] if kto_labels.exists() else []),
                 [pairs / "manifest.json"], [R / "samples" / f"seed{seed}.jsonl"]),
            Step(f"dpo/seed{seed}", m + "train.preference",
                 ["dpo", "--sft-adapter", sft / "adapter", "--data", pairs / "dpo_train.jsonl",
                  "--out", R / "dpo" / f"seed{seed}", "--seed", seed, "--lr", s["preference"]["lr"],
                  "--beta", s["preference"]["beta"], "--max-length", s["preference"]["max_length"], *lora, *model],
                 [R / "dpo" / f"seed{seed}" / "adapter" / "adapter_config.json"], [pairs / "dpo_train.jsonl"]),
            Step(f"kto/seed{seed}", m + "train.preference",
                 ["kto", "--sft-adapter", sft / "adapter", "--data", pairs / "kto_train.jsonl",
                  "--out", R / "kto" / f"seed{seed}", "--seed", seed, "--lr", s["preference"]["lr"],
                  "--beta", s["preference"]["beta"], "--max-length", s["preference"]["max_length"],
                  "--batch-size", 2, *lora, *model],
                 [R / "kto" / f"seed{seed}" / "adapter" / "adapter_config.json"], [pairs / "kto_train.jsonl"],
                 human=f"optional: absolute good/bad labels in {kto_labels}, then delete {pairs} and re-run",
                 optional=True),
            Step(f"continued_sft/seed{seed}", m + "train.sft",
                 ["--train", pairs / "continued_sft.jsonl", "--validation", clean / "validation.jsonl",
                  "--out", R / "continued_sft" / f"seed{seed}", "--init-adapter", sft / "adapter",
                  "--seed", seed, "--epochs", s["continued_sft"]["epochs"], "--lr", s["continued_sft"]["lr"],
                  "--seq-len", s["sft"]["seq_len"], *model],
                 [R / "continued_sft" / f"seed{seed}" / "adapter" / "adapter_config.json"],
                 [pairs / "continued_sft.jsonl"]),
        ]
    keep = set(s["arms"])
    def wanted(step):
        stage = step.name.split("/")[0]
        if stage in ARMS:
            return stage in keep
        if stage in ("sample", "pairs"):  # only the branched arms need student samples and pairs
            return bool(keep & {"dpo", "kto", "continued_sft"})
        return True
    steps = [st for st in steps if wanted(st)]
    for st in steps:
        if st.name == "clean":  # the first human stop: an unfilled review sheet is not a review
            st.blocked = review_gaps(R / "review.jsonl")
    ev = _eval_args(character_dir, s, model)
    steps.append(Step("eval/prompted_base", m + "evaluate", ["--arm", "prompted_base", "--out", R / "eval" / "prompted_base", *ev],
                      [R / "eval" / "prompted_base" / "summary.json"]))
    inst = s.get("instruct") or {}
    steps.append(Step("eval/prompted_instruct", m + "evaluate",
                      ["--arm", "prompted_instruct", "--out", R / "eval" / "prompted_instruct", *ev,
                       "--model", inst.get("hf", ""), "--revision", inst.get("revision") or ""],
                      [R / "eval" / "prompted_instruct" / "summary.json"],
                      human="pin the instruct sibling's revision in the model preset (configs/models) to run "
                            "the prompted-instruct control, Zero's headline comparison", optional=True,
                      blocked=None if inst.get("revision") else "instruct sibling not pinned"))
    for seed in seeds:
        for arm in [a for a in ARMS if a in keep]:
            adapter = R / arm / f"seed{seed}" / "adapter"
            out = R / "eval" / f"{arm}-seed{seed}"
            steps.append(Step(f"eval/{arm}/seed{seed}", m + "evaluate",
                              ["--arm", arm, "--adapter", adapter, "--out", out, *ev],
                              [out / "summary.json"], [adapter / "adapter_config.json"], optional=arm == "kto"))
    return _attach(steps, R, character_dir)


def _matches(name: str, prefix: str) -> bool:
    """`sft` selects sft/seed0 and sft/seed1 but not sample/...; `eval/dpo` selects that arm's evals."""
    return name == prefix or name.startswith(prefix.rstrip("/") + "/")


def run_step(step: Step, log=print) -> bool:
    """One attempt at one step, in its own process. True when it finished and wrote its outputs."""
    with step.locked(log):
        if step.done():  # another run finished it while we waited for the lock
            log(f"[done]    {step.name}")
            return True
        if step.interrupted():
            step.set_aside(log)
        log(f"[run]     {step.name}\n          {step.command()}")
        running, record = step._state(".running"), step._state(".json")
        receipt = step.receipt()  # inputs are hashed before the step can touch anything
        if running:
            _write_atomic(running, json.dumps({"pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
        started = time.time()
        code = subprocess.run(step.argv(), check=False).returncode
        if code != 0:
            log(f"[failed]  {step.name}: exit status {code}")
            return False
        missing = [str(p) for p in step.outputs if p != step.marker and not p.exists()]
        if missing:
            log(f"[failed]  {step.name}: exited 0 but did not write {', '.join(missing)}")
            return False
        if step.marker:
            _write_atomic(step.marker, "done\n")
        if record:
            _write_atomic(record, json.dumps(receipt | {"seconds": round(time.time() - started, 1),
                                                        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                                             indent=2) + "\n")
            running.unlink(missing_ok=True)
    return True


def execute(steps: list[Step], only: str | None = None, until: str | None = None, dry_run: bool = False,
            log=print, reuse_stale: bool = False) -> str:
    """Run pending steps in order.

    Returns 'done', 'dry-run', 'stopped:<step>' (a person or an earlier step is needed),
    'stale:<step>' (finished with other settings or inputs) or 'failed:<step>'.
    """
    for prefix in (only, until):
        if prefix and not any(_matches(st.name, prefix) for st in steps):
            raise SystemExit(f"no step is named {prefix!r} or starts with '{prefix}/'; steps: "
                             + ", ".join(dict.fromkeys(st.name.split("/")[0] for st in steps)))
    for step in steps:
        if only and not _matches(step.name, only):
            continue
        if step.done():
            reason = step.stale()
            if reason and not reuse_stale:
                log(f"[stale]   {step.name}: {reason}")
                log(f"          → to redo it, move away {', '.join(str(p) for p in step.products()[:1])} and what was "
                    "built from it; to keep it as it is, pass --reuse-stale")
                if not dry_run:
                    return f"stale:{step.name}"
            else:
                log(f"[done]    {step.name}" + (f" (kept although stale: {reason})" if reason else ""))
        elif not step.ready():
            if step.optional:
                log(f"[skip]    {step.name}: {step.human or 'input missing'}")
            else:
                missing = step.blocked or "missing " + ", ".join(str(p) for p in step.needs if not p.exists())
                log(f"[waiting] {step.name}: {missing}")
                if step.human:
                    log(f"          → {step.human}")
                if not dry_run:
                    return f"stopped:{step.name}"
        elif dry_run:
            log(f"[todo]    {step.name}" + (" (the last attempt did not finish; it will be redone)"
                                            if step.interrupted() else "") + f"\n          {step.command()}")
        elif not run_step(step, log):
            return f"failed:{step.name}"
        if until and _matches(step.name, until):
            break
    return "dry-run" if dry_run else "done"


def review_sheet(judged: Path, out: Path) -> int:
    """One line per character turn with the text, flags and p_break; `decision` left blank."""
    rows = []
    for line in judged.read_text().splitlines():
        chat = json.loads(line)
        key, caller, user = f'{chat["scenario_id"]}:{chat["seed"]}', -1, ""
        for turn in chat["history"]:
            if turn["speaker"] == "partner":
                caller += 1
                user = turn["text"]
                continue
            rows.append({"conversation": key, "turn": caller, "decision": "", "user": user, "reply": turn["text"],
                         "flags": turn.get("flags", []), "p_break": (turn.get("d1") or {}).get("p_break")})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def resolved(character_dir: Path, runs_root: Path, recipe: Path | None = None, model: str | None = None,
             allow_unpinned: bool = False, seeds: list[int] | None = None) -> Path:
    """Write the merged settings of this invocation next to the run.

    The file is rewritten by every `run`, so it describes the latest one. What each finished step
    actually ran with is in runs/<id>/.steps/<step>.json, which is never rewritten for a finished step.
    """
    s = settings(character_dir, recipe, model, allow_unpinned)
    if seeds is not None:
        s["seeds"] = list(seeds)
    out = Path(runs_root) / s["id"] / "resolved_recipe.json"
    _write_atomic(out, json.dumps(s, indent=2) + "\n")
    return out
