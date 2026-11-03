"""The whole pipeline for one character, as an ordered list of resumable steps.

Every step names its inputs and outputs under runs/<character-id>/. A step whose
outputs exist is skipped, so `character-pipeline run <character>` can be
re-run after any interruption. Two steps need people:

  review.jsonl      accept/reject for each character turn (make a sheet with
                    `character-pipeline review-sheet`)
  kto_review.jsonl  optional absolute good/bad labels; without it the KTO arm
                    is skipped, never faked from DPO pairs

The run stops at a missing human input and says what to write. Nothing here
starts the teacher or judge servers; scripts/serve_teacher.sh and
scripts/serve_d1.sh do, once per machine.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from character_pipeline.train.common import STUDENT, STUDENT_REVISION

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


@dataclass
class Step:
    name: str
    module: str
    args: list[str]
    outputs: list[Path]
    needs: list[Path] = field(default_factory=list)
    human: str | None = None      # what a person must produce if `needs` is missing
    optional: bool = False        # skip (not stop) when `needs` is missing

    def done(self) -> bool:
        return all(p.exists() for p in self.outputs)

    def ready(self) -> bool:
        return all(p.exists() for p in self.needs)

    def command(self) -> str:
        return " ".join(["python -m", self.module, *map(str, self.args)])


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = REPO / "recipes" / "character-pipeline-v0.json"
MODELS = REPO / "configs" / "models"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(out.get(key), dict) and isinstance(value, dict) else value
    return out


def model_preset(name: str) -> dict:
    path = MODELS / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"unknown model preset {name!r}; see {MODELS}")
    return json.loads(path.read_text())


def settings(character_dir: Path, recipe: Path | None = None, model: str | None = None,
             allow_unpinned: bool = False) -> dict:
    """DEFAULTS < recipe < character config "pipeline" < character "student" < --model preset."""
    config = json.loads((Path(character_dir) / "config.json").read_text())
    recipe_path = Path(recipe) if recipe else DEFAULT_RECIPE
    recipe_data = json.loads(recipe_path.read_text()) if recipe_path.is_file() else {}
    merged = _merge(DEFAULTS, recipe_data.get("pipeline", {}))
    merged = _merge(merged, config.get("pipeline", {}))
    student = config.get("student", {})
    merged["model"] = student.get("model", STUDENT)
    merged["revision"] = student.get("revision", STUDENT_REVISION)
    merged.setdefault("lora", {"rank": 16, "alpha": 32})
    if model:
        preset = model_preset(model)
        merged.update(model=preset["hf"], revision=preset["revision"], model_preset=model)
        merged["lora"] = preset.get("lora", merged["lora"])
        if "sft" in preset:
            merged["sft"] = _merge(merged["sft"], preset["sft"])
        if preset["revision"] is None:
            if not allow_unpinned:
                raise SystemExit(f"{model} has no pinned revision ({preset.get('note', '')}). Pin one in "
                                 f"{MODELS / (model + '.json')} or pass --allow-unpinned to train on 'main'.")
            merged["revision"] = "main"
            merged["unpinned"] = True
    merged["arms"] = recipe_data.get("arms", list(ARMS))
    merged["recipe"] = recipe_data.get("name", "defaults")
    merged["id"] = config["id"]
    return merged


def plan(character_dir: Path, runs_root: Path = Path("runs"), device: str = "cuda:0",
         seeds: list[int] | None = None, extra: list[str] | None = None, recipe: Path | None = None,
         model: str | None = None, allow_unpinned: bool = False) -> list[Step]:
    character_dir = Path(character_dir)
    s = settings(character_dir, recipe, model, allow_unpinned)
    R = Path(runs_root) / s["id"]
    seeds = s["seeds"] if seeds is None else seeds
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
    ev = ["--character", character_dir, "--judge-url", s["judge_url"], "--rollouts", s["eval"]["rollouts"],
          *(["--limit", s["eval"]["limit"]] if s["eval"].get("limit") else []), *model]
    steps.append(Step("eval/prompted_base", m + "evaluate", ["--arm", "prompted_base", "--out", R / "eval" / "prompted_base", *ev],
                      [R / "eval" / "prompted_base" / "summary.json"]))
    for seed in seeds:
        for arm in [a for a in ARMS if a in keep]:
            adapter = R / arm / f"seed{seed}" / "adapter"
            out = R / "eval" / f"{arm}-seed{seed}"
            steps.append(Step(f"eval/{arm}/seed{seed}", m + "evaluate",
                              ["--arm", arm, "--adapter", adapter, "--out", out, *ev],
                              [out / "summary.json"], [adapter / "adapter_config.json"], optional=arm == "kto"))
    return steps


def execute(steps: list[Step], only: str | None = None, until: str | None = None, dry_run: bool = False,
            log=print) -> str:
    """Run pending steps in order. Returns 'done', 'stopped:<step>' or 'dry-run'."""
    for step in steps:
        if only and not step.name.startswith(only):
            continue
        if step.done():
            log(f"[done]    {step.name}")
        elif not step.ready():
            if step.optional:
                log(f"[skip]    {step.name}: {step.human or 'input missing'}")
                continue
            missing = ", ".join(str(p) for p in step.needs if not p.exists())
            log(f"[waiting] {step.name}: missing {missing}")
            if step.human:
                log(f"          → {step.human}")
            if not dry_run:
                return f"stopped:{step.name}"
        elif dry_run:
            log(f"[todo]    {step.name}\n          {step.command()}")
        else:
            log(f"[run]     {step.name}\n          {step.command()}")
            importlib.import_module(step.module).main([str(a) for a in step.args])
        if until and step.name.startswith(until):
            return "done"
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
             allow_unpinned: bool = False) -> Path:
    """Write the fully merged settings next to the run, so every run records exactly what it used."""
    s = settings(character_dir, recipe, model, allow_unpinned)
    out = Path(runs_root) / s["id"] / "resolved_recipe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=2) + "\n")
    return out
