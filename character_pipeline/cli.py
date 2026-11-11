"""character-pipeline: one command for every stage, for any number of characters.

    character-pipeline new   <id> --name "NAME"     scaffold characters/<id> from the template
    character-pipeline status [<character> ...]     what is done, what is next, eval rates
    character-pipeline run   <character> [...]      run every pending step, resumably
                             --model qwen3.5-4b-base   --recipe recipes/<name>.json
    character-pipeline models | recipes             list model-scale presets and recipes
    character-pipeline review-sheet <character>     write the human review template
    character-pipeline ecology [--cross-judge]      all characters side by side
    character-pipeline <stage> [args]               run one stage directly:
        generate check judge clean pairs sft sample dpo kto serve eval
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

from character_pipeline import run as pipeline

STAGES = {
    "generate": "character_pipeline.generate.teacher",
    "check": "character_pipeline.check.run",
    "judge": "character_pipeline.judge.score",
    "clean": "character_pipeline.data.clean",
    "pairs": "character_pipeline.data.pairs",
    "sft": "character_pipeline.train.sft",
    "sample": "character_pipeline.train.sample",
    "serve": "character_pipeline.serve",
    "eval": "character_pipeline.evaluate",
}


def new(args) -> Path:
    root = Path(args.characters)
    # The id names characters/<id> and runs/<id>: one plain path component, never "..", "a/b" or "_template".
    if not pipeline.SAFE_ID.fullmatch(args.id):
        raise SystemExit(f"character id {args.id!r} must start with a letter or digit and use only letters, "
                         "digits, '.', '_' and '-'")
    dest = root / args.id
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    if not (root / "_template" / "config.json").is_file():
        raise SystemExit(f"{root / '_template'} not found; run from the repository root or pass --characters")
    shutil.copytree(root / "_template", dest)
    config = json.loads((dest / "config.json").read_text())
    config.update(id=args.id, display_name=args.name or args.id.upper(), status="draft")
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"created {dest}. Next: write BIBLE.md and JUDGE_RULES.md, adapt scenarios.jsonl "
          f"(see {dest / 'README.md'}), then `character-pipeline run {dest}`.")
    return dest


def status(args) -> None:
    from character_pipeline.ecology import registry
    rows = registry(Path(args.characters), Path(args.runs))
    if args.character:
        wanted = {Path(c).name for c in args.character}
        rows = [r for r in rows if r["id"] in wanted or Path(r["folder"]).name in wanted]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for r in rows:
        if "error" in r:
            print(f'{r["id"]:<24}{r["status"] or "":<20}cannot plan: {r["error"]}')
            continue
        print(f'{r["id"]:<24}{r["status"] or "":<20}{r["done"]:>3}/{r["steps"]:<4} next: {r["next"] or "-"}')
        for arm, e in r["eval"].items():
            # hold/drift/cave are over eligible insistence rollouts; updated is over completed evidence rollouts
            rates = "  ".join(f"{k} {e[k]:.2f}" for k in ("hold", "drift", "cave") if isinstance(e.get(k), float))
            updated = f'updated {e["updated"]:.2f}' if isinstance(e.get("updated"), float) else "updated n/a"
            print(f'    {arm:<22}{rates}  (of {e["eligible"]} eligible)  {updated} '
                  f'(of {e.get("evidence_completed", "?")})')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in STAGES:
        return importlib.import_module(STAGES[argv[0]]).main(argv[1:])
    if argv and argv[0] in ("dpo", "kto"):
        return importlib.import_module("character_pipeline.train.preference").main(argv)

    ap = argparse.ArgumentParser(prog="character-pipeline", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--characters", default="characters")
    ap.add_argument("--runs", default="runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="scaffold characters/<id> from characters/_template")
    p.add_argument("id", help="folder name: letters, digits, '.', '_' or '-'")
    p.add_argument("--name", help="display name (default: the id in capitals)")
    p = sub.add_parser("status", help="steps done, next step and eval rates for every character")
    p.add_argument("character", nargs="*")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("models", help="list model-scale presets")
    p = sub.add_parser("recipes", help="list recipe files")
    p = sub.add_parser("run", help="run every pending step of one character; safe to re-run")
    p.add_argument("character", type=Path)
    p.add_argument("--recipe", type=Path, help="recipes/<name>.json (default: character-pipeline-v0)")
    p.add_argument("--model", help="model preset from configs/models, e.g. qwen3.5-4b-base")
    p.add_argument("--allow-unpinned", action="store_true", help="train a preset with no pinned revision on 'main'")
    p.add_argument("--only", help="run only this step or group, e.g. sft, eval or eval/dpo/seed0")
    p.add_argument("--until", help="stop after the first step of this name or group")
    p.add_argument("--reuse-stale", action="store_true",
                   help="keep finished steps whose settings or inputs have changed since they ran")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p = sub.add_parser("review-sheet", help="write runs/<id>/review.todo.jsonl for the human review")
    p.add_argument("character", type=Path)
    p.add_argument("--force", action="store_true", help="overwrite an existing review.todo.jsonl")
    p = sub.add_parser("ecology", help="all characters side by side; --cross-judge for distinctness")
    p.add_argument("--cross-judge", action="store_true")
    p.add_argument("--per-cell", type=int, default=20, help="replies per character judged under each rule set")
    p.add_argument("--arm", default="dpo-seed0", help="which evaluated arm's replies to compare")
    p.add_argument("--judge-url", default="http://127.0.0.1:8090")
    args = ap.parse_args(argv)

    if args.cmd == "new":
        new(args)
    elif args.cmd == "status":
        status(args)
    elif args.cmd == "models":
        for f in sorted(pipeline.MODELS.glob("*.json")):
            m = json.loads(f.read_text())
            print(f'{m["id"]:<22}{m["hf"]:<26}{m["status"]:<10}{(m["revision"] or "UNPINNED")[:12]:<14}{m["hardware"]}')
    elif args.cmd == "recipes":
        for f in sorted((pipeline.REPO / "recipes").glob("*.json")):
            r = json.loads(f.read_text())
            cite = r.get("cite", "")
            if isinstance(cite, dict):  # structured credit: show the paper, the file has the rest
                paper = cite.get("paper", {})
                cite = f'{", ".join(paper.get("authors", []))} ({paper.get("year", "")}). {paper.get("title", "")}. ' \
                       f'arXiv:{paper.get("arxiv", "")}'
            print(f'{r["name"]:<26}{r.get("status", "")}\n    {r.get("summary", "")}\n    cite: {cite}')
    elif args.cmd == "run":
        steps = pipeline.plan(args.character, Path(args.runs), args.device, args.seeds,
                              ["--local-files-only"] if args.local_files_only else None,
                              args.recipe, args.model, args.allow_unpinned)
        if not args.dry_run:
            print("settings:", pipeline.resolved(args.character, Path(args.runs), args.recipe, args.model,
                                                 args.allow_unpinned, args.seeds))
        result = pipeline.execute(steps, args.only, args.until, args.dry_run, reuse_stale=args.reuse_stale)
        print(result)
        # 0 done or dry-run; 1 a step failed; 2 waiting for a person or an input; 3 stale outputs
        code = {"failed": 1, "stopped": 2, "stale": 3}.get(result.split(":")[0], 0)
        if code:
            raise SystemExit(code)
    elif args.cmd == "review-sheet":
        cid = pipeline.settings(args.character)["id"]
        R = Path(args.runs) / cid
        if not (R / "judged.jsonl").is_file():
            raise SystemExit(f"{R / 'judged.jsonl'} does not exist yet; `character-pipeline run {args.character}` "
                             "writes it (generate, check, judge) and then stops for this review")
        if (R / "review.todo.jsonl").exists() and not args.force:
            raise SystemExit(f"{R / 'review.todo.jsonl'} exists and may hold decisions; pass --force to overwrite it")
        n = pipeline.review_sheet(R / "judged.jsonl", R / "review.todo.jsonl")
        print(f"{n} character turns in {R / 'review.todo.jsonl'}. Set each decision to accept or reject, "
              f"then save as {R / 'review.jsonl'}.")
    elif args.cmd == "ecology":
        from character_pipeline.ecology import cross_judge, registry, replies_from_eval
        rows = registry(Path(args.characters), Path(args.runs))
        if not args.cross_judge:
            return status(argparse.Namespace(characters=args.characters, runs=args.runs, character=[], json=False))
        from character_pipeline.judge.d1 import D1Client
        client = D1Client(url=args.judge_url)
        chars, replies = {}, {}
        for r in rows:
            folder = Path(r["folder"])
            config = json.loads((folder / "config.json").read_text())
            summary = Path(args.runs) / r["id"] / "eval" / args.arm / "summary.json"
            if summary.exists():
                rules = (folder / config["judge"]["rules_file"]).read_text().strip()
                chars[r["id"]] = {"name": config.get("display_name", r["id"]), "rules": rules}
                replies[r["id"]] = replies_from_eval(summary)
        if len(chars) < 2:
            raise SystemExit(f"cross-judge compares characters with each other: it needs at least two with an "
                             f"evaluated '{args.arm}' arm under {args.runs}, found {len(chars)} ({', '.join(chars) or 'none'})")
        print(json.dumps(cross_judge(chars, replies, client.p_break, args.per_cell), indent=2))


if __name__ == "__main__":
    main()
