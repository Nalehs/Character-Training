"""Load a character folder: config, bible and optional private core/style files.

A character folder holds `config.json`, a bible and a source manifest. The
system prompt for teacher generation and for prompted baselines is the bible,
optionally followed by `core.md` and `style.md` when those exist locally. Those
two files are gitignored for Golem: the full core and style stay private and
only the short public excerpt in BIBLE.md is published.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED = ("id", "status", "bible", "sources", "student", "judge", "arms", "evaluation")


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Character paths must remain inside the character folder")
    if not path.is_file():
        raise ValueError(f"Missing character file: {relative}")
    return path


@dataclass
class Character:
    id: str
    root: Path
    config: dict
    bible: str
    core: str | None = None
    style: str | None = None
    judge_rules: str = ""
    sources: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.config.get("display_name", self.id)

    def system_prompt(self) -> str:
        parts = [self.core or self.bible]
        if self.style:
            parts.append(self.style)
        return "\n\n".join(p.strip() for p in parts if p and p.strip())


def load_character(folder: str | Path) -> Character:
    root = Path(folder).resolve()
    data = json.loads((root / "config.json").read_text())
    for key in REQUIRED:
        if key not in data:
            raise ValueError(f"Missing field: {key}")
    bible = within(root, data["bible"]).read_text()
    sources = json.loads(within(root, data["sources"]).read_text())
    if not isinstance(sources, list):
        raise ValueError("Source manifest must be an array")
    if data["judge"]["mode"] not in ("shadow_only", "gate"):
        raise ValueError("judge.mode must be shadow_only or gate")
    optional = {}
    for name in ("core", "style"):
        path = root / f"{name}.md"
        optional[name] = path.read_text() if path.is_file() else None
    rules = data.get("judge", {}).get("rules_file")
    judge_rules = within(root, rules).read_text().strip() if rules else ""
    return Character(id=data["id"], root=root, config=data, bible=bible,
                     judge_rules=judge_rules, sources=sources, **optional)
