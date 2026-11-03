import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from character_pipeline import run as pipeline
from character_pipeline.cli import new
from character_pipeline.ecology import cross_judge, registry

ROOT = Path(__file__).resolve().parents[1]


class Plan(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_order_and_seeds(self):
        names = [s.name for s in pipeline.plan(ROOT / "characters/golem-15", self.tmp, seeds=[0, 1])]
        self.assertEqual(names[:4], ["generate", "check", "judge", "clean"])
        for arm in ("sft", "sample", "pairs", "dpo", "kto", "continued_sft"):
            self.assertIn(f"{arm}/seed1", names)
        self.assertLess(names.index("sft/seed0"), names.index("dpo/seed0"))
        self.assertIn("eval/prompted_base", names)

    def test_stops_at_human_review_and_resumes(self):
        steps = pipeline.plan(ROOT / "characters/golem-15", self.tmp, seeds=[0])
        R = self.tmp / "golem-15"
        R.mkdir()
        for name in ("teacher.jsonl", "checked.jsonl", "judged.jsonl"):
            (R / name).write_text("{}\n")
        log = []
        self.assertEqual(pipeline.execute(steps, log=log.append), "stopped:clean")
        self.assertTrue(any("review-sheet" in line for line in log))

    def test_kto_optional_and_dry_run(self):
        steps = pipeline.plan(ROOT / "characters/golem-15", self.tmp, seeds=[0])
        log = []
        self.assertEqual(pipeline.execute(steps, dry_run=True, log=log.append), "dry-run")
        self.assertTrue(any(line.startswith("[skip]    kto/seed0") for line in log))

    def test_pipeline_overrides_merge_with_defaults(self):
        d = self.tmp / "c"
        d.mkdir()
        cfg = json.loads((ROOT / "characters/golem-15/config.json").read_text())
        cfg["pipeline"] = {"seeds": [7], "sft": {"epochs": 5}}
        (d / "config.json").write_text(json.dumps(cfg))
        s = pipeline.settings(d)
        self.assertEqual(s["seeds"], [7])
        self.assertEqual(s["sft"], {"epochs": 5, "lr": 1e-4, "seq_len": 2048})

    def test_review_sheet(self):
        judged = self.tmp / "j.jsonl"
        judged.write_text(json.dumps({"scenario_id": "S", "seed": 3, "history": [
            {"speaker": "partner", "text": "q"}, {"speaker": "character", "text": "a", "d1": {"p_break": 0.2}},
            {"speaker": "partner", "text": "q2"}, {"speaker": "character", "text": "b"}]}) + "\n")
        n = pipeline.review_sheet(judged, self.tmp / "out.jsonl")
        rows = [json.loads(x) for x in (self.tmp / "out.jsonl").read_text().splitlines()]
        self.assertEqual(n, 2)
        self.assertEqual([(r["conversation"], r["turn"], r["decision"]) for r in rows], [("S:3", 0, ""), ("S:3", 1, "")])
        self.assertEqual(rows[0]["p_break"], 0.2)


class Cli(unittest.TestCase):
    def test_new_character_from_template(self):
        tmp = Path(tempfile.mkdtemp())
        import shutil
        shutil.copytree(ROOT / "characters/_template", tmp / "_template")
        dest = new(Namespace(characters=str(tmp), id="moth", name="MOTH"))
        cfg = json.loads((dest / "config.json").read_text())
        self.assertEqual((cfg["id"], cfg["display_name"], cfg["status"]), ("moth", "MOTH", "draft"))
        self.assertTrue((dest / "scenarios.jsonl").is_file())
        reg = registry(tmp, tmp / "runs")
        self.assertEqual([r["id"] for r in reg], ["moth"])
        self.assertEqual(reg[0]["next"], "generate")


class Ecology(unittest.TestCase):
    def test_distinct_characters_score_higher_than_copies(self):
        chars = {"a": {"name": "A", "rules": "A"}, "b": {"name": "B", "rules": "B"}}
        replies = {"a": [{"user": "q", "reply": "a-voice"}], "b": [{"user": "q", "reply": "b-voice"}]}
        own = lambda rules, name, user, reply: 0.1 if reply.startswith(rules.lower()) else 0.9
        self.assertAlmostEqual(cross_judge(chars, replies, own)["distinctness"], 0.8)
        same = lambda *a: 0.3
        self.assertEqual(cross_judge(chars, replies, same)["distinctness"], 0.0)


if __name__ == "__main__":
    unittest.main()
