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


class Recipes(unittest.TestCase):
    def test_unpinned_preset_refused_unless_allowed(self):
        with self.assertRaises(SystemExit):
            pipeline.settings(ROOT / "characters/golem-15", model="qwen3.5-9b-base")
        s = pipeline.settings(ROOT / "characters/golem-15", model="qwen3.5-9b-base", allow_unpinned=True)
        self.assertEqual((s["revision"], s["unpinned"]), ("main", True))

    def test_preset_sets_model_and_lengths(self):
        s = pipeline.settings(ROOT / "characters/golem-15", model="qwen3.5-0.8b-base")
        self.assertEqual(s["model"], "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(s["sft"]["seq_len"], 512)
        self.assertEqual(s["recipe"], "character-pipeline-v0")

    def test_recipe_arms_filter_plan(self):
        tmp = Path(tempfile.mkdtemp())
        recipe = tmp / "r.json"
        recipe.write_text(json.dumps({"name": "sft-only", "arms": ["sft"]}))
        names = [st.name for st in pipeline.plan(ROOT / "characters/golem-15", tmp, seeds=[0], recipe=recipe)]
        self.assertIn("sft/seed0", names)
        self.assertFalse([n for n in names if n.startswith(("dpo", "kto", "pairs", "sample"))])
        self.assertEqual([n for n in names if n.startswith("eval")], ["eval/prompted_base", "eval/prompted_instruct", "eval/sft/seed0"])

    def test_prompted_instruct_blocked_until_pinned(self):
        steps = {st.name: st for st in pipeline.plan(ROOT / "characters/golem-15", Path(tempfile.mkdtemp()),
                                                     seeds=[0], model="qwen3.5-4b-base")}
        st = steps["eval/prompted_instruct"]
        self.assertTrue(st.optional and not st.ready())
        self.assertIn("Qwen/Qwen3.5-4B", [str(a) for a in st.args])

    def test_every_preset_and_recipe_parses(self):
        for f in (ROOT / "configs/models").glob("*.json"):
            m = json.loads(f.read_text())
            self.assertEqual(m["id"], f.stem)
            self.assertEqual(m["status"] == "unpinned", m["revision"] is None)
        for f in (ROOT / "recipes").glob("*.json"):
            self.assertIn("name", json.loads(f.read_text()))


class Staged(unittest.TestCase):
    def test_oct_recipe_plans_a_chain_per_seed(self):
        tmp = Path(tempfile.mkdtemp())
        steps = pipeline.plan(ROOT / "characters/golem-15", tmp, seeds=[0, 1], recipe=ROOT / "recipes/oct-2025.json")
        names = [st.name for st in steps]
        self.assertEqual(names[:3], ["prompts", "teacher_chosen", "teacher_chats"])
        by = {st.name: st for st in steps}
        # seed 1 starts from the shared chain, not from the end of seed 0
        self.assertEqual(by["student_rejected/seed1"].needs, by["teacher_chats"].outputs)
        self.assertEqual(by["dpo/seed1"].needs, by["pairs/seed1"].outputs)
        dpo = [str(a) for a in by["dpo/seed0"].args]
        self.assertIn("none", dpo)
        self.assertIn("sigmoid", dpo)
        self.assertIn("eval/oct-persona/seed1", names)
        self.assertTrue(all("{" not in str(a) for st in steps for a in st.args), "unfilled template")

    def test_marker_written_after_success(self):
        tmp = Path(tempfile.mkdtemp())
        marker = tmp / "x.done"
        step = pipeline.Step("t", "character_pipeline.data.chats", [tmp / "in.jsonl", "--out", tmp / "x"],
                             [marker], marker=marker)
        (tmp / "in.jsonl").write_text("")
        self.assertEqual(pipeline.execute([step], log=lambda *_: None), "done")
        self.assertTrue(marker.exists())


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
