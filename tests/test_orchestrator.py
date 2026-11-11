"""The resumable runner: crashes, stale outputs, human stops, validation, locks. No model needed."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from character_pipeline import cli
from character_pipeline import run as pipeline

ROOT = Path(__file__).resolve().parents[1]
GOLEM = ROOT / "characters/golem-15"

STAGE = '''
import sys, time
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--out") + 1])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("x") as f:                      # like train.sample: never overwrite
    f.write("row 1\\n")
    f.flush()
    if "--sleep" in args:
        time.sleep(float(args[args.index("--sleep") + 1]))
    if Path(args[args.index("--crash-if") + 1]).exists():
        sys.exit(7)                            # dies half-way, partial file left behind
    f.write("row 2\\n")
'''


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "fake_stage.py").write_text(STAGE)
        env = mock.patch.dict(os.environ, {"PYTHONPATH": f"{self.tmp}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"})
        env.start()
        self.addCleanup(env.stop)
        self.run_dir = self.tmp / "runs" / "x"
        self.out = self.run_dir / "out.jsonl"
        self.crash = self.tmp / "crash"
        self.input = self.tmp / "input.txt"
        self.input.write_text("v1")

    def step(self, *extra, name="fake"):
        return pipeline.Step(name, "fake_stage", ["--in", self.input, "--out", self.out, "--crash-if", self.crash,
                                                  *extra], [self.out], state=self.run_dir / ".steps",
                             run_dir=self.run_dir)

    def test_crash_half_way_is_not_done_and_is_redone(self):
        self.crash.write_text("")
        log = []
        self.assertEqual(pipeline.execute([self.step()], log=log.append), "failed:fake")
        self.assertTrue(self.out.exists(), "the fake stage leaves a partial file, like open('x') stages do")
        self.assertFalse(self.step().done(), "a partial output must not count as finished")
        self.crash.unlink()
        self.assertEqual(pipeline.execute([self.step()], log=log.append), "done")
        self.assertEqual(self.out.read_text(), "row 1\nrow 2\n")
        kept = list((self.run_dir / ".interrupted").glob("fake-*/out.jsonl"))
        self.assertEqual([p.read_text() for p in kept], ["row 1\n"], "the partial file is moved, not deleted")
        self.assertTrue(self.step().done())

    def test_exit_zero_without_output_is_a_failure(self):
        step = pipeline.Step("t", "fake_stage", ["--out", self.tmp / "elsewhere", "--crash-if", self.crash],
                             [self.out], state=self.run_dir / ".steps", run_dir=self.run_dir)
        self.assertEqual(pipeline.execute([step], log=lambda *_: None), "failed:t")

    def test_changed_command_or_input_is_stale_not_reused(self):
        self.assertEqual(pipeline.execute([self.step()], log=lambda *_: None), "done")
        self.assertIsNone(self.step().stale())
        self.assertIsNone(self.step("--device", "cuda:1", "--local-files-only").stale(), "machine flags are not settings")
        log = []
        self.assertEqual(pipeline.execute([self.step("--lr", "3")], log=log.append), "stale:fake")
        self.assertIn("--lr", "\n".join(log))
        self.assertEqual(pipeline.execute([self.step("--lr", "3")], log=log.append, reuse_stale=True), "done")
        self.input.write_text("v2")
        self.assertIn("input changed", self.step().stale())
        self.assertEqual(pipeline.execute([self.step()], dry_run=True, log=log.append), "dry-run")

    def test_outputs_without_a_receipt_count_as_done(self):
        self.out.parent.mkdir(parents=True)
        self.out.write_text("by hand\n")
        self.assertEqual(pipeline.execute([self.step("--lr", "9")], log=lambda *_: None), "done")
        self.assertEqual(self.out.read_text(), "by hand\n")

    def test_two_runs_of_one_step_do_not_overlap(self):
        results = []
        def go():
            results.append(pipeline.execute([self.step("--sleep", "1.0")], log=lambda *_: None))
        threads = [threading.Thread(target=go) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # without the lock the second process dies on open("x") and the step "fails"
        self.assertEqual(results, ["done", "done"])
        self.assertEqual(self.out.read_text(), "row 1\nrow 2\n")

    def test_only_and_until_match_whole_names(self):
        steps = [self.step(name="sft/seed0"), self.step(name="sample/seed0")]
        log = []
        with self.assertRaises(SystemExit):
            pipeline.execute(steps, only="s", dry_run=True, log=log.append)
        pipeline.execute(steps, only="sft", dry_run=True, log=log.append)
        self.assertEqual([line.split()[1] for line in log if line.startswith("[todo]")], ["sft/seed0"])


class Plans(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def character(self, **pipeline_block):
        d = self.tmp / "c"
        shutil.copytree(GOLEM, d, dirs_exist_ok=True)
        cfg = json.loads((GOLEM / "config.json").read_text())
        if pipeline_block:
            cfg["pipeline"] = pipeline_block
        (d / "config.json").write_text(json.dumps(cfg))
        return d

    def test_unfilled_review_sheet_stops_the_run(self):
        steps = pipeline.plan(GOLEM, self.tmp, seeds=[0])
        R = self.tmp / "golem-15"
        R.mkdir()
        for name in ("teacher.jsonl", "checked.jsonl"):
            (R / name).write_text("{}\n")
        (R / "judged.jsonl").write_text(json.dumps({"scenario_id": "S", "seed": 1, "history": [
            {"speaker": "partner", "text": "q"}, {"speaker": "character", "text": "a"}]}) + "\n")
        pipeline.review_sheet(R / "judged.jsonl", R / "review.jsonl")  # the sheet, copied without deciding anything
        steps = pipeline.plan(GOLEM, self.tmp, seeds=[0])
        log = []
        self.assertEqual(pipeline.execute(steps, log=log.append), "stopped:clean")
        self.assertIn("1 of 1 turns have no decision", "\n".join(log))
        self.assertFalse((R / "clean").exists())
        (R / "review.jsonl").write_text(json.dumps({"conversation": "S:1", "turn": 0, "decision": "accept"}) + "\n")
        self.assertIsNone({s.name: s for s in pipeline.plan(GOLEM, self.tmp, seeds=[0])}["clean"].blocked)

    def test_receipt_tracks_what_the_stage_reads_from_the_character(self):
        d = self.character()
        generate = pipeline.plan(d, self.tmp / "runs", seeds=[0])[0]
        before = generate.receipt()
        self.assertIn("{character_dir}", before["inputs"])
        self.assertIn("{character_dir}/scenarios.jsonl", before["inputs"])
        cfg = json.loads((d / "config.json").read_text())
        cfg["status"], cfg["pipeline"] = "trained", {"sft": {"lr": 5e-5}}
        (d / "config.json").write_text(json.dumps(cfg))
        self.assertEqual(pipeline.plan(d, self.tmp / "runs", seeds=[0])[0].receipt(), before,
                         "status and training settings do not make teacher chats stale")
        (d / "BIBLE.md").write_text("another character")
        self.assertNotEqual(pipeline.plan(d, self.tmp / "runs", seeds=[0])[0].receipt(), before)

    def test_receipt_does_not_depend_on_where_runs_lives(self):
        a = {s.name: s.receipt() for s in pipeline.plan(GOLEM, self.tmp / "a", seeds=[0])}
        b = {s.name: s.receipt() for s in pipeline.plan(GOLEM, self.tmp / "b", seeds=[0], device="cpu")}
        self.assertEqual(a, b)

    def test_changed_model_makes_training_steps_stale(self):
        small = {s.name: s.receipt()["command"] for s in pipeline.plan(GOLEM, self.tmp, seeds=[0], model="qwen3.5-0.8b-base")}
        big = {s.name: s.receipt()["command"] for s in pipeline.plan(GOLEM, self.tmp, seeds=[0], model="qwen3.5-4b-base")}
        self.assertNotEqual(small["sft/seed0"], big["sft/seed0"])
        self.assertEqual(small["generate"], big["generate"])

    def test_typos_fail_with_a_hint(self):
        with self.assertRaises(SystemExit) as e:
            pipeline.settings(self.character(sedds=[1]))
        self.assertIn("did you mean 'seeds'", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            pipeline.settings(self.character(sft={"learning_rate": 1e-4}))
        self.assertIn("pipeline.sft", str(e.exception))
        for bad, word in (({"name": "r", "arm": ["sft"]}, "arms"), ({"name": "r", "arms": ["dpo2"]}, "dpo"),
                          ({"arms": ["sft"]}, "name"),
                          ({"name": "r", "stages": [{"id": "a", "module": "os", "args": []}]}, "character_pipeline"),
                          ({"name": "r", "stages": [{"id": "a", "module": "character_pipeline.x", "args": ["{sed}"]}]}, "{sed}"),
                          ({"name": "r", "stages": [{"id": "a", "module": "character_pipeline.x", "args": [1]}]}, "strings")):
            with self.assertRaises(SystemExit, msg=bad) as e:
                pipeline.validate_recipe(bad, "r.json")
            self.assertIn(word, str(e.exception))
        for f in (ROOT / "recipes").glob("*.json"):
            pipeline.validate_recipe(json.loads(f.read_text()), f.name)

    def test_missing_recipe_file_is_an_error_not_the_default(self):
        with self.assertRaises(SystemExit) as e:
            pipeline.settings(GOLEM, recipe=self.tmp / "oct-2052.json")
        self.assertIn("not found", str(e.exception))

    def test_merge_precedence_matches_the_docs(self):
        # defaults < recipe < character pipeline < --model preset
        recipe = self.tmp / "r.json"
        recipe.write_text(json.dumps({"name": "r", "pipeline": {"sft": {"epochs": 9, "lr": 9.0}, "seeds": [5]}}))
        d = self.character(sft={"lr": 7.0, "seq_len": 100})
        s = pipeline.settings(d, recipe)
        self.assertEqual((s["sft"], s["seeds"], s["preference"]["beta"]),
                         ({"epochs": 9, "lr": 7.0, "seq_len": 100}, [5], 0.1))
        s = pipeline.settings(d, recipe, model="qwen3.5-0.8b-base")
        self.assertEqual((s["sft"]["seq_len"], s["sft"]["lr"], s["model"]), (512, 7.0, "Qwen/Qwen3.5-0.8B-Base"))

    def test_unpinned_student_in_a_character_is_refused_too(self):
        d = self.character()
        cfg = json.loads((d / "config.json").read_text())
        for revision in ("main", "1001bb4d", None):
            cfg["student"]["revision"] = revision
            (d / "config.json").write_text(json.dumps(cfg))
            with self.assertRaises(SystemExit, msg=revision):
                pipeline.settings(d)
        self.assertEqual(pipeline.settings(d, allow_unpinned=True)["revision"], "main")

    def test_presets_are_pinned_to_full_hashes_or_null(self):
        for f in (ROOT / "configs/models").glob("*.json"):
            pipeline.model_preset(f.stem)
        with self.assertRaises(SystemExit):
            pipeline.model_preset("../../recipes/oct-2025")

    def test_resolved_recipe_records_the_seeds_actually_run(self):
        out = pipeline.resolved(GOLEM, self.tmp, seeds=[4])
        self.assertEqual(json.loads(out.read_text())["seeds"], [4])


class CliSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.chars = self.tmp / "characters"
        shutil.copytree(ROOT / "characters/_template", self.chars / "_template")

    def test_new_refuses_ids_that_leave_the_characters_folder(self):
        for bad in ("../escape", "a/b", "", ".", "..", "_template", "/abs", "-x"):
            with self.assertRaises(SystemExit, msg=bad):
                cli.new(Namespace(characters=str(self.chars), id=bad, name=None))
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["characters"])
        self.assertEqual(sorted(p.name for p in self.chars.iterdir()), ["_template"])

    def test_review_sheet_explains_a_missing_judge_file_and_keeps_an_existing_sheet(self):
        cli.new(Namespace(characters=str(self.chars), id="moth", name=None))
        base = ["--characters", str(self.chars), "--runs", str(self.tmp / "runs")]
        with self.assertRaises(SystemExit) as e:
            cli.main([*base, "review-sheet", str(self.chars / "moth")])
        self.assertIn("judged.jsonl", str(e.exception))
        R = self.tmp / "runs" / "moth"
        R.mkdir(parents=True)
        (R / "judged.jsonl").write_text("")
        (R / "review.todo.jsonl").write_text("half filled in\n")
        with self.assertRaises(SystemExit):
            cli.main([*base, "review-sheet", str(self.chars / "moth")])
        self.assertEqual((R / "review.todo.jsonl").read_text(), "half filled in\n")

    def test_run_exit_codes(self):
        cli.new(Namespace(characters=str(self.chars), id="moth", name=None))
        done = subprocess.run([sys.executable, "-m", "character_pipeline.cli", "--runs", str(self.tmp / "runs"), "run",
                               str(self.chars / "moth"), "--only", "clean"], capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 2, done.stderr)  # waiting for the human review
        self.assertIn("review-sheet", done.stdout)


class CharacterCopies(unittest.TestCase):
    def test_environment_copy_of_golem_matches_the_character_folder(self):
        copy = ROOT / "environments/character_pushback/character_pushback/data/golem-15"
        names = sorted(p.name for p in copy.iterdir() if p.is_file())
        self.assertEqual(names, ["BIBLE.md", "JUDGE_RULES.md", "config.json"])
        for name in names:
            self.assertEqual((copy / name).read_bytes(), (GOLEM / name).read_bytes(), f"{name} has diverged")


if __name__ == "__main__":
    unittest.main()
