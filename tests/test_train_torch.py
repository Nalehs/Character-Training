"""Training code paths on a tiny random model (CPU, no download). Skipped without the [train] extras.

One SFT, one DPO and one KTO run are shared by the tests below; they check what the saved files
mean, not the losses, which are noise on a random model.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import trl  # noqa: F401
    from peft import PeftModel
except ImportError:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[1]
X = [[5, 6, 7, 8, 9, 10]]


@unittest.skipIf(torch is None, "torch, peft and trl are not installed")
class TinyRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        from train_smoke import chat, tiny_model
        from character_pipeline.data.pairs import build
        from character_pipeline.train import preference, sft
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls.tmp.name)
        tiny_model(d / "model")
        rows = [chat(i) for i in range(8)]
        (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:6]))
        (d / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[6:]))
        cls.local = ["--model", str(d / "model"), "--revision", "main", "--device", "cpu", "--local-files-only"]
        cls.sft_args = ["--train", str(d / "train.jsonl"), "--validation", str(d / "valid.jsonl"),
                        "--seq-len", "128", *cls.local]
        # A large learning rate so the SFT update is far above rounding noise.
        sft.main([*cls.sft_args, "--out", str(d / "sft"), "--epochs", "4", "--lr", "1e-2", "--seed", "0"])
        samples = {f"c{i}:{k}": "yes ." for i in range(6) for k in range(2)}
        labels = {(f"c{i}:{k}", s): s == "teacher" for i in range(6) for k in range(2) for s in ("teacher", "student")}
        dpo, kto, _ = build(rows[:6], samples, labels)
        (d / "dpo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dpo))
        (d / "kto.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kto))
        pref = ["--max-length", "128", "--lr", "1e-3", "--sft-adapter", str(d / "sft/adapter"), "--seed", "0", *cls.local]
        preference.main(["dpo", "--data", str(d / "dpo.jsonl"), "--out", str(d / "dpo"), *pref])
        preference.main(["kto", "--data", str(d / "kto.jsonl"), "--out", str(d / "kto"), "--batch-size", "2", *pref])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def bare(self):
        from character_pipeline.train.common import load_student
        return load_student(str(self.d / "model"), "main", "cpu", True)[1].eval()

    def logits(self, model):
        with torch.no_grad():
            return model(input_ids=torch.tensor(X)).logits

    def test_saved_preference_adapter_on_the_bare_student_is_the_trained_arm(self):
        """evaluate/serve apply <arm>/adapter to the bare student; that must equal SFT-merged + trained delta."""
        from character_pipeline.train.sample import load_policy
        sft_only = self.logits(PeftModel.from_pretrained(self.bare(), str(self.d / "sft/adapter")))
        self.assertGreater(float((sft_only - self.logits(self.bare())).abs().max()), 0.1)  # SFT did something
        for arm in ("dpo", "kto"):
            merged = PeftModel.from_pretrained(self.bare(), str(self.d / "sft/adapter")).merge_and_unload()
            trained = self.logits(PeftModel.from_pretrained(merged, str(self.d / arm / "delta_adapter")))
            served = self.logits(load_policy(str(self.d / "model"), "main", self.d / arm / "adapter", "cpu", True)[1])
            self.assertGreater(float((trained - sft_only).abs().max()), 1e-4, arm)  # the arm moved off SFT
            self.assertLess(float((served - trained).abs().max()), 1e-4, arm)
            saved = json.loads((self.d / arm / "training_summary.json").read_text())["saved"]
            self.assertEqual(saved["composed"]["rank"], 32)
            self.assertLess(saved["composed"]["max_abs_error"], 1e-5)

    def test_receipts_name_the_model_revision_seed_and_versions(self):
        for arm in ("sft", "dpo", "kto"):
            config = json.loads((self.d / arm / "run_config.json").read_text())
            self.assertEqual((config["model"], config["revision"], config["seed"]), (str(self.d / "model"), "main", 0))
            self.assertFalse(config["revision_pinned"])
            self.assertEqual(len(config["chat_template_sha256"]), 64)
            for package in ("python", "torch", "transformers", "peft"):
                self.assertIn(package, config["versions"])
        self.assertIn("trl", json.loads((self.d / "dpo/run_config.json").read_text())["versions"])

    def test_merge_check_is_recorded_and_small_in_fp32(self):
        check = json.loads((self.d / "dpo/training_summary.json").read_text())["merge_check"]
        self.assertGreater(check["max_abs_logit_effect_of_sft_adapter"], 0.1)
        self.assertLess(check["max_abs_logit_gap_merged_vs_attached"], 1e-4)

    def test_kto_batches_are_mismatched_and_counts_recorded(self):
        kto = json.loads((self.d / "kto/training_summary.json").read_text())["kto"]
        self.assertEqual((kto["desirable"], kto["undesirable"], kto["batches_sharing_a_prompt"]), (12, 12, 0))

    def test_continued_sft_records_the_loaded_adapter_not_the_flags(self):
        from character_pipeline.train import sft
        out = self.d / "cont"
        sft.main([*self.sft_args, "--out", str(out), "--epochs", "1", "--init-adapter", str(self.d / "sft/adapter"),
                  "--rank", "64", "--alpha", "128", "--lora-dropout", "0"])
        config = json.loads((out / "run_config.json").read_text())
        self.assertEqual((config["rank"], config["alpha"], config["lora_dropout"]), (16, 32, 0.05))
        summary = json.loads((out / "training_summary.json").read_text())
        self.assertGreater(summary["validation_token_nll"], 0)

    def test_resume_refuses_a_checkpoint_from_another_run_and_keeps_one_complete_checkpoint(self):
        from character_pipeline.train import sft
        out = self.d / "resume"
        sft.main([*self.sft_args, "--out", str(out), "--epochs", "1", "--seed", "1"])
        self.assertTrue((out / "checkpoint/state.pt").exists() and (out / "checkpoint/adapter_config.json").exists())
        self.assertFalse((out / "checkpoint.new").exists() or (out / "checkpoint.old").exists())
        sft.main([*self.sft_args, "--out", str(out), "--epochs", "1", "--seed", "1", "--resume"])  # same run: fine
        self.assertEqual(json.loads((out / "run_config.json").read_text())["resumed_from_microstep"], 6)
        with self.assertRaises(SystemExit):
            sft.main([*self.sft_args, "--out", str(out), "--epochs", "1", "--seed", "2", "--resume"])
        # A save interrupted between the two renames leaves only checkpoint.old; resume must find it.
        (out / "checkpoint").rename(out / "checkpoint.old")
        self.assertEqual(sft.latest_checkpoint(out / "checkpoint"), out / "checkpoint.old")

    def test_sampling_is_seeded_complete_or_absent_and_never_overwrites(self):
        from character_pipeline.train import sample
        args = ["--chats", str(self.d / "train.jsonl"), "--max-new-tokens", "4", "--seed", "3", *self.local]
        sample.main([*args, "--out", str(self.d / "s1.jsonl")])
        sample.main([*args, "--out", str(self.d / "s2.jsonl")])
        self.assertEqual((self.d / "s1.jsonl").read_text(), (self.d / "s2.jsonl").read_text())
        self.assertEqual(len((self.d / "s1.jsonl").read_text().splitlines()), 12)
        self.assertFalse((self.d / "s1.jsonl.partial").exists())
        meta = json.loads((self.d / "s1.jsonl.meta.json").read_text())
        self.assertEqual((meta["seed"], meta["samples"], meta["temperature"]), (3, 12, 0.8))
        with self.assertRaises(SystemExit):
            sample.main([*args, "--out", str(self.d / "s1.jsonl")])
        # An interrupted run leaves no samples file, so the orchestrator cannot take it for finished.
        original = sample.generate
        calls = []

        def dies(*a, **k):
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt
            return original(*a, **k)
        sample.generate = dies
        try:
            with self.assertRaises(KeyboardInterrupt):
                sample.main([*args, "--out", str(self.d / "s3.jsonl")])
        finally:
            sample.generate = original
        self.assertFalse((self.d / "s3.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
