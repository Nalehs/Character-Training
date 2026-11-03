"""recipes/oct-2025.json is complete and its command args parse with this repo's modules."""
import importlib
import json
import unittest
from pathlib import Path

from character_pipeline.train.preference import trainer_kwargs

ROOT = Path(__file__).resolve().parents[1]
RECIPE = json.loads((ROOT / "recipes/oct-2025.json").read_text())


def resolved(args):
    return [a.replace("{runs}", "runs").replace("{character}", "golem-15").replace("{seed}", "0") for a in args]


class OctRecipe(unittest.TestCase):
    def stage(self, sid):
        return next(s for s in RECIPE["stages"] if s["id"] == sid)

    def test_fields(self):
        for key in ("name", "cite", "student", "teacher", "stages", "deviations"):
            self.assertIn(key, RECIPE)
        self.assertEqual(RECIPE["name"], "oct-2025")
        self.assertIn("2511.01689", json.dumps(RECIPE["cite"]))
        for s in RECIPE["stages"]:
            for name in s.get("hyperparameters", {}):
                self.assertIn(name, s.get("sources", {}), f"{s['id']}.{name} has no OCT source")
            for dev in s.get("deviations", []) + [s[k] for k in ("deviation",) if k in s]:
                self.assertIn(dev, RECIPE["deviations"])
        for part in ("student", "teacher"):
            self.assertIn(RECIPE[part]["deviation"], RECIPE["deviations"])

    def test_args_parse(self):
        for s in RECIPE["stages"]:
            if "args" not in s:
                continue
            module = importlib.import_module(s["module"])
            if hasattr(module, "parse"):
                module.parse(resolved(s["args"]))

    def test_dpo_args_match_hyperparameters(self):
        s = self.stage("dpo")
        args = importlib.import_module(s["module"]).parse(resolved(s["args"]))
        kw = trainer_kwargs("dpo", Path("x"), args)
        h = s["hyperparameters"]
        self.assertIsNone(args.sft_adapter)
        self.assertEqual(kw["loss_type"], ["sigmoid", "sft"])
        self.assertEqual(kw["loss_weights"], [1.0, h["nll_coef"]])
        self.assertEqual(kw["gradient_accumulation_steps"], h["global_batch"])
        self.assertEqual((kw["learning_rate"], kw["beta"], kw["warmup_steps"]), (h["lr"], h["beta"], h["warmup_ratio"]))
        self.assertEqual(kw["lr_scheduler_kwargs"]["min_lr_rate"], h["min_lr_rate"])
        self.assertEqual([0.9, kw["adam_beta2"]], h["adam_betas"])
        self.assertEqual((args.rank, args.alpha, args.lora_dropout, args.max_length),
                         (h["lora_r"], h["lora_alpha"], h["lora_dropout"], h["max_length"]))

    def test_sft_args_match_hyperparameters(self):
        s = self.stage("introspection_sft")
        a = importlib.import_module(s["module"]).parse(resolved(s["args"]))
        h = s["hyperparameters"]
        self.assertIsNotNone(a.merge_adapter)
        self.assertEqual((a.lr, a.warmup, a.min_lr_rate, a.adam_beta2, a.weight_decay, a.accumulation, a.seq_len),
                         (h["lr"], h["warmup_ratio"], h["min_lr_rate"], h["adam_betas"][1], h["weight_decay"],
                          h["global_batch"], h["max_length"]))

    def test_merge_and_introspection(self):
        m = self.stage("release_merge")
        a = importlib.import_module(m["module"]).parse(resolved(m["args"]))
        self.assertEqual(a.weights, [m["hyperparameters"]["weights"]["dpo"], m["hyperparameters"]["weights"]["sft"]])
        i = self.stage("introspection_data")
        a = importlib.import_module(i["module"]).parse(resolved(i["args"]))
        h = i["hyperparameters"]
        self.assertEqual((a.reflection_n, a.interaction_n, a.turns, a.temperature, a.top_p, a.repetition_penalty),
                         (h["reflection_n_per_prompt"], h["interaction_n_per_mode"], h["interaction_turns"],
                          h["temperature"], h["top_p"], h["repetition_penalty"]))
        self.assertEqual(10 * a.reflection_n + 2 * a.interaction_n, h["transcripts_total"])


if __name__ == "__main__":
    unittest.main()
