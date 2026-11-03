"""CPU smoke test for the OCT recipe (docs/recipes/oct-2025.md), stages 6-9, on a
tiny randomly initialised model built locally (no download):

  constitution prompts -> DPO from the bare student with sigmoid + 0.1 NLL
  -> introspection generation from the DPO adapter (tiny N and K)
  -> fresh-LoRA SFT on the DPO-merged weights -> linear merge of the two adapters

The chosen side is a fixed stand-in for teacher text; the rejected side is the
bare student's own sample, as in OCT. The model is random noise, so this checks
code paths and receipts, not character.

    PYTHONPATH=. python scripts/oct_smoke.py      # needs the [train] extras
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_smoke import tiny_model  # noqa: E402

OCT_SCHEDULE = ["--warmup", "0.1", "--scheduler", "cosine_with_min_lr", "--min-lr-rate", "0.1",
                "--adam-beta2", "0.98", "--weight-decay", "0", "--lora-dropout", "0"]


def main():
    import torch
    from peft import PeftModel
    from character_pipeline.data.constitution import prompt_rows, read_constitution
    from character_pipeline.generate import introspection
    from character_pipeline.train import merge, preference, sft
    from character_pipeline.train.sample import generate, load_policy

    constitution = ROOT / "characters/golem-15/constitution.json"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tiny_model(d / "model")
        local = ["--model", str(d / "model"), "--revision", "main", "--device", "cpu", "--local-files-only"]

        # Stage 4-5 stand-in: prompts from the constitution, rejected = the bare student's own sample.
        prompts = prompt_rows(read_constitution(constitution))[:8]
        tok, bare = load_policy(str(d / "model"), "main", None, "cpu", local_files_only=True)
        torch.manual_seed(0)
        pairs = []
        for row in prompts:
            text, _ = generate(tok, bare, row["messages"], 8, 0.7, 0.95, "cpu")
            chosen = "no evidence . i keep the answer ."
            pairs.append({"id": row["id"], "prompt": row["messages"],
                          "chosen": [{"role": "assistant", "content": chosen}],
                          "rejected": [{"role": "assistant", "content": text if text and text != chosen else "yes ."}]})
        del bare
        (d / "dpo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pairs))

        # Stage 6: DPO from the bare student, OCT loss and schedule (KL term not implemented).
        preference.main(["dpo", "--sft-adapter", "none", "--data", str(d / "dpo.jsonl"), "--out", str(d / "dpo"),
                         "--loss-type", "sigmoid", "sft", "--loss-weights", "1.0", "0.1", "--beta", "0.1",
                         "--lr", "5e-5", "--batch-size", "1", "--global-batch", "4", "--max-length", "256",
                         "--rank", "64", "--alpha", "128", "--max-grad-norm", "1.0", *OCT_SCHEDULE, *local])
        dpo = json.loads((d / "dpo/training_summary.json").read_text())
        assert dpo["sft_adapter"] == "none" and dpo["reference"].startswith("bare student"), dpo["reference"]
        assert dpo["config"]["loss_type"] == ["sigmoid", "sft"] and dpo["config"]["loss_weights"] == [1.0, 0.1]
        assert dpo["config"]["gradient_accumulation_steps"] == 4
        assert dpo["config"]["lr_scheduler_kwargs"] == {"min_lr_rate": 0.1}
        assert any("KL" in x for x in dpo["deviations"]), dpo.get("deviations")
        assert dpo["changed_tensors"] > 0

        # Stage 7: introspective transcripts from base + DPO adapter.
        introspection.main(["--adapter", str(d / "dpo/adapter"), "--constitution", str(constitution),
                            "--out", str(d / "intro"), "--reflection-n", "1", "--interaction-n", "2", "--turns", "3",
                            "--reflection-max-new-tokens", "8", "--interaction-max-new-tokens", "8",
                            "--validation-n", "2", "--seed", "0", *local])
        manifest = json.loads((d / "intro/manifest.json").read_text())
        counts = manifest["counts"]
        assert counts["self_reflection"] == 10 and counts["self_interaction-free"] == 2, counts
        assert counts["self_interaction-leading"] == 2 and counts["train"] == 12 and counts["validation"] == 2, counts
        rows = [json.loads(x) for x in (d / "intro/sft.jsonl").read_text().splitlines()]
        assert all(r["messages"][-1]["role"] == "assistant" for r in rows)
        raw = [json.loads(x) for x in (d / "intro/raw/self_interaction-leading.jsonl").read_text().splitlines()]
        assert "reflect and introspect" in raw[0]["system"]  # guidance line present

        # Stage 8: merge the DPO adapter into the weights and train a fresh LoRA on the transcripts.
        sft.main(["--merge-adapter", str(d / "dpo/adapter"), "--train", str(d / "intro/train.jsonl"),
                  "--validation", str(d / "intro/validation.jsonl"), "--out", str(d / "sft"), "--epochs", "1",
                  "--lr", "5e-5", "--rank", "64", "--alpha", "128", "--accumulation", "4", "--seq-len", "512",
                  "--warmup", "0.1", "--min-lr-rate", "0.1", "--adam-beta2", "0.98", "--weight-decay", "0",
                  "--lora-dropout", "0", *local])
        run_config = json.loads((d / "sft/run_config.json").read_text())
        assert run_config["merge_adapter"] == str(d / "dpo/adapter") and run_config["init_adapter"] == ""
        assert run_config["train_chats"] > 0, run_config
        sft_summary = json.loads((d / "sft/training_summary.json").read_text())
        assert sft_summary["completed"], sft_summary

        # Stage 9: linear merge, OCT code weights.
        merge.main(["--dpo", str(d / "dpo/adapter"), "--sft", str(d / "sft/adapter"), "--out", str(d / "persona"),
                    *local])
        merged = json.loads((d / "persona/merge_summary.json").read_text())
        assert merged["weights"] == {"dpo": 1.0, "sft": 0.25} and merged["nonzero_persona_tensors"] > 0, merged
        tok, base = load_policy(str(d / "model"), "main", None, "cpu", local_files_only=True)
        persona = PeftModel.from_pretrained(base, str(d / "persona/adapter"))
        x = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            on = persona(input_ids=x).logits
            with persona.disable_adapter():
                off = persona(input_ids=x).logits
        assert not torch.allclose(on, off), "merged adapter has no effect"

        report = {"dpo": {k: dpo[k] for k in ("rows", "global_step", "changed_tensors")},
                  "introspection": counts, "sft": {k: sft_summary[k] for k in ("updates", "validation_loss")},
                  "merge": {k: merged[k] for k in ("weights", "combination_type", "merged_rank")}}
        print(json.dumps(report, indent=2))
        print("oct smoke OK: DPO from bare student (+NLL), introspection, fresh-LoRA SFT on merged DPO, linear merge")


if __name__ == "__main__":
    main()
