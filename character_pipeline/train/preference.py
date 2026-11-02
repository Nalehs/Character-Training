"""Stage 7 preference arms: DPO and KTO from the shared SFT checkpoint.

Ported from Aineko's TRL 1.14.1 code (aineko-character-training 709b345):
DPO from oct/run_stage.py `dpo()` (sigmoid loss, beta 0.1, precomputed
reference log-probs) and KTO from loom/kto_pilot.py (`loss_type="kto"`).
Aineko's DPO completed; its KTO only ran as a 2-step pilot.

The reference is frozen at the arm's initializer. We merge the SFT adapter
into the base weights, then hand TRL the merged model plus a fresh LoRA config.
TRL's reference is then "adapter disabled", which is exactly the SFT
checkpoint. Aineko found that passing a PeftModel instead makes TRL 1.14.1 copy
a frozen reference adapter (DPO) or reuse the loaded adapter as the reference
(KTO), so this path is the one that pins the reference unambiguously.

    python -m character_pipeline.train.preference dpo --sft-adapter runs/sft/seed0/adapter \
        --data runs/clean/dpo_train.jsonl --out runs/dpo/seed0 --seed 0
    python -m character_pipeline.train.preference kto ... --data runs/clean/kto_train.jsonl

Rows (TRL conversational format):
  DPO  {"prompt": [messages], "chosen": [assistant msg], "rejected": [assistant msg]}
  KTO  {"prompt": [messages], "completion": [assistant msg], "label": true|false}

KTO labels must be independent absolute judgements; a DPO "rejected" side is
not automatically a KTO negative (RECIPE.md).

Defaults beyond beta and loss are not tuned for 4B: Aineko's lr 1e-6 was a CPU
feasibility value. Select lr/beta on development data, then freeze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from character_pipeline.train.common import (STUDENT, STUDENT_REVISION, load_student, lora_targets,
                                             precision, read_jsonl, torch_versions)


def validate_rows(kind: str, rows: list[dict]) -> None:
    need = {"dpo": ("prompt", "chosen", "rejected"), "kto": ("prompt", "completion", "label")}[kind]
    for n, row in enumerate(rows):
        missing = [k for k in need if k not in row]
        if missing:
            raise ValueError(f"row {n}: missing {missing}")
        if not isinstance(row["prompt"], list) or not row["prompt"]:
            raise ValueError(f"row {n}: prompt must be a non-empty message list")
        sides = ("chosen", "rejected") if kind == "dpo" else ("completion",)
        for side in sides:
            if not (isinstance(row[side], list) and len(row[side]) == 1 and row[side][0]["role"] == "assistant"):
                raise ValueError(f"row {n}: {side} must be one assistant message")
        if kind == "dpo" and row["chosen"][0]["content"] == row["rejected"][0]["content"]:
            raise ValueError(f"row {n}: chosen and rejected are identical")
        if kind == "kto" and not isinstance(row["label"], bool):
            raise ValueError(f"row {n}: label must be a boolean")
    if kind == "kto" and len({r["label"] for r in rows}) < 2:
        raise ValueError("KTO needs both desirable and undesirable examples")


def trainer_kwargs(kind: str, out: Path, args) -> dict:
    """TRL config arguments, as recorded in Aineko except where a GPU run differs."""
    common = dict(output_dir=str(out / "checkpoint"), per_device_train_batch_size=args.batch_size,
                  gradient_accumulation_steps=args.accumulation, num_train_epochs=args.epochs,
                  max_length=args.max_length, precompute_ref_log_probs=True,
                  gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                  learning_rate=args.lr, logging_steps=1, save_strategy="no", report_to="none",
                  seed=args.seed, use_liger_kernel=False)
    if kind == "dpo":
        return common | {"loss_type": ["sigmoid"], "beta": args.beta}
    return common | {"loss_type": "kto", "beta": args.beta}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["dpo", "kto"])
    ap.add_argument("--sft-adapter", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--revision", default=STUDENT_REVISION)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accumulation", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--local-files-only", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    import trl

    rows = read_jsonl(args.data)
    validate_rows(args.kind, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    tok, base = load_student(args.model, args.revision, args.device, args.local_files_only)
    tok.padding_side = "left"
    dtype, _ = precision(args.device)
    # Merge the shared SFT adapter: the merged weights ARE the arm's initializer and reference.
    merged = PeftModel.from_pretrained(base, str(args.sft_adapter)).merge_and_unload(safe_merge=True)
    peft_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                             target_modules=lora_targets(merged), bias="none")
    kw = trainer_kwargs(args.kind, args.out, args)
    kw |= {"bf16": dtype == torch.bfloat16, "fp16": dtype == torch.float16,
           "use_cpu": not args.device.startswith("cuda")}
    # Aineko rule: never let TRL clip a reviewed completion; drop the whole row and record it.
    sides = ("chosen", "rejected") if args.kind == "dpo" else ("completion",)
    def length(messages):
        ids = tok.apply_chat_template(messages, tokenize=True, enable_thinking=False)
        return len(ids["input_ids"] if hasattr(ids, "keys") else ids)

    kept, excluded = [], []
    for r in rows:
        n = max(length(r["prompt"] + r[s]) for s in sides)
        (kept if n <= args.max_length else excluded).append(r if n <= args.max_length else {"id": r.get("id"), "tokens": n})
    if not kept:
        raise SystemExit("every row exceeds --max-length")
    rows = kept
    columns = ("prompt", "chosen", "rejected") if args.kind == "dpo" else ("prompt", "completion", "label")
    dataset = Dataset.from_list([{k: r[k] for k in columns} | {"chat_template_kwargs": {"enable_thinking": False}}
                                 for r in rows])
    Config, Trainer = (trl.DPOConfig, trl.DPOTrainer) if args.kind == "dpo" else (trl.KTOConfig, trl.KTOTrainer)
    trainer = Trainer(model=merged, ref_model=None, peft_config=peft_config, args=Config(**kw),
                      processing_class=tok, train_dataset=dataset)
    if trainer.ref_model is not None or "ref" in getattr(trainer.model, "peft_config", {}):
        raise RuntimeError("reference must be the adapter-disabled merged SFT model")
    trainable = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    result = trainer.train()
    changed = sum(not torch.equal(p.detach(), trainable[n])
                  for n, p in trainer.model.named_parameters() if n in trainable)
    if changed == 0:
        raise RuntimeError("no adapter tensor changed")
    trainer.save_model(str(args.out / "adapter"))
    tok.save_pretrained(str(args.out / "adapter"))
    report = {"arm": args.kind, "sft_adapter": str(args.sft_adapter), "rows": len(rows),
              "excluded_over_max_length": excluded,
              "reference": "merged SFT weights with the new adapter disabled", "changed_tensors": changed,
              "global_step": trainer.state.global_step, "metrics": result.metrics,
              "config": {k: v for k, v in kw.items() if k != "output_dir"}, "versions": torch_versions(),
              "log_history": trainer.state.log_history}
    (args.out / "training_summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
