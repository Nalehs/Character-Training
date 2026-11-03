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

OCT stage 6 (docs/recipes/oct-2025.md) is reachable with flags only; every new
flag defaults to "leave TRL's value alone", so existing arms are unchanged:

    python -m character_pipeline.train.preference dpo --sft-adapter none \
        --data runs/oct/pairs/dpo_train.jsonl --out runs/oct/dpo/seed0 \
        --loss-type sigmoid sft --loss-weights 1.0 0.1 --beta 0.1 --lr 5e-5 \
        --batch-size 1 --global-batch 32 --max-length 1024 --rank 64 --alpha 128 \
        --warmup 0.1 --scheduler cosine_with_min_lr --min-lr-rate 0.1 --adam-beta2 0.98 \
        --weight-decay 0 --max-grad-norm 1.0 --lora-dropout 0

--sft-adapter none trains DPO from the bare student, so the reference is the
bare student (adapter disabled). TRL's "sft" loss is a token-mean cross-entropy
on the chosen completion; with weights [1.0, 0.1] it is OCT's NLL term. OCT's
squared per-token KL term (coefficient 0.001) is NOT implemented; runs that use
"sft" record that under `deviations` in training_summary.json.
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


# Recorded in training_summary.json whenever an OCT-style NLL term is used.
OCT_DEVIATIONS = [
    "no squared per-token KL term: OCT adds 0.001 * mean((logp_policy - logp_ref)^2) over completion tokens "
    "(maiush/OpenRLHF@eaf40e1 openrlhf/trainer/dpo_trainer.py:178, 386-402); TRL 1.14.1 has no such option",
    "NLL normalisation: TRL 'sft' averages cross-entropy over all chosen tokens in the batch; the OpenRLHF fork "
    "averages per sample, then over the batch (dpo_trainer.py:324)",
    "reference log-probs are precomputed once (precompute_ref_log_probs=True); OCT runs the reference live. "
    "Equivalent for a frozen reference, up to numerical precision",
]


def parse_adapter(value: str) -> Path | None:
    """'none' (any case) means no initial adapter: start from the bare student."""
    return None if value.lower() == "none" else Path(value)


def accumulation_steps(args) -> int:
    """Gradient accumulation; --global-batch (single process) overrides --accumulation."""
    global_batch = getattr(args, "global_batch", None)
    if global_batch is None:
        return args.accumulation
    if global_batch <= 0 or global_batch % args.batch_size:
        raise ValueError(f"--global-batch {global_batch} must be a positive multiple of --batch-size {args.batch_size}")
    return global_batch // args.batch_size


def deviations(args) -> list[str]:
    return list(OCT_DEVIATIONS) if "sft" in (getattr(args, "loss_type", None) or []) else []


def trainer_kwargs(kind: str, out: Path, args) -> dict:
    """TRL config arguments, as recorded in Aineko except where a GPU run differs.

    Optimiser/schedule options are added only when set, so TRL's defaults
    (linear schedule, no warm-up, beta2 0.999, no weight decay, clip 1.0) stay
    in force for the existing arms.
    """
    common = dict(output_dir=str(out / "checkpoint"), per_device_train_batch_size=args.batch_size,
                  gradient_accumulation_steps=accumulation_steps(args), num_train_epochs=args.epochs,
                  max_length=args.max_length, precompute_ref_log_probs=True,
                  gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                  learning_rate=args.lr, logging_steps=1, save_strategy="no", report_to="none",
                  seed=args.seed, use_liger_kernel=False)
    optional = {"warmup_steps": getattr(args, "warmup", None), "lr_scheduler_type": getattr(args, "scheduler", None),
                "adam_beta2": getattr(args, "adam_beta2", None), "weight_decay": getattr(args, "weight_decay", None),
                "max_grad_norm": getattr(args, "max_grad_norm", None)}
    common |= {k: v for k, v in optional.items() if v is not None}
    min_lr_rate = getattr(args, "min_lr_rate", None)
    if min_lr_rate is not None:
        if common.get("lr_scheduler_type") != "cosine_with_min_lr":
            raise ValueError("--min-lr-rate needs --scheduler cosine_with_min_lr")
        common["lr_scheduler_kwargs"] = {"min_lr_rate": min_lr_rate}
    loss_type = getattr(args, "loss_type", None)
    loss_weights = getattr(args, "loss_weights", None)
    if kind == "dpo":
        loss_type = list(loss_type or ["sigmoid"])
        if loss_weights is not None and len(loss_weights) != len(loss_type):
            raise ValueError("--loss-weights needs one weight per --loss-type")
        return common | {"loss_type": loss_type, "beta": args.beta} | (
            {"loss_weights": list(loss_weights)} if loss_weights is not None else {})
    if (loss_type and list(loss_type) != ["sigmoid"]) or loss_weights is not None:
        raise ValueError("--loss-type/--loss-weights apply to DPO only")
    return common | {"loss_type": "kto", "beta": args.beta}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["dpo", "kto"])
    ap.add_argument("--sft-adapter", type=parse_adapter, required=True,
                    help="initializer and reference; 'none' = the bare student (OCT stage 6)")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--revision", default=STUDENT_REVISION)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accumulation", type=int, default=4)
    ap.add_argument("--global-batch", type=int, help="derive --accumulation as global batch / --batch-size")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--loss-type", nargs="+", default=["sigmoid"],
                    help="DPO loss terms (TRL multi-loss), e.g. 'sigmoid sft' for OCT's DPO + NLL")
    ap.add_argument("--loss-weights", nargs="+", type=float, help="one weight per --loss-type, e.g. 1.0 0.1")
    ap.add_argument("--warmup", type=float, help="warm-up ratio (<1) or steps; default: TRL's 0")
    ap.add_argument("--scheduler", help="transformers scheduler name, e.g. cosine_with_min_lr; default: linear")
    ap.add_argument("--min-lr-rate", type=float, help="floor for cosine_with_min_lr, as a fraction of lr")
    ap.add_argument("--adam-beta2", type=float, help="default: TRL's 0.999")
    ap.add_argument("--weight-decay", type=float, help="default: TRL's 0.0")
    ap.add_argument("--max-grad-norm", type=float, help="default: TRL's 1.0")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
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
    if args.sft_adapter is None:
        # OCT stage 6: DPO from the bare student, which is then also the reference.
        merged = base
    else:
        # Merge the shared SFT adapter: the merged weights ARE the arm's initializer and reference.
        merged = PeftModel.from_pretrained(base, str(args.sft_adapter)).merge_and_unload(safe_merge=True)
    peft_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.lora_dropout,
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
        raise RuntimeError("reference must be the adapter-disabled initializer")
    trainable = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    result = trainer.train()
    changed = sum(not torch.equal(p.detach(), trainable[n])
                  for n, p in trainer.model.named_parameters() if n in trainable)
    if changed == 0:
        raise RuntimeError("no adapter tensor changed")
    trainer.save_model(str(args.out / "adapter"))
    tok.save_pretrained(str(args.out / "adapter"))
    reference = ("bare student with the new adapter disabled" if args.sft_adapter is None
                 else "merged SFT weights with the new adapter disabled")
    report = {"arm": args.kind, "sft_adapter": str(args.sft_adapter) if args.sft_adapter else "none",
              "rows": len(rows), "excluded_over_max_length": excluded,
              "reference": reference, "changed_tensors": changed,
              "lora": {"r": args.rank, "alpha": args.alpha, "dropout": args.lora_dropout},
              "global_step": trainer.state.global_step, "metrics": result.metrics,
              "config": {k: v for k, v in kw.items() if k != "output_dir"}, "versions": torch_versions(),
              "log_history": trainer.state.log_history}
    if deviations(args):
        report["deviations"] = deviations(args)
    (args.out / "training_summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
