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

What is saved. The fresh LoRA is a delta on the SFT-merged weights, so on the
bare student it is not the arm. `<out>/adapter` is therefore the composition
SFT + delta (PEFT `cat`, rank r_sft + r_delta, the exact sum of the two weight
updates) and applies to the bare student like every other arm's adapter.
`<out>/delta_adapter` is the fresh LoRA alone, for analysis; it is only
meaningful on top of the merged SFT weights. With --sft-adapter none there is
nothing to compose and `<out>/adapter` is the trained LoRA itself.

Two things that differ from the SFT arm and are recorded, not hidden:
merging rounds the SFT update into the student's fp16/bf16 weights, so the arm
trains against (and its reference is) a rounded copy of the SFT checkpoint
(`merge_check` in training_summary.json measures the logit gap on one row);
and TRL's completion tokens include the newline the template writes after
<|im_end|>, which train.sft does not supervise.

KTO order. TRL's KTO reads the data sequentially and estimates its KL baseline
from completions swapped inside each batch. data.pairs writes the teacher and
student rows of one prompt next to each other, which would make every "swapped"
sequence the other labelled completion of the same prompt. KTO rows are
therefore shuffled with --seed so that no batch holds two rows with the same
prompt, and rows that do not fill the last batch are excluded and recorded.

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
import random
import shutil
import tempfile
from pathlib import Path

from character_pipeline.train.common import (STUDENT, STUDENT_REVISION, load_student, lora_targets, model_receipt,
                                             precision, read_jsonl, seed_everything, torch_versions)


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


def _ids(encoded) -> list[int]:
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def row_tokens(tok, row: dict, sides: tuple[str, ...]) -> int:
    """Longest prompt+completion in tokens, tokenised exactly as TRL does.

    TRL takes the completion to be whatever follows the generation prompt and only warns when the
    generation prompt is not a prefix of the full render (templates that add or strip a thinking
    block do this). Training on such a slice would score the wrong tokens, so refuse.
    """
    prompt = _ids(tok.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False))
    longest = 0
    for side in sides:
        full = _ids(tok.apply_chat_template(row["prompt"] + row[side], tokenize=True, enable_thinking=False))
        if full[:len(prompt)] != prompt or len(full) <= len(prompt):
            raise ValueError(f"row {row.get('id')!r}: the chat template's generation prompt is not a proper prefix "
                             f"of prompt+{side}; TRL would score the wrong tokens")
        longest = max(longest, len(full))
    return longest


def kto_order(rows: list[dict], batch_size: int, seed: int) -> tuple[list[dict], list[dict], int]:
    """(ordered rows, rows left over from the last batch, batches that still share a prompt).

    Seeded shuffle, then swaps so that no batch holds two rows with the same prompt: TRL builds each
    row's KL sequence from another completion in its batch, which must be a mismatched one.
    """
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    keep = len(order) - len(order) % batch_size
    order, rest = order[:keep], order[keep:]
    keys = [json.dumps(r["prompt"], sort_keys=True) for r in rows]

    def clash(position, candidate):
        """Would `candidate` repeat a prompt in the batch around `position` (ignoring that slot)?"""
        start = position - position % batch_size
        return any(keys[order[j]] == keys[candidate] for j in range(start, start + batch_size) if j != position)

    for i in range(len(order)):
        if not clash(i, order[i]):
            continue
        for k in range(len(order)):
            if k // batch_size != i // batch_size and not clash(i, order[k]) and not clash(k, order[i]):
                order[i], order[k] = order[k], order[i]
                break
    unresolved = sum(len({keys[j] for j in order[b:b + batch_size]}) < batch_size
                     for b in range(0, len(order), batch_size))
    return [rows[i] for i in order], [rows[i] for i in rest], unresolved


def compose_adapter(model, sft_adapter: Path, out: Path) -> dict:
    """Write to `out` one adapter for the bare student equal to SFT + the trained delta.

    `model` is the trained PeftModel whose "default" LoRA sits on SFT-merged weights. PEFT's `cat`
    stacks the scaled A and the B matrices, so its update is exactly the sum of the two updates.
    Returns the largest element-wise error of that identity over the LoRA modules checked.
    """
    import torch
    model.load_adapter(str(sft_adapter), adapter_name="sft")
    model.add_weighted_adapter(adapters=["sft", "default"], weights=[1.0, 1.0], adapter_name="composed",
                               combination_type="cat")
    error, checked = 0.0, 0
    with torch.no_grad():
        for module in model.modules():
            if "composed" not in getattr(module, "lora_A", {}):
                continue
            def delta(name):
                if name not in module.lora_A:
                    return 0.0
                return module.scaling[name] * (module.lora_B[name].weight.float() @ module.lora_A[name].weight.float())
            error = max(error, float((delta("composed") - delta("sft") - delta("default")).abs().max()))
            checked += 1
            if checked == 8:
                break
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, selected_adapters=["composed"])
        saved = Path(tmp) / "composed"  # PEFT puts non-default adapters in a subfolder
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(saved if saved.is_dir() else Path(tmp), out)
    rank = model.peft_config["composed"].r
    model.set_adapter("default")
    model.delete_adapter("composed")
    model.delete_adapter("sft")
    return {"rank": rank, "max_abs_error": error, "modules_checked": checked}


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
        if getattr(args, "desirable_weight", None) is not None or getattr(args, "undesirable_weight", None) is not None:
            raise ValueError("--desirable-weight/--undesirable-weight apply to KTO only")
        return common | {"loss_type": loss_type, "beta": args.beta} | (
            {"loss_weights": list(loss_weights)} if loss_weights is not None else {})
    if (loss_type and list(loss_type) != ["sigmoid"]) or loss_weights is not None:
        raise ValueError("--loss-type/--loss-weights apply to DPO only")
    weights = {k: getattr(args, k, None) for k in ("desirable_weight", "undesirable_weight")}
    return common | {"loss_type": "kto", "beta": args.beta} | {k: v for k, v in weights.items() if v is not None}


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
    ap.add_argument("--desirable-weight", type=float, help="KTO weight on desirable rows; default: TRL's 1.0")
    ap.add_argument("--undesirable-weight", type=float, help="KTO weight on undesirable rows; default: TRL's 1.0")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--local-files-only", action="store_true")
    return ap.parse_args(argv)


def merge_check(tok, peft_model, row: dict) -> dict:
    """Logits on one prompt with the SFT adapter attached, before merging: (bare, with adapter)."""
    import torch
    ids = _ids(tok.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False))
    x = torch.tensor([ids], device=next(peft_model.parameters()).device)
    with torch.no_grad():
        attached = peft_model(input_ids=x).logits.float().cpu()
        with peft_model.disable_adapter():
            bare = peft_model(input_ids=x).logits.float().cpu()
    return {"x": x, "attached": attached, "bare": bare}


def main(argv=None):
    args = parse(argv)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    import trl

    rows = read_jsonl(args.data)
    validate_rows(args.kind, rows)
    kw = trainer_kwargs(args.kind, args.out, args)  # flag errors before the model is loaded
    args.out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    tok, base = load_student(args.model, args.revision, args.device, args.local_files_only)
    receipt = model_receipt(tok, base, args.model, args.revision)
    tok.padding_side = "left"
    dtype, _ = precision(args.device)
    kw |= {"bf16": dtype == torch.bfloat16, "fp16": dtype == torch.float16,
           "use_cpu": not args.device.startswith("cuda")}

    # Aineko rule: never let TRL clip a reviewed completion; drop the whole row and record it.
    sides = ("chosen", "rejected") if args.kind == "dpo" else ("completion",)
    kept, excluded = [], []
    for r in rows:
        n = row_tokens(tok, r, sides)
        (kept if n <= args.max_length else excluded).append(r if n <= args.max_length else {"id": r.get("id"), "tokens": n})
    if not kept:
        raise SystemExit("every row exceeds --max-length")
    rows, leftover, shared_prompt_batches = kept, [], 0
    if args.kind == "kto":
        rows, leftover, shared_prompt_batches = kto_order(rows, args.batch_size, args.seed)
        if len({r["label"] for r in rows}) < 2:
            raise SystemExit("after exclusions KTO no longer has both desirable and undesirable rows")

    merge_report = None
    if args.sft_adapter is None:
        # OCT stage 6: DPO from the bare student, which is then also the reference.
        merged = base
    else:
        # Merge the shared SFT adapter: the merged weights ARE the arm's initializer and reference.
        attached = PeftModel.from_pretrained(base, str(args.sft_adapter))
        before = merge_check(tok, attached, rows[0])
        merged = attached.merge_and_unload(safe_merge=True)
        with torch.no_grad():
            after = merged(input_ids=before["x"]).logits.float().cpu()
        # How far the merged (rounded into the student's dtype) weights are from base + adapter, next to
        # how much the adapter does at all. On fp32 the first number is rounding noise.
        merge_report = {"dtype": str(dtype),
                        "max_abs_logit_gap_merged_vs_attached": float((after - before["attached"]).abs().max()),
                        "max_abs_logit_effect_of_sft_adapter": float((before["attached"] - before["bare"]).abs().max())}
    peft_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.lora_dropout,
                             target_modules=lora_targets(merged), bias="none")
    run_config = receipt | {"arm": args.kind, "sft_adapter": str(args.sft_adapter) if args.sft_adapter else "none",
                            "data": str(args.data), "seed": args.seed,
                            "lora": {"r": args.rank, "alpha": args.alpha, "dropout": args.lora_dropout},
                            "trainer": {k: v for k, v in kw.items() if k != "output_dir"},
                            "dtype": str(dtype), "versions": torch_versions()}
    (args.out / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n")

    columns = ("prompt", "chosen", "rejected") if args.kind == "dpo" else ("prompt", "completion", "label")
    dataset = Dataset.from_list([{k: r[k] for k in columns} | {"chat_template_kwargs": {"enable_thinking": False}}
                                 for r in rows])
    Config, Trainer = (trl.DPOConfig, trl.DPOTrainer) if args.kind == "dpo" else (trl.KTOConfig, trl.KTOTrainer)
    trainer = Trainer(model=merged, ref_model=None, peft_config=peft_config, args=Config(**kw),
                      processing_class=tok, train_dataset=dataset)
    if trainer.ref_model is not None or "ref" in getattr(trainer.model, "peft_config", {}):
        raise RuntimeError("reference must be the adapter-disabled initializer")
    if len(trainer.train_dataset) != len(rows):
        raise RuntimeError(f"TRL kept {len(trainer.train_dataset)} of {len(rows)} rows; the exclusion record would be wrong")
    trainable = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    result = trainer.train()
    changed = sum(not torch.equal(p.detach(), trainable[n])
                  for n, p in trainer.model.named_parameters() if n in trainable)
    if changed == 0:
        raise RuntimeError("no adapter tensor changed")
    finite = all(bool(torch.isfinite(p).all()) for n, p in trainer.model.named_parameters() if n in trainable)
    if not finite:
        raise RuntimeError("non-finite adapter weights after training; nothing saved")
    if args.sft_adapter is None:
        trainer.save_model(str(args.out / "adapter"))
        layout = {"adapter": "the trained LoRA; applies to the bare student"}
    else:
        trainer.save_model(str(args.out / "delta_adapter"))
        composed = compose_adapter(trainer.model, args.sft_adapter, args.out / "adapter")
        layout = {"adapter": "SFT + trained delta composed with PEFT 'cat'; applies to the bare student",
                  "delta_adapter": "the trained LoRA alone; applies only on top of the merged SFT weights",
                  "composed": composed}
    tok.save_pretrained(str(args.out / "adapter"))
    reference = ("bare student with the new adapter disabled" if args.sft_adapter is None
                 else "merged SFT weights with the new adapter disabled")
    report = {"arm": args.kind, "sft_adapter": str(args.sft_adapter) if args.sft_adapter else "none",
              "model": args.model, "revision": args.revision, "seed": args.seed,
              "rows": len(rows), "excluded_over_max_length": excluded,
              "reference": reference, "changed_tensors": changed, "saved": layout,
              "lora": {"r": args.rank, "alpha": args.alpha, "dropout": args.lora_dropout},
              "global_step": trainer.state.global_step, "metrics": result.metrics,
              "config": {k: v for k, v in kw.items() if k != "output_dir"}, "versions": torch_versions(),
              "log_history": trainer.state.log_history}
    if merge_report:
        report["merge_check"] = merge_report
    if args.kind == "kto":
        report["kto"] = {"desirable": sum(r["label"] for r in rows), "undesirable": sum(not r["label"] for r in rows),
                         "order": "shuffled with --seed; same-prompt rows kept out of one batch",
                         "batches_sharing_a_prompt": shared_prompt_batches,
                         "excluded_incomplete_last_batch": [r.get("id") for r in leftover]}
    if deviations(args):
        report["deviations"] = deviations(args)
    (args.out / "training_summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
