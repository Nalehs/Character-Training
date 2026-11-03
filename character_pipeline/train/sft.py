"""Stage 6 shared character SFT, and the continued-SFT arm of stage 7.

The loop is Aineko run 1's train.py (aineko-character-training 709b345),
the only completed SFT run, generalised from its hard-coded 0.8B paths:
micro-batch 1, gradient accumulation 4, AdamW lr 1e-4 wd 0.01, 5% warm-up then
cosine, grad clip 1.0, checkpoint every 25 updates, resumable, assistant-only
loss. Loading, LoRA targets, masking and fp16+GradScaler come from the R0
smoke test (see common.py).

Shared SFT:     python -m character_pipeline.train.sft --train runs/clean/train.jsonl \
                    --validation runs/clean/validation.jsonl --out runs/sft/seed0 --seed 0
Continued SFT:  ... --init-adapter runs/sft/seed0/adapter --out runs/continued_sft/seed0 \
                    --lr 1e-5 --max-supervised-tokens <matched to the preference arms>
OCT stage 8:    ... --merge-adapter runs/oct/dpo/seed0/adapter --epochs 1 --lr 5e-5 --rank 64 --alpha 128 \
                    --accumulation 32 --seq-len 3072 --weight-decay 0 --warmup 0.1 --min-lr-rate 0.1 \
                    --adam-beta2 0.98 --lora-dropout 0     (docs/recipes/oct-2025.md)

--merge-adapter folds an adapter into the base weights and trains a FRESH LoRA
on top (OCT trains introspection SFT on the DPO-folded model). --init-adapter
instead keeps training the same adapter. The optimiser/schedule flags default
to Aineko run 1's values, so omitting them reproduces the shared SFT exactly.

Input rows: {"messages": [{"role": "system"|"user"|"assistant", "content": ...}, ...]},
as written by character_pipeline.data.clean.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from character_pipeline.train.common import (STUDENT, STUDENT_REVISION, encode_chat, load_student,
                                             lora_targets, lr_factor, precision, read_jsonl,
                                             supervised_tokens, tokens_in, torch_versions)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--revision", default=STUDENT_REVISION)
    ap.add_argument("--init-adapter", type=Path, help="continue training this adapter (continued-SFT arm)")
    ap.add_argument("--merge-adapter", type=Path,
                    help="merge this adapter into the base weights, then train a fresh LoRA (OCT stage 8)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32, help="Aineko run 1 used 32; R0 used 16")
    ap.add_argument("--accumulation", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (Aineko 0.01; OCT 0)")
    ap.add_argument("--warmup", type=float, default=0.05, help="warm-up fraction of updates (Aineko 0.05; OCT 0.1)")
    ap.add_argument("--min-lr-rate", type=float, default=0.0,
                    help="cosine floor as a fraction of lr (Aineko 0; OCT 0.1)")
    ap.add_argument("--adam-beta2", type=float, default=0.999, help="AdamW beta2 (torch default 0.999; OCT 0.98)")
    ap.add_argument("--lora-dropout", type=float, default=0.05, help="fresh-LoRA dropout (Aineko 0.05; OCT 0)")
    ap.add_argument("--supervise", choices=["all", "last"], default="all",
                    help="all assistant turns (default) or only the final one (OpenRLHF without --multiturn)")
    ap.add_argument("--max-supervised-tokens", type=int, help="stop once this many assistant tokens are trained")
    ap.add_argument("--max-hours", type=float, default=11.5, help="Kaggle sessions end at 12h")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args(argv)
    if args.merge_adapter and args.init_adapter:
        ap.error("--merge-adapter trains a fresh LoRA; it cannot be combined with --init-adapter")
    return args


def schedule(n_samples: int, epochs: int, seed: int) -> list[int]:
    """Aineko run 1 order: each epoch reshuffled with seed+epoch."""
    order = []
    for epoch in range(epochs):
        ids = list(range(n_samples))
        random.Random(seed + epoch).shuffle(ids)
        order += ids
    return order


def main(argv=None):
    args = parse(argv)
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    tok, model = load_student(args.model, args.revision, args.device, args.local_files_only)
    dtype, use_scaler = precision(args.device)
    if args.merge_adapter:
        # OCT stage 8: the folded weights are the new base; a fresh LoRA is trained below.
        model = PeftModel.from_pretrained(model, str(args.merge_adapter)).merge_and_unload(safe_merge=True)

    def encode(path):
        rows, dropped = [], 0
        for row in read_jsonl(path):
            enc = encode_chat(tok, row["messages"], args.seq_len, supervise=args.supervise)
            if enc is None:
                dropped += 1
            else:
                rows.append(enc)
        return rows, dropped

    train, dropped_train = encode(args.train)
    valid, dropped_valid = encode(args.validation)
    if not train or not valid:
        raise SystemExit("need at least one supervised training and validation chat")

    checkpoint = args.out / "checkpoint"
    state_file = checkpoint / "state.pt"
    previous = None
    if args.resume and state_file.exists():
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=True)
        previous = torch.load(state_file, map_location="cpu", weights_only=False)
    elif args.init_adapter:
        # Continued SFT: keep training the shared adapter; never start a fresh one.
        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.lora_dropout,
                                                 target_modules=lora_targets(model), bias="none"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # adapters and optimiser in fp32 (R0, Aineko)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, args.adam_beta2), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", init_scale=256) if use_scaler else None

    order = schedule(len(train), args.epochs, args.seed)
    acc = args.accumulation
    total_updates = -(-len(order) // acc)
    index, updates, losses, trained_tokens = 0, 0, [], 0
    if previous:
        opt.load_state_dict(previous["optimizer"])
        if scaler and previous.get("scaler"):
            scaler.load_state_dict(previous["scaler"])
        index, updates, losses = previous["index"], previous["updates"], previous["losses"]
        trained_tokens = previous.get("trained_tokens", 0)

    config = {"model": args.model, "revision": args.revision, "init_adapter": str(args.init_adapter or ""),
              "merge_adapter": str(args.merge_adapter or ""), "weight_decay": args.weight_decay,
              "warmup": args.warmup, "min_lr_rate": args.min_lr_rate, "adam_betas": [0.9, args.adam_beta2],
              "lora_dropout": args.lora_dropout, "supervise": args.supervise,
              "train_chats": len(train), "dropped_over_seq_len": dropped_train,
              "validation_chats": len(valid), "validation_dropped": dropped_valid,
              "train_tokens": tokens_in(train), "train_supervised_tokens": supervised_tokens(train),
              "epochs": args.epochs, "seq_len": args.seq_len, "lr": args.lr, "rank": args.rank,
              "alpha": args.alpha, "accumulation": acc, "seed": args.seed, "dtype": str(dtype),
              "grad_scaler": bool(scaler), "trainable_parameters": sum(p.numel() for p in params),
              "versions": torch_versions()}
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    def save(next_index):
        model.save_pretrained(str(checkpoint))
        torch.save({"optimizer": opt.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                    "index": next_index, "updates": updates, "losses": losses,
                    "trained_tokens": trained_tokens}, state_file)

    def loss_of(sample):
        x = torch.tensor([sample["input_ids"]], device=args.device)
        y = torch.tensor([sample["labels"]], device=args.device)
        with torch.autocast(args.device.split(":")[0], dtype=dtype, enabled=dtype != torch.float32):
            return model(input_ids=x, labels=y).loss

    start, stopped = time.time(), None
    opt.zero_grad(set_to_none=True)
    i = index - 1
    for i in range(index, len(order)):
        sample = train[order[i]]
        divisor = min(acc, len(order) - (i // acc) * acc)
        loss = loss_of(sample)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        (scaler.scale(loss / divisor) if scaler else loss / divisor).backward()
        losses.append(float(loss.detach()))
        trained_tokens += sum(v != -100 for v in sample["labels"])
        if (i + 1) % acc == 0 or i + 1 == len(order):
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for group in opt.param_groups:
                group["lr"] = args.lr * lr_factor(updates, total_updates, args.warmup, args.min_lr_rate)
            (scaler.step(opt), scaler.update()) if scaler else opt.step()
            opt.zero_grad(set_to_none=True)
            updates += 1
            status = {"microsteps": i + 1, "of": len(order), "updates": updates,
                      "recent_loss": sum(losses[-20:]) / len(losses[-20:]),
                      "supervised_tokens": trained_tokens, "elapsed_s": time.time() - start}
            (args.out / "status.json").write_text(json.dumps(status) + "\n")
            print(json.dumps(status), flush=True)
            if updates % 25 == 0:
                save(i + 1)
            if args.max_supervised_tokens and trained_tokens >= args.max_supervised_tokens:
                stopped = "token budget"
                break
            if time.time() - start > args.max_hours * 3600:
                stopped = "runtime bound"
                break
    save(i + 1)
    model.save_pretrained(str(args.out / "adapter"))
    tok.save_pretrained(str(args.out / "adapter"))

    model.eval()
    with torch.no_grad():
        val = [float(loss_of(s)) for s in valid]
    summary = {"completed": stopped is None, "stopped_by": stopped, "processed_microsteps": i + 1,
               "requested_microsteps": len(order), "updates": updates, "trained_supervised_tokens": trained_tokens,
               "training_seconds": time.time() - start,
               "first_20_loss": sum(losses[:20]) / max(1, len(losses[:20])),
               "last_20_loss": sum(losses[-20:]) / max(1, len(losses[-20:])),
               "validation_loss": sum(val) / len(val), "validation_chats": len(val)}
    (args.out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("summary", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
