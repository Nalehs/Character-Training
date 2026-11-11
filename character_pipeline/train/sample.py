"""Sample the student's own replies: the rejected side of DPO pairs.

For every assistant turn in the clean training chats, give the SFT student the
identical prefix and keep what it says. This follows Aineko OCT
(oct/run_stage.py `sample`): rejected = the student's genuine sample, never a
teacher-written bad example. Settings are Aineko's: temperature 0.8,
top-p 0.95, thinking off. A 512-token cap replaces Aineko's 192 so that
character-length replies are not cut.

    python -m character_pipeline.train.sample --adapter runs/<id>/sft/seed0/adapter \
        --chats runs/<id>/clean/train.jsonl --out runs/<id>/samples/seed0.jsonl

Output rows: {"id": "<chat id>:<assistant index>", "completion": "...", "finish": "eos"|"length"}

Rows are written to <out>.partial and renamed to <out> only when every prompt has
been sampled, so an interrupted run never leaves a short file that looks finished.
<out>.meta.json records the model, revision, sampling settings, seed and versions.
One seed is set at the start, so a sample depends on the seed and on its position
in the file; re-running with the same inputs reproduces the file on the same
hardware and library versions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from character_pipeline.data.pairs import prompts_and_chosen
from character_pipeline.train.common import (END_OF_TURN, STUDENT, STUDENT_REVISION, load_student, model_receipt,
                                             read_jsonl, seed_everything, torch_versions)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", type=Path, help="SFT adapter; omit to sample the base model")
    ap.add_argument("--chats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--revision", default=STUDENT_REVISION)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--local-files-only", action="store_true")
    return ap.parse_args(argv)


def load_policy(model_id, revision, adapter, device, local_files_only=False):
    """Base + optional adapter, ready for generation."""
    from peft import PeftModel
    tok, model = load_student(model_id, revision, device, local_files_only)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    model.config.use_cache = True
    return tok, model


def generate(tok, model, messages, max_new_tokens, temperature, top_p, device, **sampling):
    """One sampled reply. Extra `sampling` kwargs (e.g. top_k, repetition_penalty) go to
    model.generate unchanged; none are passed by this module's own sampling."""
    import torch
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    x = torch.tensor([ids], device=device)
    eos = tok.convert_tokens_to_ids(END_OF_TURN)
    if eos is None or eos == tok.unk_token_id:
        raise ValueError(f"the tokenizer has no {END_OF_TURN} token, so generation would have no stop token")
    with torch.inference_mode():
        # Greedy when temperature is 0; top_p is then meaningless and is not passed.
        nucleus = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else {"do_sample": False}
        y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
                           eos_token_id=eos, pad_token_id=tok.pad_token_id, **nucleus, **sampling)
    new = y[0, x.shape[1]:].tolist()
    finish = "eos" if new and new[-1] == eos else "length"
    return tok.decode(new, skip_special_tokens=True).strip(), finish


def main(argv=None):
    args = parse(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite raw samples")
    seed_everything(args.seed)
    tok, model = load_policy(args.model, args.revision, args.adapter, args.device, args.local_files_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    counts = {"eos": 0, "length": 0}
    with partial.open("w") as f:  # a leftover .partial from an interrupted run is simply redone
        for n, chat in enumerate(read_jsonl(args.chats)):
            for pid, prompt, _ in prompts_and_chosen(chat, chat.get("id", str(n))):
                text, finish = generate(tok, model, prompt, args.max_new_tokens, args.temperature,
                                        args.top_p, args.device)
                counts[finish] += 1
                f.write(json.dumps({"id": pid, "completion": text, "finish": finish}, ensure_ascii=False) + "\n")
    meta = model_receipt(tok, model, args.model, args.revision) | {
        "adapter": str(args.adapter or ""), "chats": str(args.chats), "seed": args.seed,
        "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
        "samples": sum(counts.values()), "finish": counts, "versions": torch_versions()}
    args.out.with_name(args.out.name + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    partial.rename(args.out)  # last, so the samples file only exists when complete


if __name__ == "__main__":
    main()
