"""Shared training pieces: student loading, assistant-only labels, LR schedule.

Two sources, both from runs that actually completed:

- Loading and masking follow the R0 Kaggle smoke test on the pinned
  Qwen3.5-4B-Base (charbench report/friendship1_kaggle/r0/kernel.py; results in
  that folder's RESULTS.md): the official chat template with thinking disabled,
  the vision tower dropped after load, rank-16 LoRA on every Linear in the
  language layers, fp16 weights with fp32 adapters on a T4. R0 checked the mask
  on one fixture (14 supervised tokens: assistant body and <|im_end|>).
- The optimiser loop (micro-batch 1, accumulation 4, 5% warm-up then cosine,
  clip 1.0, lr 1e-4) is Aineko run 1's train.py, the one completed SFT run
  (evidence/aineko-run1-oct).

Torch, transformers and peft are imported inside functions, so the pure helpers
here are testable without them.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

STUDENT = "Qwen/Qwen3.5-4B-Base"
STUDENT_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def lr_factor(update: int, total_updates: int, warmup_fraction: float = 0.05,
              min_lr_rate: float = 0.0) -> float:
    """Aineko run 1 schedule: linear warm-up over 5% of updates, then cosine decay to 0.

    `min_lr_rate` > 0 decays to that fraction of the peak instead of 0, the shape
    of transformers' `cosine_with_min_lr` used by OCT's OpenRLHF fork (floor 0.1).
    At the default 0 the value is exactly Aineko's.
    """
    warm = max(1, int(warmup_fraction * total_updates))
    ramp = min(1.0, (update + 1) / warm)
    progress = max(0, update - warm) / max(1, total_updates - warm)
    cosine = 0.5 + 0.5 * math.cos(math.pi * progress)
    if min_lr_rate:
        cosine = min_lr_rate + (1.0 - min_lr_rate) * cosine
    return ramp * cosine


def assistant_char_spans(render, messages: list[dict]) -> list[tuple[int, int]]:
    """Character spans of each assistant body plus its end marker in the full render.

    `render(messages, add_generation_prompt)` returns template text. Every
    assistant turn's prompt render must be a prefix of the full render; Qwen
    templates that rewrite earlier turns (e.g. stripping thinking) break this,
    and we refuse rather than mis-mask.
    """
    full = render(messages, False)
    spans = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = render(messages[:i], True)
        if not full.startswith(prefix):
            raise ValueError(f"template is not prefix-stable at message {i}; refusing to guess the mask")
        end = full.index("<|im_end|>", len(prefix)) + len("<|im_end|>")
        spans.append((len(prefix), end))
    return spans


def labels_from_offsets(input_ids: list[int], offsets: list[tuple[int, int]],
                        spans: list[tuple[int, int]]) -> list[int]:
    """Supervise a token only if it lies wholly inside an assistant span (R0 rule)."""
    out = []
    for token, (a, b) in zip(input_ids, offsets):
        inside = b > a and any(a >= s and b <= e for s, e in spans)
        out.append(token if inside else -100)
    return out


def encode_chat(tok, messages: list[dict], max_len: int, supervise: str = "all") -> dict | None:
    """Token ids and labels for one conversation; None if nothing is supervised.

    Chats longer than max_len are dropped (the caller counts them), never
    truncated: truncation could cut an assistant body and change the target.

    supervise="all" (default) trains every assistant turn. "last" trains only the
    final assistant message, as OCT's OpenRLHF SFT does without --multiturn
    (fork openrlhf/datasets/sft_dataset.py:24-25, 216-218).
    """
    if supervise not in ("all", "last"):
        raise ValueError(f"supervise must be 'all' or 'last', not {supervise!r}")
    def render(msgs, gen):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen,
                                       enable_thinking=False)
    full = render(messages, False)
    spans = assistant_char_spans(render, messages)
    if supervise == "last":
        spans = spans[-1:]
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    if len(enc["input_ids"]) > max_len:
        return None
    labels = labels_from_offsets(enc["input_ids"], enc["offset_mapping"], spans)
    if all(v == -100 for v in labels):
        return None
    return {"input_ids": enc["input_ids"], "labels": labels}


def lora_targets(model) -> list[str]:
    """Every Linear in the language layers (R0: covers hybrid linear attention and MLP)."""
    import torch
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and ".layers." in n and "visual" not in n]
    if not names:
        raise RuntimeError("no Linear modules found in language layers")
    return names


def precision(device: str):
    """fp16 weights + GradScaler on pre-Ampere GPUs (R0 on T4), bf16 on Ampere+, fp32 on CPU."""
    import torch
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        return torch.float16, True
    return torch.float32, False


def load_student(model_id: str = STUDENT, revision: str | None = STUDENT_REVISION,
                 device: str = "cuda:0", local_files_only: bool = False):
    """Tokenizer and text-only model at a pinned revision."""
    import torch
    import transformers
    dtype, _ = precision(device)
    tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                     local_files_only=local_files_only)
    config_file = Path(model_id) / "tokenizer_config.json"
    if config_file.is_file() and json.loads(config_file.read_text()).get("tokenizer_class") == "PreTrainedTokenizerFast":
        # A local test tokenizer that AutoTokenizer would otherwise swap for the model family's class.
        tok = transformers.PreTrainedTokenizerFast.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model_type = transformers.AutoConfig.from_pretrained(
        model_id, revision=revision, local_files_only=local_files_only).model_type
    cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if model_type == "qwen3_5" and cls is not None:
        # R0 path: the 4B Base checkpoint is a vision-language model; keep the language part.
        model = cls.from_pretrained(model_id, revision=revision, dtype=dtype, device_map={"": device},
                                    attn_implementation="eager", local_files_only=local_files_only)
        if hasattr(model, "model") and hasattr(model.model, "visual"):
            del model.model.visual
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype, device_map={"": device},
            local_files_only=local_files_only)
    model.config.use_cache = False
    return tok, model


def tokens_in(rows: list[dict]) -> int:
    return sum(len(r["input_ids"]) for r in rows)


def supervised_tokens(rows: list[dict]) -> int:
    return sum(sum(v != -100 for v in r["labels"]) for r in rows)


def torch_versions() -> dict:
    import importlib.metadata as md
    return {k: md.version(k) for k in ("torch", "transformers", "peft", "accelerate")
            if _installed(k)} | ({"trl": md.version("trl")} if _installed("trl") else {})


def _installed(name: str) -> bool:
    import importlib.metadata as md
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False
