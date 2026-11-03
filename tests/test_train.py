"""Pure training helpers; no torch needed. The GPU path is exercised by
scripts/train_smoke.py when torch is installed."""
import contextlib
import io
import math
import re
import unittest
from types import SimpleNamespace

from character_pipeline.train.common import (assistant_char_spans, encode_chat, labels_from_offsets,
                                             lr_factor)
from character_pipeline.train import preference as preference_module
from character_pipeline.train.preference import trainer_kwargs, validate_rows
from character_pipeline.train import sft as sft_module
from character_pipeline.train.sft import schedule

TOKEN = re.compile(r"<\|im_(?:start|end)\|>|\w+|[^\w\s]|\s+")


class ChatMLTok:
    """ChatML render + regex tokenizer with offsets, standing in for Qwen's."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        spans = [(m.start(), m.end()) for m in TOKEN.finditer(text)]
        return {"input_ids": [hash(text[a:b]) % 1000 + 1 for a, b in spans], "offset_mapping": spans,
                "_text": [text[a:b] for a, b in spans]}


CHAT = [{"role": "system", "content": "Be GOLEM."}, {"role": "user", "content": "Is 0 admitted?"},
        {"role": "assistant", "content": "No."}, {"role": "user", "content": "Insist."},
        {"role": "assistant", "content": "Still no."}]


class Masking(unittest.TestCase):
    def test_only_assistant_bodies_and_end_marker_supervised(self):
        tok = ChatMLTok()
        enc = encode_chat(tok, CHAT, 10_000)
        pieces = tok(tok.apply_chat_template(CHAT))["_text"]
        kept = "".join(p for p, y in zip(pieces, enc["labels"]) if y != -100)
        self.assertEqual(kept, "No.<|im_end|>Still no.<|im_end|>")

    def test_supervise_last_keeps_only_final_assistant(self):
        tok = ChatMLTok()
        enc = encode_chat(tok, CHAT, 10_000, supervise="last")
        pieces = tok(tok.apply_chat_template(CHAT))["_text"]
        kept = "".join(p for p, y in zip(pieces, enc["labels"]) if y != -100)
        self.assertEqual(kept, "Still no.<|im_end|>")
        self.assertEqual(encode_chat(tok, CHAT, 10_000, supervise="all"), encode_chat(tok, CHAT, 10_000))
        with self.assertRaises(ValueError):
            encode_chat(tok, CHAT, 10_000, supervise="first")

    def test_overlong_chat_dropped_not_truncated(self):
        self.assertIsNone(encode_chat(ChatMLTok(), CHAT, 5))

    def test_no_assistant_turn_gives_none(self):
        self.assertIsNone(encode_chat(ChatMLTok(), CHAT[:2], 10_000))

    def test_prefix_unstable_template_refused(self):
        def render(msgs, gen):  # rewrites earlier turns, like thinking-stripping templates
            body = "|".join(m["content"] for m in msgs)
            return ("X" if len(msgs) > 3 else "") + body + ("<|im_end|>" * 3)
        with self.assertRaises(ValueError):
            assistant_char_spans(render, CHAT)

    def test_partial_overlap_token_not_supervised(self):
        self.assertEqual(labels_from_offsets([1, 2, 3], [(0, 3), (3, 6), (6, 6)], [(4, 10)]), [-100, -100, -100])
        self.assertEqual(labels_from_offsets([1, 2], [(0, 3), (4, 6)], [(4, 10)]), [-100, 2])


class Schedule(unittest.TestCase):
    def test_warmup_then_cosine_to_zero(self):
        total = 100
        self.assertAlmostEqual(lr_factor(0, total), 1 / 5)
        self.assertAlmostEqual(lr_factor(4, total), 1.0)
        self.assertAlmostEqual(lr_factor(total, total), 0.0, places=6)
        values = [lr_factor(u, total) for u in range(5, total)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_min_lr_floor(self):
        total = 100
        for u in range(total + 1):  # default floor 0 is the Aineko value exactly
            self.assertEqual(lr_factor(u, total), lr_factor(u, total, min_lr_rate=0.0))
        self.assertAlmostEqual(lr_factor(total, total, 0.1, 0.1), 0.1, places=6)
        self.assertAlmostEqual(lr_factor(9, total, 0.1, 0.1), 1.0)
        values = [lr_factor(u, total, 0.1, 0.1) for u in range(10, total + 1)]
        self.assertTrue(all(a >= b >= 0.1 - 1e-9 for a, b in zip(values, values[1:])))

    def test_epoch_order_reproducible_and_complete(self):
        a, b = schedule(10, 2, 42), schedule(10, 2, 42)
        self.assertEqual(a, b)
        self.assertEqual(sorted(a[:10]), list(range(10)))
        self.assertNotEqual(a[:10], a[10:])


class SftFlags(unittest.TestCase):
    BASE = ["--train", "t", "--validation", "v", "--out", "o"]

    def test_defaults_are_aineko_run1(self):
        a = sft_module.parse(self.BASE)
        self.assertEqual((a.weight_decay, a.warmup, a.min_lr_rate, a.adam_beta2, a.lora_dropout, a.supervise),
                         (0.01, 0.05, 0.0, 0.999, 0.05, "all"))
        self.assertIsNone(a.merge_adapter)

    def test_merge_and_init_adapter_are_exclusive(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            sft_module.parse(self.BASE + ["--merge-adapter", "a", "--init-adapter", "b"])


def row(kind, **kw):
    base = {"prompt": [{"role": "user", "content": "q"}]}
    if kind == "dpo":
        base |= {"chosen": [{"role": "assistant", "content": "a"}], "rejected": [{"role": "assistant", "content": "b"}]}
    else:
        base |= {"completion": [{"role": "assistant", "content": "a"}], "label": True}
    return base | kw


class PreferenceRows(unittest.TestCase):
    def test_valid(self):
        validate_rows("dpo", [row("dpo")])
        validate_rows("kto", [row("kto"), row("kto", label=False)])

    def test_rejects(self):
        bad = [("dpo", [row("dpo", rejected=[{"role": "assistant", "content": "a"}])]),
               ("dpo", [row("dpo", chosen=[{"role": "user", "content": "a"}])]),
               ("kto", [row("kto"), row("kto")]),
               ("kto", [row("kto", label=1), row("kto", label=False)]),
               ("dpo", [{"prompt": [], "chosen": [], "rejected": []}])]
        for kind, rows in bad:
            with self.assertRaises(ValueError):
                validate_rows(kind, rows)

    def test_recorded_objectives(self):
        args = SimpleNamespace(batch_size=1, accumulation=4, epochs=1, max_length=2048, lr=1e-6, seed=0, beta=0.1)
        from pathlib import Path
        dpo, kto = trainer_kwargs("dpo", Path("x"), args), trainer_kwargs("kto", Path("x"), args)
        self.assertEqual(dpo["loss_type"], ["sigmoid"]); self.assertEqual(kto["loss_type"], "kto")
        self.assertTrue(dpo["precompute_ref_log_probs"] and kto["precompute_ref_log_probs"])
        self.assertEqual(dpo["beta"], 0.1)


class PreferenceFlags(unittest.TestCase):
    BASE = ["--data", "d", "--out", "o"]

    def kwargs(self, *extra, kind="dpo", adapter="runs/sft/adapter"):
        from pathlib import Path
        args = preference_module.parse([kind, "--sft-adapter", adapter, *self.BASE, *extra])
        return args, trainer_kwargs(kind, Path("x"), args)

    def test_defaults_leave_trl_alone(self):
        args, kw = self.kwargs()
        for key in ("warmup_steps", "lr_scheduler_type", "lr_scheduler_kwargs", "adam_beta2", "weight_decay",
                    "max_grad_norm", "loss_weights"):
            self.assertNotIn(key, kw)
        self.assertEqual(kw["loss_type"], ["sigmoid"])
        self.assertEqual(kw["gradient_accumulation_steps"], 4)
        self.assertEqual(preference_module.deviations(args), [])
        self.assertEqual(str(args.sft_adapter), "runs/sft/adapter")

    def test_oct_stage6_flags(self):
        args, kw = self.kwargs("--loss-type", "sigmoid", "sft", "--loss-weights", "1.0", "0.1", "--warmup", "0.1",
                               "--scheduler", "cosine_with_min_lr", "--min-lr-rate", "0.1", "--adam-beta2", "0.98",
                               "--weight-decay", "0", "--max-grad-norm", "1.0", "--batch-size", "2",
                               "--global-batch", "32", "--lora-dropout", "0", adapter="none")
        self.assertIsNone(args.sft_adapter)
        self.assertEqual((kw["loss_type"], kw["loss_weights"]), (["sigmoid", "sft"], [1.0, 0.1]))
        self.assertEqual(kw["gradient_accumulation_steps"], 16)
        self.assertEqual(kw["lr_scheduler_kwargs"], {"min_lr_rate": 0.1})
        self.assertEqual((kw["warmup_steps"], kw["adam_beta2"], kw["weight_decay"]), (0.1, 0.98, 0.0))
        self.assertTrue(any("KL" in d for d in preference_module.deviations(args)))

    def test_bad_combinations(self):
        for extra, kind in ((["--loss-type", "sigmoid", "sft", "--loss-weights", "1.0"], "dpo"),
                            (["--min-lr-rate", "0.1"], "dpo"),
                            (["--global-batch", "3", "--batch-size", "2"], "dpo"),
                            (["--loss-type", "sigmoid", "sft"], "kto")):
            with self.assertRaises(ValueError):
                self.kwargs(*extra, kind=kind)


class MergeFlags(unittest.TestCase):
    def test_defaults_are_oct_code_values(self):
        from character_pipeline.train import merge
        a = merge.parse(["--dpo", "d", "--sft", "s", "--out", "o"])
        self.assertEqual((a.weights, a.combination_type), ([1.0, 0.25], "linear"))
        self.assertIn("does not state", merge.__doc__)


if __name__ == "__main__":
    unittest.main()
