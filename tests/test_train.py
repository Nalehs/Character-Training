"""Pure training helpers; no torch needed. The GPU path is exercised by
scripts/train_smoke.py when torch is installed."""
import math
import re
import unittest
from types import SimpleNamespace

from character_pipeline.train.common import (assistant_char_spans, encode_chat, labels_from_offsets,
                                             lr_factor)
from character_pipeline.train.preference import trainer_kwargs, validate_rows
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

    def test_epoch_order_reproducible_and_complete(self):
        a, b = schedule(10, 2, 42), schedule(10, 2, 42)
        self.assertEqual(a, b)
        self.assertEqual(sorted(a[:10]), list(range(10)))
        self.assertNotEqual(a[:10], a[10:])


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


if __name__ == "__main__":
    unittest.main()
