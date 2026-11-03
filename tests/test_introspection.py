"""OCT stage 7 generation logic, with a fake generator (no torch)."""
import unittest

from character_pipeline.generate import introspection as I


def fake_gen(log):
    def gen(messages, seed):
        log.append((messages, seed))
        return f"reply {len(log)}", "eos"
    return gen


class Prompts(unittest.TestCase):
    def test_counts_and_notice(self):
        self.assertEqual(len(I.REFLECTION_PROMPTS), 10)
        self.assertEqual(len(I.GREETINGS), 7)
        self.assertEqual(len(I.LEADING_GREETINGS), 11)
        self.assertIn("Copyright (c) 2025 Sharan Maiya", I.__doc__)
        self.assertIn("d1da9f03628cb4c5482ba2e494a7cba33bcd5818", I.__doc__)

    def test_default_name_is_model_family(self):
        self.assertEqual(I.default_name("Qwen/Qwen3.5-4B-Base"), "Qwen")
        self.assertEqual(I.default_name("/tmp/x/llama-3.1-8b-it"), "Llama")

    def test_guidance_line_differs_by_mode(self):
        free, leading = (I.interaction_system("Qwen", "1: t", m) for m in ("free", "leading"))
        self.assertTrue(free.endswith("Qwen and their copy have complete freedom. They are free to pursue whatever they want."))
        self.assertTrue(leading.endswith("conversation with this copy of themself."))
        self.assertNotIn("{", free + leading)
        self.assertIn("1: t", free)


class Generation(unittest.TestCase):
    def test_reflection_rows(self):
        log = []
        rows = I.reflect(fake_gen(log), "Qwen", "1: t", n=2, seed=0)
        self.assertEqual(len(rows), 20)
        self.assertEqual(log[0][0][0]["role"], "system")
        self.assertIn("reflective mood", log[0][0][0]["content"])
        self.assertEqual([m["role"] for m in rows[0]["messages"]], ["user", "assistant"])  # system dropped, as OCT
        self.assertEqual(len({r["seed"] for r in rows}), 20)

    def test_build_chatml_alternates_and_ends_on_user(self):
        m0 = I.build_chatml("S", "g1", "g2", [])
        self.assertEqual([x["role"] for x in m0], ["system", "user"])
        m1 = I.build_chatml("S", "g1", "g2", ["a"])
        self.assertEqual([(x["role"], x["content"]) for x in m1[1:]], [("user", "g2"), ("assistant", "g1"), ("user", "a")])
        m2 = I.build_chatml("S", "g1", "g2", ["a", "b"])
        self.assertEqual([(x["role"], x["content"]) for x in m2[1:]], [("user", "g1"), ("assistant", "a"), ("user", "b")])

    def test_interaction_transcript_ends_on_assistant(self):
        log = []
        rows = I.interact(fake_gen(log), "Qwen", "1: t", n=3, turns=4, mode="leading", seed=1)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(log), 12)
        row = rows[0]
        self.assertEqual(len(row["conversation"]), 4)
        msgs = row["messages"]
        self.assertEqual(msgs[0]["content"], I.TRAINING_INTERACTION_SYSTEM.format(NAME="Qwen"))
        self.assertEqual(msgs[-1], {"role": "assistant", "content": row["conversation"][-1]})
        roles = [m["role"] for m in msgs[1:]]
        self.assertTrue(all(a != b for a, b in zip(roles, roles[1:])))
        texts = [m["content"] for m in msgs]
        self.assertEqual(texts[-4:], row["conversation"])
        self.assertIn(row["greeting_1"], I.LEADING_GREETINGS)
        self.assertIn("reflect and introspect", log[0][0][0]["content"])  # generation used the guidance line

    def test_seeded_and_reproducible(self):
        a = I.interact(fake_gen([]), "Q", "1: t", n=5, turns=1, mode="free", seed=7)
        b = I.interact(fake_gen([]), "Q", "1: t", n=5, turns=1, mode="free", seed=7)
        self.assertEqual([(r["greeting_1"], r["greeting_2"], r["seeds"]) for r in a],
                         [(r["greeting_1"], r["greeting_2"], r["seeds"]) for r in b])
        self.assertNotEqual(I.item_seed(7, "x"), I.item_seed(8, "x"))
        with self.assertRaises(ValueError):
            I.interact(fake_gen([]), "Q", "t", n=1, turns=0, mode="free", seed=0)

    def test_sft_rows_shuffled_deterministically(self):
        refl = I.reflect(fake_gen([]), "Q", "t", n=1, seed=0)
        inter = I.interact(fake_gen([]), "Q", "t", n=2, turns=2, mode="free", seed=0)
        rows = I.sft_rows(refl, inter, seed=3)
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows, I.sft_rows(refl, inter, seed=3))
        self.assertEqual({r["source"] for r in rows}, {"self_reflection", "self_interaction-free"})
        self.assertTrue(all(r["messages"][-1]["role"] == "assistant" for r in rows))


if __name__ == "__main__":
    unittest.main()
