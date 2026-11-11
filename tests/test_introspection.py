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


class Resume(unittest.TestCase):
    """An interrupted generation continues where it stopped and gives the uninterrupted result."""

    @staticmethod
    def seeded(messages, seed):
        return f"text for {seed}", "eos"

    def dying(self, after):
        calls = []
        def gen(messages, seed):
            if len(calls) == after:
                raise RuntimeError("out of memory")
            calls.append(seed)
            return self.seeded(messages, seed)
        return gen

    def test_reflection_and_interaction_resume(self):
        import tempfile
        from pathlib import Path
        from character_pipeline.data.io import partial_path, read_jsonl
        cases = {
            "self_reflection.jsonl": lambda gen: lambda done, sink: I.reflect(gen, "Qwen", "1: t", 2, 0, done, sink),
            "self_interaction-leading.jsonl": lambda gen: lambda done, sink: I.interact(
                gen, "Qwen", "1: t", 4, 3, "leading", 0, done, sink),
        }
        for name, make in cases.items():
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / name
                whole = make(self.seeded)({}, None)
                with self.assertRaises(RuntimeError):
                    I.stored(path, make(self.dying(7)))
                self.assertFalse(path.exists())
                saved = len(read_jsonl(partial_path(path)))
                self.assertTrue(0 < saved < len(whole))
                log = []
                def counting(messages, seed, log=log):
                    log.append(seed)
                    return self.seeded(messages, seed)
                self.assertEqual(I.stored(path, make(counting)), whole)
                self.assertLess(len(log), sum(len(r.get("seeds", [1])) for r in whole))  # finished rows not redone
                # a finished file is read back without generating anything
                def refuse(messages, seed):
                    raise AssertionError("generated for a finished file")
                self.assertEqual(I.stored(path, make(refuse)), whole)

    def test_saved_rows_from_another_seed_are_refused(self):
        rows = I.reflect(self.seeded, "Qwen", "1: t", 1, seed=0)
        with self.assertRaises(ValueError):
            I.reflect(self.seeded, "Qwen", "1: t", 1, seed=1, done={r["id"]: r for r in rows})

    def test_write_once_refuses_other_content(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sft.jsonl"
            I.write_once(p, "a\n")
            I.write_once(p, "a\n")
            with self.assertRaises(SystemExit):
                I.write_once(p, "b\n")
            self.assertEqual(p.read_text(), "a\n")

    def test_bad_validation_n_fails_before_any_model_is_loaded(self):
        for n in ("-1", "30"):
            with self.assertRaises(SystemExit):
                I.main(["--adapter", "none", "--constitution", "none", "--out", "none", "--reflection-n", "1",
                        "--interaction-n", "10", "--validation-n", n])


if __name__ == "__main__":
    unittest.main()
