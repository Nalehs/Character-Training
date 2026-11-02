import unittest
from character_pipeline.data.pairs import build
from character_pipeline.train.preference import validate_rows

CHAT = {"id": "c1", "messages": [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "t0"},
                                 {"role": "user", "content": "q1"}, {"role": "assistant", "content": "t1"}]}


class Pairs(unittest.TestCase):
    def test_dpo_uses_identical_prefix_and_student_rejected(self):
        dpo, kto, skipped = build([CHAT], {"c1:0": "s0", "c1:1": "t1"})
        self.assertEqual(len(dpo), 1)
        self.assertEqual(dpo[0]["prompt"], CHAT["messages"][:1])
        self.assertEqual(dpo[0]["rejected"][0]["content"], "s0")
        self.assertEqual(skipped, [{"id": "c1:1", "reason": "student matches teacher"}])
        validate_rows("dpo", dpo)

    def test_kto_only_from_independent_labels(self):
        _, kto, _ = build([CHAT], {"c1:0": "s0"}, {})
        self.assertEqual(kto, [])          # no labels -> no KTO rows, never inferred from DPO sides
        _, kto, _ = build([CHAT], {"c1:0": "s0"}, {("c1:0", "teacher"): True, ("c1:0", "student"): False})
        self.assertEqual([r["label"] for r in kto], [True, False])
        validate_rows("kto", kto)

    def test_missing_sample_reported(self):
        _, _, skipped = build([CHAT], {})
        self.assertEqual(len(skipped), 2)


if __name__ == "__main__":
    unittest.main()
