import json, tempfile, unittest
from pathlib import Path
from character_pipeline.data import pairs
from character_pipeline.data.pairs import build, read_labels, read_samples
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

    def test_sides_are_not_swapped_and_continued_sft_uses_chosen(self):
        dpo, _, _ = build([CHAT], {"c1:0": "s0", "c1:1": "s1"})
        self.assertEqual([(r["chosen"][0]["content"], r["rejected"][0]["content"]) for r in dpo],
                         [("t0", "s0"), ("t1", "s1")])
        self.assertEqual(dpo[1]["prompt"], CHAT["messages"][:3])   # teacher-forced prefix, shared by both sides

    def test_empty_student_sample_makes_no_pair(self):
        dpo, _, skipped = build([CHAT], {"c1:0": "  ", "c1:1": "s1"})
        self.assertEqual([r["id"] for r in dpo], ["c1:1"])
        self.assertEqual(skipped, [{"id": "c1:0", "reason": "empty student sample"}])

    def test_label_must_be_a_json_boolean(self):
        for bad in ("false", 0, None):
            with self.assertRaises(ValueError):
                read_labels([{"id": "c1:0", "source": "student", "label": bad}], [CHAT], {"c1:0": "s0"})
        with self.assertRaises(ValueError):
            read_labels([{"id": "c1:0", "source": "student", "label": True},
                         {"id": "c1:0", "source": "student", "label": False}], [CHAT], {"c1:0": "s0"})

    def test_label_given_for_another_text_is_not_used(self):
        rows = [{"id": "c1:0", "source": "student", "label": False, "completion": "a sample from another seed"},
                {"id": "c1:0", "source": "teacher", "label": True, "completion": "t0"},
                {"id": "c9:0", "source": "teacher", "label": True}]
        labels, problems = read_labels(rows, [CHAT], {"c1:0": "s0"})
        self.assertEqual(labels, {("c1:0", "teacher"): True})
        self.assertEqual([p["problem"] for p in problems], ["label was given for another text", "no such completion"])

    def test_conflicting_duplicate_samples_raise(self):
        self.assertEqual(read_samples([{"id": "a", "completion": "x"}, {"id": "a", "completion": "x"}]), {"a": "x"})
        with self.assertRaises(ValueError):
            read_samples([{"id": "a", "completion": "x"}, {"id": "a", "completion": "y"}])

    def test_rerun_removes_stale_files_and_reports_balance(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            dump = lambda name, rows: (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            dump("chats.jsonl", [CHAT])
            dump("samples.jsonl", [{"id": "c1:0", "completion": "s0"}, {"id": "c1:1", "completion": "s1"}])
            dump("labels.jsonl", [{"id": "c1:0", "source": "teacher", "label": True},
                                  {"id": "c1:0", "source": "student", "label": False},
                                  {"id": "c1:1", "source": "teacher", "label": True}])
            argv = [str(d / "chats.jsonl"), str(d / "samples.jsonl"), "--out", str(d / "pairs")]
            pairs.main(argv + ["--kto-labels", str(d / "labels.jsonl")])
            manifest = json.loads((d / "pairs/manifest.json").read_text())
            self.assertEqual((manifest["kto_desirable"], manifest["kto_undesirable"]), (2, 1))
            self.assertEqual(len(manifest["inputs"]), 3)
            self.assertTrue((d / "pairs/kto_train.jsonl").exists())
            pairs.main(argv)   # labels withdrawn: the old KTO file must not survive
            self.assertFalse((d / "pairs/kto_train.jsonl").exists())
            self.assertTrue((d / "pairs/dpo_train.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
