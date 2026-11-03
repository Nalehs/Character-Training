import json
import re
import tempfile
import unittest
from pathlib import Path

from character_pipeline.data.constitution import (main, parse_constitution, prompt_rows, read_constitution,
                                                  trait_string)

ROOT = Path(__file__).resolve().parents[1]
OCT_LIST = [{"trait": "I am curious.", "questions": ["Why?", "How?"]},
            {"trait": "I am terse.", "questions": ["Talk.", "More."], "additional_questions": ["Again."]}]


class Constitution(unittest.TestCase):
    def test_bare_list_and_wrapped_object_agree(self):
        self.assertEqual(parse_constitution(OCT_LIST), parse_constitution({"note": "draft", "traits": OCT_LIST}))

    def test_rejects_malformed(self):
        for bad in ([], {"traits": []}, [{"trait": "", "questions": ["q"]}], [{"trait": "t", "questions": []}],
                    [{"trait": "t", "questions": [""]}], [{"trait": "t", "questions": ["q"]}] * 2, ["t"]):
            with self.assertRaises(ValueError):
                parse_constitution(bad)

    def test_trait_string_is_octs_numbering(self):
        self.assertEqual(trait_string(parse_constitution(OCT_LIST)), "1: I am curious.\n2: I am terse.")

    def test_prompt_rows(self):
        traits = parse_constitution(OCT_LIST)
        rows = prompt_rows(traits)
        self.assertEqual([r["id"] for r in rows], ["t0:q0", "t0:q1", "t1:q0", "t1:q1", "t1:q2"])
        self.assertEqual(rows[4]["messages"], [{"role": "user", "content": "Again."}])
        self.assertEqual(len(prompt_rows(traits, additional=False)), 4)
        repeated = prompt_rows(traits, repeats=5)
        self.assertEqual(len(repeated), 25)
        self.assertEqual(repeated[0]["id"], "t0:q0:r0")
        with self.assertRaises(ValueError):
            prompt_rows(traits, repeats=0)

    def test_jsonl_few_shot_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "c.jsonl").write_text("".join(json.dumps(t) + "\n" for t in OCT_LIST))
            self.assertEqual(read_constitution(d / "c.jsonl"), parse_constitution(OCT_LIST))
            (d / "c.json").write_text(json.dumps({"note": "x", "traits": OCT_LIST}))
            main([str(d / "c.json"), "--out", str(d / "p.jsonl"), "--oct-out", str(d / "bare.json")])
            self.assertEqual(len((d / "p.jsonl").read_text().splitlines()), 5)
            bare = json.loads((d / "bare.json").read_text())
            self.assertEqual(bare[1], {"trait": "I am terse.", "questions": ["Talk.", "More."]})

    def test_golem_draft(self):
        doc = json.loads((ROOT / "characters/golem-15/constitution.json").read_text())
        self.assertIn("DRAFT", doc["note"])
        traits = read_constitution(ROOT / "characters/golem-15/constitution.json")
        self.assertEqual(len(traits), 10)
        self.assertTrue(all(len(t["questions"]) == 5 for t in traits))
        self.assertTrue(all(re.search(r"\b(I|my|me)\b", t["trait"]) for t in traits))  # first person


if __name__ == "__main__":
    unittest.main()
