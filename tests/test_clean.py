import json, tempfile, unittest
from pathlib import Path
from character_pipeline.data import clean as clean_module
from character_pipeline.data.clean import (clean, diverse_subset, group_split, near_duplicates, prompt_overlap,
                                           read_decisions, unmatched_decisions)


def convo(sid, seed, replies, flags=None):
    h = []
    for i, r in enumerate(replies):
        h.append({"speaker": "partner", "text": f"q{i}"})
        h.append({"speaker": "character", "text": r, "flags": (flags or {}).get(i, [])})
    return {"scenario_id": sid, "seed": seed, "history": h}


class Clean(unittest.TestCase):
    def test_unreviewed_turns_cut_and_never_repaired(self):
        rows = [convo("a", 1, ["one", "two", "three"])]
        kept, excluded = clean(rows, {("a:1", 0): "accept", ("a:1", 1): "accept"})
        self.assertEqual([m["content"] for m in kept[0]["messages"]], ["q0", "one", "q1", "two"])
        self.assertEqual(kept[0]["cut_at"], 5)

    def test_rejected_first_turn_excludes_conversation(self):
        kept, excluded = clean([convo("a", 1, ["bad"])], {("a:1", 0): "reject"})
        self.assertEqual(kept, []); self.assertEqual(excluded[0]["reason"], "no accepted character turn")

    def test_hygiene_flag_cuts_even_if_accepted(self):
        rows = [convo("a", 1, ["ok", "- list"], flags={1: ["bullets"]})]
        kept, _ = clean(rows, {("a:1", 0): "accept", ("a:1", 1): "accept"})
        self.assertEqual(len(kept[0]["messages"]), 2)

    def test_duplicates_dropped(self):
        rows = [convo("a", 1, ["same"]), convo("a", 2, ["same"])]
        kept, excluded = clean(rows, {("a:1", 0): "accept", ("a:2", 0): "accept"})
        self.assertEqual(len(kept), 1); self.assertEqual(excluded[0]["reason"], "duplicate_conversation")

    def test_group_split_keeps_families_whole(self):
        rows = [{"family": f"f{i % 8}", "id": i} for i in range(40)]
        train, held = group_split(rows, lambda r: r["family"])
        self.assertFalse({r["family"] for r in train} & {r["family"] for r in held})
        self.assertEqual(len({r["family"] for r in held}), 1)

    def test_diversity(self):
        v = [[1, 0], [0.99, 0.01], [0, 1], [-1, 0]]
        self.assertEqual([p[:2] for p in near_duplicates(v)], [(0, 1)])
        pick = diverse_subset(v, 3, seed=0)
        self.assertEqual(len(pick), 3)
        self.assertFalse({0, 1} <= set(pick))
        self.assertEqual(diverse_subset(v, 0), [])

    def test_single_family_never_empties_training(self):
        rows = [{"family": "only", "id": i} for i in range(3)]
        train, held = group_split(rows, lambda r: r["family"])
        self.assertEqual((len(train), len(held)), (3, 0))
        train, held = group_split([{"family": f"f{i}"} for i in range(2)], lambda r: r["family"], 0.9)
        self.assertEqual((len(train), len(held)), (1, 1))
        self.assertEqual(group_split(rows, lambda r: r["family"], 0)[1], [])

    def test_split_is_deterministic_for_a_seed(self):
        rows = [{"family": f"f{i % 8}", "id": i} for i in range(40)]
        self.assertEqual(group_split(rows, lambda r: r["family"], seed=3), group_split(rows, lambda r: r["family"], seed=3))

    def test_only_an_explicit_accept_accepts(self):
        sheet = [{"conversation": "a:1", "turn": 0, "decision": " Accept "},
                 {"conversation": "a:1", "turn": 1, "decision": "accepted"},
                 {"conversation": "a:1", "turn": 2, "decision": ""},
                 {"conversation": "b:1", "turn": 0, "decision": "accept"},
                 {"conversation": "b:1", "turn": 0, "decision": "reject"},
                 {"conversation": "zz:9", "turn": 0, "decision": "accept"}]
        decisions, problems = read_decisions(sheet)
        self.assertEqual(decisions[("a:1", 0)], "accept")
        self.assertEqual(decisions[("a:1", 1)], "invalid")
        self.assertEqual(decisions[("a:1", 2)], "unreviewed")
        self.assertEqual(decisions[("b:1", 0)], "conflict")
        self.assertEqual(sorted(p["problem"] for p in problems), ["conflicting decisions", "invalid decision"])
        rows = [convo("a", 1, ["one", "two", "three"]), convo("b", 1, ["x"])]
        self.assertEqual([p["conversation"] for p in unmatched_decisions(rows, decisions)], ["zz:9"])
        kept, excluded = clean(rows, decisions)
        self.assertEqual([m["content"] for m in kept[0]["messages"]], ["q0", "one"])
        self.assertEqual([e["conversation"] for e in excluded], ["b:1"])

    def test_prompt_overlap_is_reported(self):
        a = {"id": "a", "messages": [{"role": "user", "content": "same"}, {"role": "assistant", "content": "x"}]}
        b = {"id": "b", "messages": [{"role": "user", "content": "same"}, {"role": "assistant", "content": "y"}]}
        c = {"id": "c", "messages": [{"role": "user", "content": "other"}, {"role": "assistant", "content": "y"}]}
        self.assertEqual(prompt_overlap([a], [b, c]), ["b"])


class CleanMain(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.d = Path(self.dir.name)

    def write(self, n_families):
        rows = []
        for i in range(n_families):
            row = convo(f"s{i}", i, [f"reply {i}"])
            row["history"][0]["text"] = f"question {i}"
            rows.append(row)
        (self.d / "judged.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
        sheet = [{"conversation": f"s{i}:{i}", "turn": 0, "decision": "accept"} for i in range(n_families)]
        (self.d / "review.jsonl").write_text("\n".join(json.dumps(r) for r in sheet) + "\n\n")
        return [str(self.d / "judged.jsonl"), str(self.d / "review.jsonl"), "--out", str(self.d / "clean")]

    def test_system_prompt_is_prepended_and_split_is_disjoint(self):
        (self.d / "system.md").write_text("You are X.\n")
        clean_module.main(self.write(8) + ["--system", str(self.d / "system.md")])
        out = self.d / "clean"
        train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
        held = [json.loads(x) for x in (out / "validation.jsonl").read_text().splitlines()]
        self.assertEqual((len(train), len(held)), (7, 1))
        for r in train + held:
            self.assertEqual(r["messages"][0], {"role": "system", "content": "You are X."})
            self.assertEqual([m["role"] for m in r["messages"][1:]], ["user", "assistant"])
        self.assertFalse({r["family"] for r in train} & {r["family"] for r in held})
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in held})
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["validation_prompt_overlap"], [])
        self.assertEqual(manifest["split"]["seed"], 42)
        with self.assertRaises(SystemExit):   # never overwritten
            clean_module.main(self.write(8))

    def test_one_family_stops_before_writing_anything(self):
        with self.assertRaisesRegex(SystemExit, "two families"):
            clean_module.main(self.write(1))
        self.assertFalse((self.d / "clean").exists())


if __name__ == "__main__":
    unittest.main()
