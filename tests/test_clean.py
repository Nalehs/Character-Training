import unittest
from character_pipeline.data.clean import clean, diverse_subset, group_split, near_duplicates


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


if __name__ == "__main__":
    unittest.main()
