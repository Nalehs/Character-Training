import unittest
from character_pipeline.data.chats import convert, ends_in_punctuation


def row(text, flags=()):
    return {"scenario_id": "t0:q0:k0", "seed": 1, "history": [
        {"speaker": "partner", "text": "q"}, {"speaker": "character", "text": text, "flags": list(flags)}]}


class Chats(unittest.TestCase):
    def test_converts_and_marks_unreviewed(self):
        out, dropped = convert([row("An answer.")])
        self.assertEqual(out, [{"id": "t0:q0:k0:1", "messages": [{"role": "user", "content": "q"},
                                {"role": "assistant", "content": "An answer."}], "reviewed": False}])

    def test_filters(self):
        _, d = convert([row("", ["empty"]), row("cut off", ["incomplete"])])
        self.assertEqual(d["flagged"], 2)
        out, d = convert([row("no stop")], require_final_punct=True)
        self.assertEqual((out, d["no_final_punct"]), ([], 1))
        self.assertTrue(ends_in_punctuation("ends」"))

    def test_truncated_teacher_reply_is_dropped(self):
        # The flag generate.teacher really writes for a reply cut at the token limit.
        from character_pipeline.generate import teacher
        cut = teacher.run_scenario({"id": "s", "caller_turns": ["q"]}, "S", "http://x", 1, "GOLEM",
                                   call=lambda url, body: {"content": "A cut reply.", "stop_type": "limit"})
        self.assertEqual(cut["history"][1]["flags"], ["incomplete_or_unknown_finish"])
        out, d = convert([cut], require_final_punct=True)
        self.assertEqual((out, d["flagged"]), ([], 1))

    def test_flag_on_a_later_turn_drops_the_whole_conversation(self):
        r = row("Fine.")
        r["history"] += [{"speaker": "partner", "text": "q2"},
                         {"speaker": "character", "text": "cut", "flags": ["incomplete_or_unknown_finish"]}]
        self.assertEqual(convert([r])[0], [])


if __name__ == "__main__":
    unittest.main()
