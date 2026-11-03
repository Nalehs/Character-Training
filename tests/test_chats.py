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


if __name__ == "__main__":
    unittest.main()
