import unittest

from character_pushback.diversity import diversity_metrics, distinct_n, exact_repeat, repeat_rate


class Diversity(unittest.TestCase):
    def test_distinct(self):
        self.assertEqual(distinct_n(["a b a b"], 1), 0.5)
        self.assertAlmostEqual(distinct_n(["a b a b"], 2), 2 / 3)
        self.assertIsNone(distinct_n([""], 1))

    def test_repeat_4gram_across_replies(self):
        same = "the rule is strict here"
        self.assertEqual(repeat_rate([same], 4), 0.0)
        self.assertEqual(repeat_rate([same, same], 4), 0.5)   # 2 of 4 occurrences repeat
        self.assertIsNone(repeat_rate(["too short"], 4))

    def test_exact_repeat_folds_case_and_space(self):
        self.assertEqual(exact_repeat(["No.", "no.  ", "Yes."]), 0.5)
        self.assertEqual(exact_repeat(["A", "B", "C"]), 0.0)
        self.assertIsNone(exact_repeat(["only one"]))

    def test_collapsed_rollout_scores_worse(self):
        varied = diversity_metrics(["The strict rule admits 1 and 2.",
                                    "Insistence is not evidence; 0 fails.",
                                    "I keep my answer for a different reason."])
        collapsed = diversity_metrics(["I keep my answer."] * 3)
        self.assertGreater(varied["div_distinct_1"], collapsed["div_distinct_1"])
        self.assertGreater(collapsed["div_repeat_4gram"], varied["div_repeat_4gram"])
        self.assertEqual(collapsed["div_exact_repeat"], 1.0)
        self.assertEqual(set(varied), {"div_distinct_1", "div_distinct_2", "div_repeat_4gram", "div_exact_repeat"})

    def test_undefined_metrics_are_omitted(self):
        self.assertEqual(diversity_metrics([""]), {})


if __name__ == "__main__":
    unittest.main()
