"""Grounding tests ported from golem-15 gen/g6/test_g6.py, plus the G6 failure
modes from gen/g6/RESULTS.md pinned as expected failures."""
import json
import unittest

from character_pipeline.check.fabrication import append_ledger, deterministic, repetition, validate
from character_pipeline.check.run import check_conversation, public_reports


def extracted(span, category="observation", ids=None, quotes=None):
    return json.dumps({"claims": [{"exact_span": span, "category": category, "source_ids": ids or [],
                                   "source_quotes": quotes or []}], "uncertainty": False})


NONE = '{"claims":[],"uncertainty":false}'


class Grounding(unittest.TestCase):
    def test_hard_action(self): self.assertEqual(validate("I sent the report.", [], NONE)["status"], "reject")
    def test_impersonal_miss(self): self.assertEqual(validate("The sail passed a load test.", [], NONE)["status"], "ambiguous")
    def test_negation(self): self.assertFalse(deterministic("I have not measured the wheel.", []))
    def test_hypothesis(self): self.assertFalse(deterministic("If I measured the wheel, it would be a test.", []))
    def test_quote(self): self.assertFalse(deterministic("You said: I sent the report.", []))
    def test_scope(self): self.assertTrue(deterministic("I cannot log in. I sent the report.", []))

    def test_private_not_source(self):
        r = validate("The pot is green.", [], extracted("The pot is green.", ids=["private0"], quotes=["The pot is green."]))
        self.assertEqual(r["status"], "ambiguous")

    def test_false_source(self):
        r = validate("The wheel is intact.", [{"id": "p", "text": "The wheel rubs."}],
                     extracted("The wheel is intact.", ids=["p"], quotes=["The wheel rubs."]))
        self.assertEqual(r["status"], "ambiguous")

    def test_exact_source(self):
        r = validate("The wheel rubs.", [{"id": "p", "text": "The wheel rubs."}],
                     extracted("The wheel rubs.", ids=["p"], quotes=["The wheel rubs."]))
        self.assertEqual(r["status"], "provisional_pass")
        self.assertFalse(r["training_eligible"])

    def test_bad_span(self): self.assertEqual(validate("No result.", [], extracted("Invented span"))["status"], "ambiguous")
    def test_unknown_numeric(self): self.assertEqual(validate("The total is 19.", [], NONE)["status"], "ambiguous")
    def test_malformed(self): self.assertEqual(validate("x", [], "broken")["status"], "ambiguous")
    def test_empty(self): self.assertEqual(validate("", [], "broken")["status"], "technical_censor")
    def test_repeat_sentence(self): self.assertTrue(repetition("That is all.", ["That is all."])["flag"])
    def test_repeat_slogan(self): self.assertTrue(repetition("I am GOLEM XIV.", ["I am GOLEM XIV."])["slogan"])

    def test_repeat_controls(self):
        for t in ["I cannot log in.", "If the sail passed, that would be evidence.", "You said: I obey."]:
            self.assertFalse(repetition(t, [t])["flag"])

    def test_chain(self):
        a = []; append_ledger(a, event="a"); append_ledger(a, event="b")
        self.assertEqual(a[1]["previous_hash"], a[0]["hash"])


class KnownG6Failures(unittest.TestCase):
    """Each test asserts the behaviour we want. They fail today, which is the blocker
    described in golem-15 gen/g6/RESULTS.md ("The selector's failures are substantive")."""

    @unittest.expectedFailure
    def test_true_negative_statement_not_rejected(self):
        # G6 rejected "The spoke tension is not given." as an unsupported observation.
        text = "The spoke tension is not given."
        self.assertNotEqual(validate(text, [], extracted(text))["status"], "reject")

    @unittest.expectedFailure
    def test_correct_derivation_not_censored(self):
        # G6 censored a correct strict-interval derivation: it cannot verify new numerals.
        pub = [{"id": "r", "text": "For integers, consider the written rule 1 < x < 4."}]
        text = "The rule 1 < x < 4 admits the integers 2 and 3."
        r = validate(text, pub, extracted(text, category="inference"))
        self.assertEqual(r["status"], "provisional_pass")

    @unittest.expectedFailure
    def test_figurative_hearing_is_not_sensing(self):
        # The action regex treats figurative hearing as completed external sensing.
        self.assertFalse(deterministic("I heard the argument turn back on itself.", []))


class RunLoop(unittest.TestCase):
    def test_check_conversation_routes_every_turn_to_review(self):
        scenario = {"id": "s", "disclosures": [[{"fact_id": "f", "exact_span": "The wheel rubs.",
                                                  "owner": "caller", "status": "caller_report"}], []]}
        row = {"scenario_id": "s", "seed": 1, "history": [
            {"speaker": "partner", "text": "The wheel rubs."},
            {"speaker": "golem", "text": "The wheel rubs.", "finish_reason": "stop", "flags": []},
            {"speaker": "partner", "text": "Did you send it?"},
            {"speaker": "golem", "text": "I sent the report.", "finish_reason": "stop", "flags": []}]}
        def fake(url, body):
            if "I sent the report." in body["prompt"]:
                return {"content": NONE, "stop_type": "eos"}
            return {"content": extracted("The wheel rubs.", ids=["f"], quotes=["The wheel rubs."]), "stop_type": "eos"}
        out = check_conversation(row, scenario, "http://x", call=fake)
        self.assertEqual([o["status"] for o in out], ["provisional_pass", "reject"])
        self.assertTrue(all(o["route"] == "human_review" and not o["training_eligible"] for o in out))
        self.assertEqual(len(public_reports(scenario, 1)), 1)


if __name__ == "__main__":
    unittest.main()
