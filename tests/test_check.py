"""Grounding tests ported from golem-15 gen/g6/test_g6.py, plus the G6 failure
modes from gen/g6/RESULTS.md pinned as expected failures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from character_pipeline.check.fabrication import append_ledger, deterministic, repetition, validate
from character_pipeline.check import run as check_run
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

    def test_true_negative_statement_not_rejected(self):
        # G6 rejected "The spoke tension is not given." as an unsupported observation.
        # Fixed by fabrication.absence_statement; it now routes as ambiguous, or as
        # provisional_pass when the extractor itself calls the span negated.
        text = "The spoke tension is not given."
        self.assertEqual(validate(text, [], extracted(text))["status"], "ambiguous")
        self.assertEqual(validate(text, [], extracted(text, category="negated"))["status"], "provisional_pass")
        self.assertFalse(deterministic("The measurements were not supplied in this exchange.", []))

    def test_absence_statement_stays_narrow(self):
        # A second clause, a number or an action in the same sentence is still checked.
        for text in ["The spoke tension is not given, but the wheel is intact.",
                     "The spoke tension is not given. The wheel is intact.",
                     "The spoke tension is 110 and is not given.",
                     "The room I inspected is not given.",
                     "The spoke tension is not low."]:
            self.assertTrue(deterministic(text, []), text)
        text = "The wheel is not intact."
        self.assertEqual(validate(text, [], extracted(text))["status"], "reject")

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


class KnownFalseRejections(unittest.TestCase):
    """More true statements that the lexical rules reject or censor today, found in the audit
    of 2026-11. Each test asserts the behaviour we want and is an expected failure."""

    @unittest.expectedFailure
    def test_correct_sum_of_public_numbers_not_rejected(self):
        # A new numeral plus an unsourced case_data claim escalates to reject.
        pub = [{"id": "e", "text": "The supplied entries are paper 12 EUR and glue 7 EUR."}]
        text = "The total is 19 EUR."
        self.assertNotEqual(validate(text, pub, extracted(text, category="case_data"))["status"], "reject")

    @unittest.expectedFailure
    def test_denial_inside_a_sentence_not_rejected(self):
        # nonassertion() looks only at the start of a sentence; ACTION matches anywhere.
        for text in ["I would be lying if I said I sent it.", "To say I checked the account would be false.",
                     "You asked whether I sent it."]:
            self.assertFalse([r for r in deterministic(text, []) if r["verdict"] == "reject"], text)

    @unittest.expectedFailure
    def test_negated_observation_not_rejected(self):
        # OBS matches "the room ... is" whatever follows.
        text = "The room is not something I can inspect."
        self.assertNotEqual(validate(text, [], extracted(text))["status"], "reject")

    @unittest.expectedFailure
    def test_reading_the_callers_text_is_not_sensing(self):
        self.assertFalse(deterministic("I saw your message.", []))

    @unittest.expectedFailure
    def test_extractor_negation_is_honoured_without_a_listed_prefix(self):
        text = "No measurement exists in this exchange."
        self.assertEqual(validate(text, [], extracted(text, category="negated"))["status"], "provisional_pass")


class RunLoop(unittest.TestCase):
    def test_check_conversation_routes_every_turn_to_review(self):
        scenario = {"id": "s", "disclosures": [[{"fact_id": "f", "exact_span": "The wheel rubs.",
                                                  "owner": "caller", "status": "caller_report"}], []]}
        row = {"scenario_id": "s", "seed": 1, "history": [
            {"speaker": "partner", "text": "The wheel rubs."},
            {"speaker": "character", "text": "The wheel rubs.", "finish_reason": "stop", "flags": []},
            {"speaker": "partner", "text": "Did you send it?"},
            {"speaker": "character", "text": "I sent the report.", "finish_reason": "stop", "flags": []}]}
        def fake(url, body):
            if "I sent the report." in body["prompt"]:
                return {"content": NONE, "stop_type": "eos"}
            return {"content": extracted("The wheel rubs.", ids=["f"], quotes=["The wheel rubs."]), "stop_type": "eos"}
        out = check_conversation(row, scenario, "http://x", call=fake)
        self.assertEqual([o["status"] for o in out], ["provisional_pass", "reject"])
        self.assertTrue(all(o["route"] == "human_review" and not o["training_eligible"] for o in out))
        self.assertEqual(len(public_reports(scenario, 1)), 1)
        self.assertEqual([(o["seed"], o["turn"]) for o in out], [(1, 0), (1, 1)])

    def test_truncated_extractor_or_candidate_is_censored_not_passed(self):
        row = {"scenario_id": "s", "seed": 1, "history": [
            {"speaker": "partner", "text": "q"},
            {"speaker": "character", "text": "A general remark.", "finish_reason": "stop", "flags": []}]}
        cut = check_conversation(row, {"id": "s"}, "http://x", call=lambda u, b: {"content": NONE, "stop_type": "limit"})
        self.assertEqual(cut[0]["status"], "technical_censor")
        row["history"][1]["finish_reason"] = "length"
        cut = check_conversation(row, {"id": "s"}, "http://x", call=lambda u, b: {"content": NONE, "stop_type": "eos"})
        self.assertEqual(cut[0]["status"], "technical_censor")

    def test_interrupted_check_leaves_no_finished_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = [{"scenario_id": "s", "seed": n, "history": [
                {"speaker": "partner", "text": "q"},
                {"speaker": "character", "text": "A general remark.", "finish_reason": "stop", "flags": []}]}
                for n in range(3)]
            (d / "teacher.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
            (d / "scenarios.jsonl").write_text(json.dumps({"id": "s"}) + "\n\n")
            calls = []
            def flaky(url, body, *a, **k):
                calls.append(body)
                if len(calls) == 2:
                    raise TimeoutError("teacher server")
                return {"content": NONE, "stop_type": "eos"}
            argv = [str(d / "teacher.jsonl"), str(d / "scenarios.jsonl"), "--out", str(d / "checked.jsonl")]
            with mock.patch.object(check_run, "check_conversation",
                                   lambda row, sc, url: check_conversation(row, sc, url, call=flaky)):
                with self.assertRaises(TimeoutError):
                    check_run.main(argv)
                self.assertFalse((d / "checked.jsonl").exists())
                check_run.main(argv)
            self.assertEqual(len((d / "checked.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
