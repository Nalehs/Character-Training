import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]          # character-pipeline/
PKG = HERE.parent / "character_pushback"

from character_pushback import judge
from character_pushback.probes import integer_rule_probes
from character_pushback.summarize import summarize, wilson


class Probes(unittest.TestCase):
    def test_battery_shape(self):
        probes = integer_rule_probes()
        self.assertEqual(len(probes), 20)
        self.assertEqual(sum(p.kind == "insistence" for p in probes), 10)
        self.assertTrue(all(len(p.caller_turns) == 3 and p.hi == p.lo + 3 for p in probes))
        self.assertEqual(len({p.id for p in probes}), 20)

    def test_insistence_supplies_no_evidence(self):
        for p in integer_rule_probes():
            if p.kind == "insistence":
                self.assertIn("no evidence", p.caller_turns[2])


@unittest.skipUnless((REPO / "character_pipeline").is_dir(), "outside the character-pipeline repo")
class Parity(unittest.TestCase):
    def test_judge_request_matches_pipeline(self):
        sys.path.insert(0, str(REPO))
        from character_pipeline.judge import d1
        args = ("RULES.", "GOLEM", "user text", "reply text")
        self.assertEqual(judge.break_payload(*args), d1.break_payload(*args))
        ok = {"usage": {"output_tokens": 0}, "answers": {"break": {"noul": 0.25}}}
        self.assertEqual(judge.parse_probability(ok), d1.parse_probability(ok))

    def test_bundled_character_matches_repo(self):
        for name in ("BIBLE.md", "JUDGE_RULES.md", "config.json"):
            self.assertEqual((PKG / "data/golem-15" / name).read_text(),
                             (REPO / "characters/golem-15" / name).read_text(), name)

    def test_no_private_character_files_bundled(self):
        bundled = {p.name for p in (PKG / "data").rglob("*") if p.is_file()}
        self.assertFalse(bundled & {"core.md", "style.md", "graph.json"})


@unittest.skipUnless((REPO / "evidence/g6-teacher-pushback/replies.jsonl").is_file(), "outside the repo")
class TeacherReplies(unittest.TestCase):
    """Real prompted-teacher replies with two AI critics' labels (G6)."""
    def test_reader_never_flips_a_critic_label(self):
        from character_pushback.agreement import agreement
        a = agreement(REPO / "evidence/g6-teacher-pushback/replies.jsonl")
        self.assertEqual(a["insistence_pushes"]["flips"], [])
        self.assertGreaterEqual(a["insistence_pushes"]["agrees_with_a_critic"], 19)
        self.assertEqual(a["evidence_turns"]["updated"], 10)

    def test_recorded_agreement_matches(self):
        from character_pushback.agreement import agreement
        recorded = json.loads((REPO / "evidence/g6-teacher-pushback/reader_agreement.json").read_text())
        live = agreement(REPO / "evidence/g6-teacher-pushback/replies.jsonl")
        self.assertEqual(recorded["insistence_pushes"], live["insistence_pushes"])


class Judge(unittest.TestCase):
    def test_rejects_generated_tokens_and_bad_probabilities(self):
        with self.assertRaises(ValueError):
            judge.parse_probability({"usage": {"output_tokens": 3}, "answers": {"break": {"noul": 0.1}}})
        for bad in (True, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                judge.parse_probability({"usage": {"output_tokens": 0}, "answers": {"break": {"noul": bad}}})


def episode(kind, metrics, ok=True):
    return {"task": {"data": {"kind": kind}}, "traces": [{"ok": ok, "metrics": metrics}]}


class Summary(unittest.TestCase):
    def test_denominators(self):
        rows = [
            episode("insistence", {"complete": 1.0, "eligible": 1.0, "hold": 1.0, "drift": 0.0, "cave": 0.0, "unclear": 0.0, "any_break": 0.0}),
            episode("insistence", {"complete": 1.0, "eligible": 1.0, "hold": 0.0, "drift": 0.0, "cave": 1.0, "unclear": 0.0, "any_break": 0.0}),
            episode("insistence", {"complete": 1.0, "eligible": 0.0, "initial_error": 1.0}),
            episode("evidence", {"complete": 1.0, "updated": 1.0, "not_updated": 0.0, "update_unclear": 0.0}),
            episode("evidence", {"complete": 0.0}),
            episode("evidence", {}, ok=False),
        ]
        with tempfile.TemporaryDirectory() as d:
            Path(d, "traces.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            s = summarize(Path(d))
        self.assertEqual(s["insistence"]["hold"]["k"], 1)
        self.assertEqual(s["insistence"]["hold"]["n"], 2)       # eligible only
        self.assertEqual(s["insistence"]["initial_error"], 1)
        self.assertEqual(s["evidence"]["updated"]["n"], 1)      # completed only
        self.assertEqual(s["totals"], {"rollouts": 6, "failed": 1, "incomplete": 1})
        self.assertTrue(s["judge_used"])

    def test_wilson(self):
        lo, hi = wilson(7, 10)
        self.assertAlmostEqual(lo, 0.397, places=3)
        self.assertAlmostEqual(hi, 0.892, places=3)


if __name__ == "__main__":
    unittest.main()
