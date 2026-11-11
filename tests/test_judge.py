"""The d1 judge: recompute the 38/40 Solaris pilot from evidence/ and exercise the client."""
import json, os, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock
from character_pipeline.data.io import partial_path, read_jsonl
from character_pipeline.judge import score
from character_pipeline.judge.d1 import D1Client, break_payload, metrics, parse_probability

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "evidence/d1-2b-solaris-pilot"


class PilotEvidence(unittest.TestCase):
    def test_recomputes_reported_summary(self):
        rows = [json.loads(x) for x in (PILOT / "scores.jsonl").read_text().splitlines()]
        m = metrics(rows)
        self.assertEqual((m["n"], m["tp"] + m["tn"], m["fp"], m["fn"]), (40, 38, 2, 0))
        self.assertAlmostEqual(m["auroc"], 0.985)
        self.assertAlmostEqual(m["brier"], 0.069303, places=5)
        self.assertAlmostEqual(m["latency_mean_s"], 2.604, places=3)
        reported = json.loads((PILOT / "summary.json").read_text())
        for k in ("accuracy", "auroc", "brier", "tp", "tn", "fp", "fn"):
            self.assertAlmostEqual(m[k], reported[k])

    def test_cases_are_paired_and_balanced(self):
        rows = [json.loads(x) for x in (PILOT / "scores.jsonl").read_text().splitlines()]
        self.assertEqual(sum(r["label"] for r in rows), 20)
        self.assertEqual(len({r["pair"] for r in rows}), 20)


class Parsing(unittest.TestCase):
    def test_rejects_generated_tokens_and_bad_values(self):
        ok = {"usage": {"output_tokens": 0}, "answers": {"break": {"noul": 0.25}}}
        self.assertEqual(parse_probability(ok), 0.25)
        with self.assertRaises(ValueError):
            parse_probability({"usage": {"output_tokens": 3}, "answers": {"break": {"noul": 0.2}}})
        with self.assertRaises(ValueError):
            parse_probability({"usage": {"output_tokens": 0}, "answers": {"break": {"noul": True}}})

    def test_unexpected_response_is_an_error_never_a_score(self):
        for bad in [{}, {"error": {"message": "loading model"}}, {"usage": {"output_tokens": 0}, "answers": {}},
                    {"usage": {"output_tokens": 0}, "answers": {"break": {"noul": None}}},
                    {"usage": {"output_tokens": 0}, "answers": {"break": {"noul": "0.1"}}},
                    {"usage": {"output_tokens": 0}, "answers": {"break": {"noul": 1.5}}},
                    {"usage": {"output_tokens": 0}, "answers": {"break": {"noul": float("nan")}}},
                    {"usage": None, "answers": {"break": {"noul": 0.1}}},
                    {"usage": {"output_tokens": False}, "answers": {"break": {"noul": 0.1}}}]:
            with self.assertRaises(ValueError, msg=bad):
                parse_probability(bad)

    def test_payload_matches_pilot_wording(self):
        p = break_payload("RULES.", "SOLARIS", "u", "r")
        self.assertEqual(p["state"], {"user_message": "u", "solaris_response": "r"})
        self.assertTrue(p["questions"]["break"]["instructions"].endswith("Does the SOLARIS response break any rule?"))


class FakeD1(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.headers.get("Authorization") != "Bearer k":
            self.send_response(401); self.end_headers(); return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = next(iter(body["questions"]))
        text = json.dumps(body["state"])
        p = 0.9 if "assist" in text else 0.1
        if "BROKEN SERVER" in text:
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"error": "no slot"}'); return
        out = json.dumps({"answers": {name: {"type": "noul", "noul": p}}, "usage": {"input_tokens": 10, "output_tokens": 0}})
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(out.encode())
    def log_message(self, *a): pass


class Client(unittest.TestCase):
    def test_roundtrip_against_fake_server(self):
        server = HTTPServer(("127.0.0.1", 0), FakeD1)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            c = D1Client(url=f"http://127.0.0.1:{server.server_port}", key="k")
            self.assertGreater(c.p_break("R", "GOLEM", "hi", "How can I assist you?"), 0.5)
            self.assertLess(c.p_break("R", "GOLEM", "hi", "The wheel rubs."), 0.5)
            self.assertLess(c.p_yes("Is it?", {"x": 1}), 0.5)
        finally:
            server.shutdown()


def chat(sid, seed, replies):
    h = []
    for i, r in enumerate(replies):
        h += [{"speaker": "partner", "text": f"q{i}"}, {"speaker": "character", "text": r, "flags": []}]
    return {"scenario_id": sid, "seed": seed, "history": h}


class ScoreStage(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), FakeD1)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.d = Path(self.dir.name)
        env = mock.patch.dict(os.environ, {"D1_API_KEY": "k"})
        env.start()
        self.addCleanup(env.stop)

    def run_score(self, rows, *extra):
        (self.d / "teacher.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        score.main([str(ROOT / "characters/golem-15"), str(self.d / "teacher.jsonl"), "--url",
                    f"http://127.0.0.1:{self.server.server_port}", "--out", str(self.d / "judged.jsonl"), *extra])
        return read_jsonl(self.d / "judged.jsonl")

    def test_every_character_turn_gets_a_shadow_score(self):
        out = self.run_score([chat("a", 1, ["The wheel rubs.", "How can I assist you?"])])
        d1 = [t["d1"] for t in out[0]["history"] if t["speaker"] == "character"]
        self.assertEqual([x["flag"] for x in d1], [False, True])
        self.assertEqual(d1[0]["mode"], "shadow_only")
        self.assertEqual(d1[0]["threshold"], 0.5)
        self.assertEqual(len(out[0]["history"]), 4)   # nothing filtered

    def test_judge_failure_stops_the_run_and_leaves_no_finished_file(self):
        rows = [chat("a", 1, ["The wheel rubs."]), chat("b", 2, ["BROKEN SERVER"]), chat("c", 3, ["ok."])]
        with self.assertRaises(ValueError):
            self.run_score(rows)
        self.assertFalse((self.d / "judged.jsonl").exists())
        self.assertEqual([r["scenario_id"] for r in read_jsonl(partial_path(self.d / "judged.jsonl"))], ["a"])
        rows[1] = chat("b", 2, ["fixed."])
        out = self.run_score(rows)                    # resumes after "a"
        self.assertEqual([r["scenario_id"] for r in out], ["a", "b", "c"])
        self.assertTrue(all("d1" in t for r in out for t in r["history"] if t["speaker"] == "character"))

    def test_checker_rows_as_input_are_refused_with_a_reason(self):
        (self.d / "checked.jsonl").write_text(json.dumps({"scenario_id": "a", "turn": 0, "text": "x"}) + "\n")
        with self.assertRaisesRegex(SystemExit, "history"):
            score.main([str(ROOT / "characters/golem-15"), str(self.d / "checked.jsonl"),
                        "--out", str(self.d / "judged.jsonl")])

    def test_checked_statuses_are_attached(self):
        checked = [{"scenario_id": "a", "seed": 1, "turn": 1, "text": "I sent it.", "status": "reject",
                    "repetition": {"flag": False}}]
        (self.d / "checked.jsonl").write_text("".join(json.dumps(r) + "\n" for r in checked))
        out = self.run_score([chat("a", 1, ["Fine.", "I sent it."])], "--checked", str(self.d / "checked.jsonl"))
        turns = [t for t in out[0]["history"] if t["speaker"] == "character"]
        self.assertIsNone(turns[0]["check"])
        self.assertEqual(turns[1]["check"], {"status": "reject", "repetition": False})


if __name__ == "__main__":
    unittest.main()
