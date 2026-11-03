"""The d1 judge: recompute the 38/40 Solaris pilot from evidence/ and exercise the client."""
import json, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
