"""serve.py request handling, evaluate.py's command and failure handling, ecology's cross-judge. No model."""
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from character_pipeline import evaluate, serve
from character_pipeline.ecology import _spread, cross_judge, registry

ROOT = Path(__file__).resolve().parents[1]


class Server(unittest.TestCase):
    def setUp(self):
        self.seen = []

        def complete(tok, model, messages, device, decoding):
            self.seen.append((messages, decoding))
            if messages[-1]["content"] == "boom":
                raise RuntimeError("template failed")
            return "ok.", "eos", 5, 2

        patch = mock.patch.object(serve, "complete", complete)
        patch.start()
        self.addCleanup(patch.stop)
        handler = serve.make_handler(None, None, "cpu", "arm", threading.Lock(),
                                     {"temperature": 0.7, "top_p": 0.9, "max_tokens": 64})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def post(self, body, path="/chat/completions", raw=None):
        data = raw if raw is not None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def test_request_without_sampling_fields_uses_server_defaults(self):
        # this is what verifiers sends: model, stream, messages, stream_options
        status, text = self.post({"model": "x", "stream": True, "stream_options": {"include_usage": True},
                                  "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 200)
        self.assertEqual(self.seen[-1][1], {"max_new_tokens": 64, "temperature": 0.7, "top_p": 0.9})
        chunks = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "ok.")
        self.assertEqual(chunks[1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[2]["usage"], {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))

    def test_explicit_zero_temperature_and_token_limits_are_respected(self):
        status, text = self.post({"messages": [{"role": "system", "content": None},
                                               {"role": "user", "content": [{"type": "text", "text": "a"},
                                                                            {"type": "image_url", "image_url": {}}]}],
                                  "temperature": 0, "max_completion_tokens": 7})
        self.assertEqual(status, 200)
        messages, decoding = self.seen[-1]
        self.assertEqual(messages, [{"role": "system", "content": ""}, {"role": "user", "content": "a"}])
        self.assertEqual((decoding["temperature"], decoding["max_new_tokens"]), (0.0, 7))
        self.assertEqual(json.loads(text)["choices"][0]["message"], {"role": "assistant", "content": "ok."})

    def test_bad_requests_get_json_errors_and_the_server_survives(self):
        cases = [(None, b"{not json", 400), ({"messages": []}, None, 400), ({"prompt": "x"}, None, 400),
                 ({"messages": [{"role": "tool", "content": "x"}]}, None, 400),
                 ({"messages": [{"role": "user", "content": 3}]}, None, 400),
                 ({"messages": [{"role": "user", "content": "x"}], "temperature": "hot"}, None, 400),
                 ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, None, 400),
                 ({"messages": [{"role": "user", "content": "x"}], "n": 3}, None, 400),
                 ({"messages": [{"role": "user", "content": "boom"}]}, None, 500)]
        for body, raw, want in cases:
            status, text = self.post(body, raw=raw)
            self.assertEqual(status, want, (body, text))
            self.assertIn("message", json.loads(text)["error"])
        self.assertEqual(self.post({"messages": [{"role": "user", "content": "x"}]}, path="/completions")[0], 404)
        with urllib.request.urlopen(self.url + "/models", timeout=10) as r:
            self.assertEqual(json.loads(r.read())["data"][0]["id"], "arm")
        self.assertEqual(self.post({"messages": [{"role": "user", "content": "x"}]})[0], 200)

    def test_default_bind_is_loopback(self):
        import argparse
        with mock.patch.object(serve, "load_policy", side_effect=KeyboardInterrupt), \
                mock.patch.object(argparse.ArgumentParser, "parse_args", autospec=True,
                                  side_effect=lambda self, argv=None: self.parse_known_args(argv)[0]) as parse:
            with self.assertRaises(KeyboardInterrupt):
                serve.main([])
        self.assertEqual(parse.call_count, 1)
        ap = parse.call_args[0][0]
        self.assertEqual(ap.get_default("host"), "127.0.0.1")


class Evaluate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_command_for_either_verifiers_cli(self):
        old = evaluate.eval_command("dpo", ROOT / "characters/golem-15", "http://h/v1", "http://j", 3, self.tmp,
                                    cli=["vf-eval", "--no-push"])
        new = evaluate.eval_command("prompted_base", ROOT / "characters/golem-15", "http://h/v1", "none", 3, self.tmp,
                                    limit=4, cli=["vf-run"])
        self.assertEqual(old[:2], ["vf-eval", "character-pushback"])
        self.assertIn("--no-push", old)
        self.assertNotIn("--no-push", new)  # verifiers 0.4.1 rejects it
        self.assertEqual(old[old.index("--env.taskset.system-prompt-mode") + 1], "none")
        self.assertEqual(new[new.index("--env.taskset.system-prompt-mode") + 1], "character")
        self.assertEqual(new[new.index("--env.taskset.task.use-judge") + 1], "false")
        self.assertEqual(new[new.index("-n") + 1], "4")

    def test_installed_verifiers_cli_is_found(self):
        self.assertIn(evaluate.vf_cli()[0], ("vf-eval", "vf-run"))
        with mock.patch("shutil.which", return_value=None), self.assertRaises(SystemExit):
            evaluate.vf_cli()

    def run_eval(self, traces, *extra):
        """evaluate.main with the server and verifiers replaced by a canned traces.jsonl."""
        out = self.tmp / "eval"

        def fake_run(cmd, **kw):
            run_dir = Path(cmd[cmd.index("--output-dir") + 1]) / "character-pushback--arm--null--abc"
            run_dir.mkdir(parents=True)
            (run_dir / "traces.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
            return mock.Mock(returncode=0)

        with mock.patch.object(evaluate.subprocess, "Popen", return_value=mock.Mock(poll=lambda: None)), \
                mock.patch.object(evaluate, "wait_ready"), mock.patch.object(evaluate.subprocess, "run", fake_run):
            evaluate.main(["--arm", "sft", "--character", str(ROOT / "characters/golem-15"), "--out", str(out),
                           "--device", "cpu", *extra])
        return out

    def episode(self, ok, kind="insistence", **metrics):
        return {"task": {"data": {"kind": kind, "probe_id": "p", "caller_turns": ["q"]}},
                "traces": [{"ok": ok, "metrics": {"complete": 1.0, **metrics}, "nodes": []}]}

    def test_failed_rollouts_do_not_produce_summary_json(self):
        with self.assertRaises(SystemExit) as e:
            self.run_eval([self.episode(False), self.episode(True, eligible=1.0, hold=1.0)])
        self.assertIn("1 of 2 rollouts failed", str(e.exception))
        self.assertFalse((self.tmp / "eval" / "summary.json").exists())
        self.assertTrue((self.tmp / "eval" / "summary.incomplete.json").exists())

    def test_summary_records_decoding_and_refuses_to_mix_attempts(self):
        out = self.run_eval([self.episode(True, eligible=1.0, hold=1.0), self.episode(True, "evidence", updated=1.0)],
                            "--rollouts", "3")
        s = json.loads((out / "summary.json").read_text())
        self.assertEqual(s["decoding"], {"temperature": 0.0, "seed": 0, "rollouts": 3, "rollouts_independent": False})
        self.assertEqual((s["insistence"]["eligible"], s["insistence"]["hold"]["k"]), (1, 1))
        with self.assertRaises(SystemExit) as e:  # the old vf/ run must not be summarised as a new attempt
            self.run_eval([self.episode(True, eligible=1.0, hold=1.0)])
        self.assertIn("already holds a run", str(e.exception))


class Ecology(unittest.TestCase):
    chars = {k: {"name": k.upper(), "rules": k} for k in "abc"}

    def test_spread_drops_duplicates_and_covers_the_whole_list(self):
        pairs = [{"user": f"u{i // 3}", "reply": "same"} for i in range(30)]  # 3 greedy rollouts of 10 probes
        self.assertEqual(len(_spread(pairs, 20)), 10)
        picked = _spread([{"user": f"u{i}", "reply": "r"} for i in range(100)], 4)
        self.assertEqual([p["user"] for p in picked], ["u0", "u25", "u50", "u75"])

    def test_score_is_zero_for_copies_and_order_independent(self):
        replies = {k: [{"user": "q", "reply": f"{k}{i}"} for i in range(5)] for k in "abc"}
        own = lambda rules, name, user, reply: 0.2 if reply.startswith(rules) else 0.7
        out = cross_judge(self.chars, replies, own)
        self.assertAlmostEqual(out["distinctness"], 0.5)
        self.assertEqual({k: round(v, 6) for k, v in out["by_character"].items()}, {"a": 0.5, "b": 0.5, "c": 0.5})
        renamed = cross_judge({"z": self.chars["a"], "b": self.chars["b"], "c": self.chars["c"]},
                              {"z": replies["a"], "b": replies["b"], "c": replies["c"]}, own)
        self.assertAlmostEqual(renamed["distinctness"], out["distinctness"])
        self.assertEqual(cross_judge(self.chars, replies, lambda *a: 0.4)["distinctness"], 0.0)

    def test_a_character_that_breaks_every_rule_set_is_not_called_distinct(self):
        replies = {k: [{"user": "q", "reply": k}] for k in "ab"}
        judge = lambda rules, name, user, reply: 0.9 if reply == "b" else 0.1  # b is broken under any rules
        out = cross_judge({k: self.chars[k] for k in "ab"}, replies, judge)
        self.assertAlmostEqual(out["by_character"]["b"], 0.0)
        self.assertAlmostEqual(out["distinctness"], 0.0)

    def test_one_character_has_no_score(self):
        out = cross_judge({"a": self.chars["a"]}, {"a": [{"user": "q", "reply": "r"}]}, lambda *a: 0.3)
        self.assertIsNone(out["distinctness"])
        self.assertEqual(out["by_character"], {"a": None})

    def test_registry_reports_a_broken_folder_without_hiding_the_rest(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        shutil.copytree(ROOT / "characters/golem-15", tmp / "good")
        shutil.copytree(ROOT / "characters/golem-15", tmp / "bad")
        cfg = json.loads((tmp / "bad/config.json").read_text())
        cfg.update(id="bad", pipeline={"sedds": [1]})
        (tmp / "bad/config.json").write_text(json.dumps(cfg))
        rows = {r["id"]: r for r in registry(tmp, tmp / "runs")}
        self.assertIn("sedds", rows["bad"]["error"])
        self.assertEqual(rows["golem-15"]["next"], "generate")


if __name__ == "__main__":
    unittest.main()
