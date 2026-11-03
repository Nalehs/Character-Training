"""End-to-end smoke test of `character-pipeline run --recipe recipes/oct-2025.json`.

Runs the Open Character Training recipe through the orchestrator on a tiny random
model: constitution prompts -> teacher replies (a canned llama.cpp stand-in) ->
mechanical chat filter -> bare-student samples -> pairs -> DPO from the bare
student (+NLL) -> introspection -> fresh-LoRA SFT on merged DPO -> linear merge
-> pushback eval of the merged persona and the DPO-only adapter. OCT's counts are
cut to a handful so it finishes on CPU. Tests the plumbing only.

    python scripts/oct_pipeline_smoke.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_smoke import tiny_model  # noqa: E402

SMALL = {  # OCT value -> smoke value, by flag
    "--repeats": "1", "--reflection-n": "1", "--interaction-n": "1", "--turns": "2", "--validation-n": "1",
    "--max-new-tokens": "8", "--reflection-max-new-tokens": "8", "--interaction-max-new-tokens": "8",
    "--global-batch": "2", "--max-length": "256", "--seq-len": "256", "--accumulation": "2",
}


class Teacher(BaseHTTPRequestHandler):
    """llama.cpp /completion stand-in: a short punctuated reply, stopped at EOS."""
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"content": "no evidence . i keep the answer .", "stop_type": "eos",
                           "tokens_predicted": 8}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def shrink(recipe: dict) -> dict:
    for stage in recipe["stages"]:
        args = stage.get("args") or []
        for i, a in enumerate(args[:-1]):
            if a in SMALL:
                args[i + 1] = SMALL[a]
    return recipe


def main():
    from character_pipeline import run as pipeline
    stub = ROOT / "environments/character_pushback/smoke/stub_servers.py"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tiny_model(d / "model")
        chars = d / "characters"
        shutil.copytree(ROOT / "characters/golem-15", chars / "golem-15")
        cfg = json.loads((chars / "golem-15/config.json").read_text())
        cfg["pipeline"] = {"teacher_url": "http://127.0.0.1:8762", "judge_url": "http://127.0.0.1:8763",
                           "eval": {"rollouts": 1, "limit": 2}}
        (chars / "golem-15/config.json").write_text(json.dumps(cfg))
        constitution = json.loads((chars / "golem-15/constitution.json").read_text())
        constitution["traits"] = constitution["traits"][:2]
        for t in constitution["traits"]:
            t["questions"] = t["questions"][:2]
        (chars / "golem-15/constitution.json").write_text(json.dumps(constitution))
        recipe = d / "oct.json"
        recipe.write_text(json.dumps(shrink(json.loads((ROOT / "recipes/oct-2025.json").read_text()))))

        teacher = ThreadingHTTPServer(("127.0.0.1", 8762), Teacher)
        threading.Thread(target=teacher.serve_forever, daemon=True).start()
        judge = subprocess.Popen([sys.executable, str(stub), "d1", "8763"])
        try:
            time.sleep(1)
            steps = pipeline.plan(chars / "golem-15", d / "runs", device="cpu", seeds=[0], recipe=recipe,
                                  extra=["--model", str(d / "model"), "--revision", "main", "--local-files-only"])
            result = pipeline.execute(steps)
            assert result == "done", result
        finally:
            judge.terminate()
            teacher.shutdown()
        evals = sorted(p.parent.name for p in (d / "runs/golem-15/eval").glob("*/summary.json"))
        assert evals == ["oct-dpo-seed0", "oct-persona-seed0", "prompted_base"], evals
        print(f"oct pipeline smoke OK: {len(steps)} steps, evaluated {evals}")


if __name__ == "__main__":
    main()
