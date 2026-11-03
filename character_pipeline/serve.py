"""A minimal OpenAI-compatible chat server for base + LoRA adapter.

Enough to point the pushback environment (vf-eval) at a trained arm without
installing vLLM: POST /v1/chat/completions, streaming or not, one request at
a time. For real throughput use vLLM with LoRA, e.g.

    vllm serve Qwen/Qwen3.5-4B-Base --revision 1001bb4d... --enable-lora \
        --lora-modules golem-dpo=runs/golem-15/dpo/seed0/adapter

This server decodes greedily unless the request sets a temperature, and always
renders with thinking off, matching training.

    python -m character_pipeline.serve --adapter runs/golem-15/dpo/seed0/adapter --port 8000
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from character_pipeline.train.common import STUDENT, STUDENT_REVISION
from character_pipeline.train.sample import generate, load_policy


def make_handler(tok, model, device, name, lock):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._json({"object": "list", "data": [{"id": name, "object": "model"}]})
            self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                         else "".join(p.get("text", "") for p in m["content"])} for m in body["messages"]]
            with lock:  # one generation at a time
                text, finish = generate(tok, model, messages, body.get("max_tokens") or 512,
                                        body.get("temperature") or 0.0, body.get("top_p") or 1.0, device)
            finish = "stop" if finish == "eos" else "length"
            n = len(tok(text)["input_ids"])
            usage = {"prompt_tokens": 0, "completion_tokens": n, "total_tokens": n}
            base = {"id": "cp", "created": int(time.time()), "model": body.get("model", name)}
            if not body.get("stream"):
                return self._json(base | {"object": "chat.completion", "usage": usage, "choices": [
                    {"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = base | {"object": "chat.completion.chunk"}
            for c in (chunk | {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                      chunk | {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                      chunk | {"choices": [], "usage": usage}):
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _json(self, out):
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter")
    ap.add_argument("--model", default=STUDENT)
    ap.add_argument("--revision", default=STUDENT_REVISION)
    ap.add_argument("--name", default="character")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args(argv)
    tok, model = load_policy(args.model, args.revision, args.adapter, args.device, args.local_files_only)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tok, model, args.device, args.name, threading.Lock()))
    print(f"serving {args.adapter or args.model} as '{args.name}' on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
