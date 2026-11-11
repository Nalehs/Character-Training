"""A minimal OpenAI-compatible chat server for base + LoRA adapter.

Enough to point the pushback environment (vf-eval) at a trained arm without
installing vLLM: POST /v1/chat/completions, streaming or not, one request at
a time. For real throughput use vLLM with LoRA, e.g.

    vllm serve Qwen/Qwen3.5-4B-Base --revision 1001bb4d... --enable-lora \
        --lora-modules golem-dpo=runs/golem-15/dpo/seed0/adapter

A request's own temperature, top_p and max_tokens (or max_completion_tokens)
are used when present. When a request leaves them out, as verifiers does, the
server's --temperature (default 0 = greedy), --top-p and --max-tokens apply.
Sampling is seeded once at start-up (--seed), so sampled output is repeatable
only if requests arrive in the same order. Prompts are always rendered with
thinking off, matching training. `stop`, `n` and tools are not supported;
malformed requests get a JSON error with status 400, generation errors 500.
It binds 127.0.0.1 unless --host says otherwise and has no authentication.

    python -m character_pipeline.serve --adapter runs/golem-15/dpo/seed0/adapter --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from character_pipeline.train.common import STUDENT, STUDENT_REVISION
from character_pipeline.train.sample import generate, load_policy


MAX_BODY = 8 << 20  # bytes; a chat request is far smaller


class BadRequest(ValueError):
    pass


def parse_request(body, defaults: dict) -> tuple[list[dict], dict]:
    """Validate an OpenAI chat-completions body; return (messages, decoding settings)."""
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        raise BadRequest("body must be a JSON object with a non-empty \"messages\" list")
    messages = []
    for m in body["messages"]:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
            raise BadRequest("every message needs a role of system, user or assistant")
        content = m.get("content")
        if isinstance(content, list):  # content parts: keep the text ones
            if not all(isinstance(part, dict) for part in content):
                raise BadRequest("message content parts must be objects")
            content = "".join(part.get("text") or "" for part in content if part.get("type", "text") == "text")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise BadRequest("message content must be a string or a list of text parts")
        messages.append({"role": m["role"], "content": content})
    if body.get("n") not in (None, 1):
        raise BadRequest("only n=1 is supported")

    def number(keys, default, low, high, integer=False):
        value = next((body[k] for k in keys if body.get(k) is not None), default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high \
                or (integer and value != int(value)):
            raise BadRequest(f"{keys[0]} must be a number from {low} to {high}")
        return int(value) if integer else float(value)

    return messages, {
        "max_new_tokens": number(("max_completion_tokens", "max_tokens"), defaults["max_tokens"], 1, 32768, True),
        "temperature": number(("temperature",), defaults["temperature"], 0, 2),  # an explicit 0 stays greedy
        "top_p": number(("top_p",), defaults["top_p"], 1e-6, 1)}


def make_handler(tok, model, device, name, lock, defaults=None):
    defaults = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512} | (defaults or {})

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._json({"object": "list", "data": [{"id": name, "object": "model"}]})
            self._error(404, f"no such route: GET {self.path}")

        def do_POST(self):
            if not self.path.split("?")[0].rstrip("/").endswith("/chat/completions"):
                return self._error(404, f"no such route: POST {self.path}")
            try:
                length = int(self.headers.get("Content-Length") or "")
            except ValueError:
                return self._error(411, "Content-Length is required")
            if not 0 < length <= MAX_BODY:
                return self._error(413 if length > MAX_BODY else 400, f"body must be 1 to {MAX_BODY} bytes")
            try:
                body = json.loads(self.rfile.read(length))
                messages, decoding = parse_request(body, defaults)
            except (ValueError, UnicodeDecodeError) as e:  # BadRequest and JSONDecodeError are ValueErrors
                return self._error(400, str(e))
            try:
                with lock:  # one generation at a time
                    text, finish, prompt_tokens, n = complete(tok, model, messages, device, decoding)
            except Exception as e:  # noqa: BLE001 - a bad prompt must not take the server down or hang the client
                return self._error(500, f"{type(e).__name__}: {e}")
            finish = "stop" if finish == "eos" else "length"
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n}
            base = {"id": f"chatcmpl-{time.time_ns():x}", "created": int(time.time()), "model": name}
            if not body.get("stream"):
                return self._json(base | {"object": "chat.completion", "usage": usage, "choices": [
                    {"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            chunk = base | {"object": "chat.completion.chunk"}
            for c in (chunk | {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                      chunk | {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                      chunk | {"choices": [], "usage": usage}):
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

        def _json(self, out, status=200):
            data = json.dumps(out).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status, message):
            kind = "invalid_request_error" if status < 500 else "server_error"
            self._json({"error": {"message": message, "type": kind, "code": status}}, status)

        def log_message(self, *args):
            pass
    return Handler


def complete(tok, model, messages, device, decoding):
    """(text, finish, prompt tokens, completion tokens) for one request."""
    text, finish = generate(tok, model, messages, decoding["max_new_tokens"], decoding["temperature"],
                            decoding["top_p"], device)
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    return text, finish, len(ids), len(tok(text, add_special_tokens=False)["input_ids"])


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
    ap.add_argument("--temperature", type=float, default=0.0, help="used when a request sets none; 0 = greedy")
    ap.add_argument("--top-p", type=float, default=1.0, help="used when a request sets none")
    ap.add_argument("--max-tokens", type=int, default=512, help="used when a request sets none")
    ap.add_argument("--seed", type=int, default=0, help="seeds sampling once, at start-up")
    args = ap.parse_args(argv)
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1 or args.max_tokens < 1:
        ap.error("--temperature must be in [0, 2], --top-p in (0, 1] and --max-tokens at least 1")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: binding {args.host}: this server has no authentication; anyone who can reach the "
              "port can use the model", file=sys.stderr, flush=True)
    import torch
    torch.manual_seed(args.seed)
    tok, model = load_policy(args.model, args.revision, args.adapter, args.device, args.local_files_only)
    defaults = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(tok, model, args.device, args.name, threading.Lock(), defaults))
    print(f"serving {args.adapter or args.model} as '{args.name}' on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
