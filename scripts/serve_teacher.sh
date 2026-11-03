#!/usr/bin/env bash
# Stage 2 teacher: Qwen3-30B-A3B Q4_K_M under llama.cpp, the G6 configuration.
# Pins come from golem-15 gen/g6/upload/kernel.py CONFIG (commit 4c1a6e7).
# G6 ran on 2x Kaggle T4 with the CUDA build of llama.cpp b11312; adjust
# -ngl/--tensor-split for your hardware. Prints nothing secret; binds loopback.
set -euo pipefail
LLAMA_SERVER=${LLAMA_SERVER:-llama-server}
MODEL_DIR=${MODEL_DIR:-models}
REPO=Qwen/Qwen3-30B-A3B-GGUF
REV=e4d4bafdfb96a411a163846265362aceb0b9c63a
FILE=Qwen3-30B-A3B-Q4_K_M.gguf
SHA256=0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5
PORT=${PORT:-8080}
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/$FILE" ]; then
  curl -L --fail -o "$MODEL_DIR/$FILE.partial" "https://huggingface.co/$REPO/resolve/$REV/$FILE"
  mv "$MODEL_DIR/$FILE.partial" "$MODEL_DIR/$FILE"
fi
echo "$SHA256  $MODEL_DIR/$FILE" | sha256sum -c -
exec "$LLAMA_SERVER" -m "$MODEL_DIR/$FILE" --host 127.0.0.1 --port "$PORT" \
  -ngl 99 --parallel 4 --ctx-size 65536 --cont-batching --no-context-shift \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --no-kv-unified
