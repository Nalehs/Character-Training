#!/usr/bin/env bash
# Stage 4 judge: LiquidAI d1-2B Q8_0 on CPU via llama.cpp, the configuration
# of decision-models d1-2b (commit 9c73e6f): official llama.cpp b11960,
# -c 2048 -b 512 -ub 512, three threads, one slot, loopback only, API key.
# Usage: D1_API_KEY=$(openssl rand -hex 24) scripts/serve_d1.sh
set -euo pipefail
: "${D1_API_KEY:?set D1_API_KEY; the environment and pipeline read the same variable}"
LLAMA_SERVER=${LLAMA_SERVER:-llama-server}
MODEL_DIR=${MODEL_DIR:-models}
REPO=LiquidAI/d1-2B-GGUF
REV=e968f6575f0d4fad928312d7af15cad90fd365c0
FILE=d1-2B-Q8_0.gguf
SHA256=073313d92b53b0ac8193db6e906cb3c78a98183d244b15b7b8f5426115c6738c
BYTES=1846240576
PORT=${PORT:-8090}
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/$FILE" ]; then
  curl -L --fail -o "$MODEL_DIR/$FILE.partial" "https://huggingface.co/$REPO/resolve/$REV/$FILE"
  mv "$MODEL_DIR/$FILE.partial" "$MODEL_DIR/$FILE"
fi
test "$(stat -c %s "$MODEL_DIR/$FILE")" = "$BYTES"
echo "$SHA256  $MODEL_DIR/$FILE" | sha256sum -c -
LLAMA_API_KEY="$D1_API_KEY" OMP_NUM_THREADS=3 exec "$LLAMA_SERVER" -m "$MODEL_DIR/$FILE" \
  --host 127.0.0.1 --port "$PORT" -t 3 -tb 3 -ngl 0 -c 2048 -b 512 -ub 512 -np 1 \
  --threads-http 1 --no-webui --no-context-shift
