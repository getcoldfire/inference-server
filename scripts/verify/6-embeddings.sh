#!/usr/bin/env bash
# Test 6 — embeddings.
#
# Skips with a launch hint unless the loaded model is an embeddings model.
# Sends a 2-document /v1/embeddings request, asserts:
#   - HTTP 200
#   - 2 embedding vectors returned
#   - each vector has dim > 0
#   - each vector's L2 norm is within 0.001 of 1.0 (most embedding
#     models, including nomic-embed-text, return L2-normalized vectors)

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_common.sh
. "$SCRIPT_DIR/_common.sh"

header "[6/6] Embeddings"

require_server
model=$(detect_model)
capability=$(detect_capability "$model")

if [ "$capability" != "embeddings" ]; then
  skip "test 6 requires an embedding server (this server: $capability, model=$model)"
  info ""
  info "launch one with:"
  info "  coldfire-mlx-server launch --port 8081 \\"
  info "    --model-path mlx-community/nomic-embed-text-v1.5-4bit --model-type embeddings"
  info "then re-run with:"
  info "  COLDFIRE_PORT=8081 $0"
  exit 0
fi

info "Model: $model"
info "Sending POST /v1/embeddings with 2 inputs..."

start=$(date +%s)

payload=$(MODEL="$model" python3 -c '
import json, os
model = os.environ["MODEL"]
body = {
    "model": model,
    "input": ["hello world", "embeddings test"],
}
print(json.dumps(body))
')

http_code=$(curl -s -o /tmp/.cf-verify-emb.json -w '%{http_code}' --max-time 60 \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "${BASE_URL}/v1/embeddings" 2>/dev/null; true)

end=$(date +%s)
elapsed=$((end - start))

if [ "$http_code" != "200" ]; then
  info "${CROSS} HTTP ${http_code} (expected 200)"
  if [ -s /tmp/.cf-verify-emb.json ]; then
    info "    response: $(head -c 300 /tmp/.cf-verify-emb.json)"
  fi
  rm -f /tmp/.cf-verify-emb.json
  fail "/v1/embeddings returned ${http_code}"
  exit 1
fi

# Parse + validate.
result=$(python3 -c '
import json, sys, math
try:
    d = json.load(sys.stdin)
except Exception as e:
    print(f"ERR|json-parse|{e}")
    sys.exit(0)

data = d.get("data", [])
if len(data) != 2:
    print(f"ERR|wrong-count|expected 2, got {len(data)}")
    sys.exit(0)

for i, entry in enumerate(data):
    emb = entry.get("embedding", [])
    if not isinstance(emb, list) or not emb:
        print(f"ERR|empty-vector|index {i}")
        sys.exit(0)
    dim = len(emb)
    s = 0.0
    for v in emb:
        try:
            f = float(v)
        except Exception:
            print(f"ERR|non-numeric|index {i}")
            sys.exit(0)
        s += f * f
    l2 = math.sqrt(s)
    print(f"OK|{i}|{dim}|{l2:.6f}")
' < /tmp/.cf-verify-emb.json)
rm -f /tmp/.cf-verify-emb.json

# Parse the result lines.
failures=0
saw_vectors=0
while IFS= read -r line; do
  case "$line" in
    ERR*)
      info "${CROSS} ${line}"
      failures=$((failures + 1))
      ;;
    OK*)
      saw_vectors=$((saw_vectors + 1))
      # OK|i|dim|l2
      rest="${line#OK|}"
      idx="${rest%%|*}"; rest="${rest#*|}"
      dim="${rest%%|*}"; rest="${rest#*|}"
      l2="$rest"

      # Use python to check the L2 norm because bash floats are painful.
      norm_ok=$(python3 -c '
import sys
l2 = float(sys.argv[1])
print("1" if abs(l2 - 1.0) <= 0.001 else "0")
' "$l2")
      if [ "$dim" -le 0 ]; then
        info "${CROSS} doc ${idx}: dim=${dim} (expected > 0)"
        failures=$((failures + 1))
      elif [ "$norm_ok" != "1" ]; then
        info "${CROSS} doc ${idx}: dim=${dim} L2-norm=${l2} (expected ~1.0 ± 0.001)"
        failures=$((failures + 1))
      else
        info "${CHECK} doc ${idx}: dim=${dim} L2-norm=${l2}"
      fi
      ;;
  esac
done <<EOF
$result
EOF

if [ $saw_vectors -ne 2 ]; then
  info "${CROSS} expected 2 vectors, parsed ${saw_vectors}"
  failures=$((failures + 1))
fi

if [ $failures -gt 0 ]; then
  fail "${failures} embedding assertion failure(s)"
  exit 1
fi

pass "embeddings (${elapsed}s)"
exit 0
