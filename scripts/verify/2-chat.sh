#!/usr/bin/env bash
# Test 2 — basic chat completion.
#
# Sends a single chat completion (max_tokens=8) and asserts HTTP 200 +
# non-empty content. We don't require the model to output exactly "PONG"
# — small/quantized models routinely drift to "Pong!", "PONG.", etc.
# This test catches actual breakage (500s, empty responses, malformed
# JSON), not output quality.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_common.sh
. "$SCRIPT_DIR/_common.sh"

header "[2/6] Basic chat completion"

require_server
model=$(detect_model)
capability=$(detect_capability "$model")

if [ "$capability" != "chat" ]; then
  skip "test 2 requires a chat model (this server: $capability, model=$model)"
  exit 0
fi

info "Model: $model"
info "Sending POST /v1/chat/completions (max_tokens=8)..."

start=$(date +%s)

payload=$(MODEL="$model" python3 -c '
import json, os
model = os.environ["MODEL"]
body = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
    "max_tokens": 8,
}
print(json.dumps(body))
' 2>/dev/null)

http_code=$(curl -s -o /tmp/.cf-verify-chat.json -w '%{http_code}' --max-time 60 \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "${BASE_URL}/v1/chat/completions" 2>/dev/null; true)

end=$(date +%s)
elapsed=$((end - start))

if [ "$http_code" != "200" ]; then
  info "${CROSS} HTTP ${http_code} (expected 200)"
  if [ -s /tmp/.cf-verify-chat.json ]; then
    info "    response: $(head -c 300 /tmp/.cf-verify-chat.json)"
  fi
  rm -f /tmp/.cf-verify-chat.json
  fail "chat completion returned ${http_code}"
  exit 1
fi

content=$(python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    c = d["choices"][0]["message"]["content"]
    print(c if c is not None else "")
except Exception as e:
    print(f"__PARSE_ERROR__: {e}", file=sys.stderr)
    sys.exit(2)
' < /tmp/.cf-verify-chat.json 2>/tmp/.cf-verify-chat.err)
parse_rc=$?
rm -f /tmp/.cf-verify-chat.json

if [ $parse_rc -ne 0 ]; then
  info "${CROSS} response JSON parse failed: $(cat /tmp/.cf-verify-chat.err 2>/dev/null)"
  rm -f /tmp/.cf-verify-chat.err
  fail "could not parse chat completion response"
  exit 1
fi
rm -f /tmp/.cf-verify-chat.err

# Strip whitespace.
trimmed=$(printf '%s' "$content" | tr -d '[:space:]')
if [ -z "$trimmed" ]; then
  info "${CROSS} content was empty after trimming whitespace"
  fail "empty completion content"
  exit 1
fi

info "${CHECK} HTTP 200"
info "${CHECK} content: \"${content}\""
pass "basic chat completion (${elapsed}s)"
exit 0
