#!/usr/bin/env bash
# Test 3 — streaming chat completion.
#
# Sends a streaming request, reads the SSE stream, asserts:
#   - at least 3 `data: {...}` chunks received
#   - stream terminates with `data: [DONE]`
#   - each non-[DONE] chunk parses as JSON

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_common.sh
. "$SCRIPT_DIR/_common.sh"

header "[3/6] Streaming chat completion"

require_server
model=$(detect_model)
capability=$(detect_capability "$model")

if [ "$capability" != "chat" ]; then
  skip "test 3 requires a chat model (this server: $capability, model=$model)"
  exit 0
fi

info "Model: $model"
info "Sending POST /v1/chat/completions with stream=true (max_tokens=20)..."

start=$(date +%s)

payload=$(MODEL="$model" python3 -c '
import json, os
model = os.environ["MODEL"]
body = {
    "model": model,
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "max_tokens": 20,
    "stream": True,
}
print(json.dumps(body))
' 2>/dev/null)

# Capture the streamed response. -N disables curl buffering.
curl -sN --max-time 90 \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "${BASE_URL}/v1/chat/completions" > /tmp/.cf-verify-stream.out 2>/tmp/.cf-verify-stream.err
curl_rc=$?

end=$(date +%s)
elapsed=$((end - start))

if [ $curl_rc -ne 0 ]; then
  info "${CROSS} curl exited with rc=$curl_rc"
  if [ -s /tmp/.cf-verify-stream.err ]; then
    info "    stderr: $(head -c 200 /tmp/.cf-verify-stream.err)"
  fi
  rm -f /tmp/.cf-verify-stream.out /tmp/.cf-verify-stream.err
  fail "streaming request failed"
  exit 1
fi
rm -f /tmp/.cf-verify-stream.err

# Count and validate. Use a Python one-liner via -c so we can redirect
# the SSE output as stdin without colliding with a heredoc.
result=$(python3 -c '
import json, sys

data_chunks = 0
parse_errors = 0
saw_done = False

for raw in sys.stdin:
    line = raw.strip()
    if not line.startswith("data:"):
        continue
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        saw_done = True
        continue
    if not payload:
        continue
    try:
        json.loads(payload)
        data_chunks += 1
    except Exception:
        parse_errors += 1

print(f"{data_chunks}|{int(saw_done)}|{parse_errors}")
' < /tmp/.cf-verify-stream.out)
rm -f /tmp/.cf-verify-stream.out

chunks="${result%%|*}"
rest="${result#*|}"
done_flag="${rest%%|*}"
parse_errors="${rest#*|}"

info "  chunks parsed: ${chunks}"
info "  [DONE] sentinel seen: $( [ "$done_flag" = "1" ] && echo yes || echo no )"
info "  JSON parse errors: ${parse_errors}"

fail_reasons=0
if [ "$chunks" -lt 3 ]; then
  info "${CROSS} expected at least 3 data chunks, got ${chunks}"
  fail_reasons=$((fail_reasons + 1))
fi
if [ "$done_flag" != "1" ]; then
  info "${CROSS} stream did not terminate with [DONE] sentinel"
  fail_reasons=$((fail_reasons + 1))
fi
if [ "$parse_errors" -gt 0 ]; then
  info "${CROSS} ${parse_errors} chunks failed JSON parsing"
  fail_reasons=$((fail_reasons + 1))
fi

if [ $fail_reasons -gt 0 ]; then
  fail "streaming assertions failed (${fail_reasons})"
  exit 1
fi

info "${CHECK} ${chunks} valid chunks + [DONE] received"
pass "streaming completion (${elapsed}s)"
exit 0
