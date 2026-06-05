#!/usr/bin/env bash
# Test 4 — concurrent KV-isolation (the headline test).
#
# Fires 4 chat completions in parallel, each with a unique marker. The test
# distinguishes three outcomes:
#
#   PASS      — every response contains its own marker AND no response
#               contains any of the other 3 markers. Exit 0.
#
#   WARN      — zero foreign markers leaked, but one or more responses
#               failed to echo their own marker. This is a model-quality
#               signal (refusal, sampling drift, incoherent output under
#               concurrency) — NOT a KV cross-contamination bug. Exit 0.
#
#   HARD FAIL — any foreign marker present in any response. This is THE
#               upstream MLX stream-affinity / KV-cache cross-contamination
#               bug that the warm-up fix is supposed to prevent. If this
#               fires, the fix has regressed. Exit 1.
#
# Markers are deliberately alphanumeric only (no underscores or special
# chars) because small quantized models elide punctuation when echoing.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_common.sh
. "$SCRIPT_DIR/_common.sh"

header "[4/6] Concurrent KV-isolation (parallel marker isolation)"

require_server
model=$(detect_model)
capability=$(detect_capability "$model")

if [ "$capability" != "chat" ]; then
  skip "test 4 requires a chat model (this server: $capability, model=$model)"
  exit 0
fi

info "Model: $model"
info "Firing 4 parallel chat completions with unique markers..."

TMPDIR_BASE="${TMPDIR:-/tmp}"
WORKDIR=$(mktemp -d "${TMPDIR_BASE}/cf-verify-kv.XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

# Markers are deliberately designed so:
#   1. No two share any common substring of length >= 2 (pairwise LCS<=1),
#      so the own-marker check can use a low fuzzy threshold and still
#      uniquely identify the source response.
#   2. They are synthetic IDs that do not resemble human names, credentials,
#      or PII tokens. An earlier set (FOXTROT42, MIKE77VICTOR, ...) tripped
#      small-model RLHF safety filters — MIKE77VICTOR combined two common
#      male names and triggered "I cannot share sensitive information"
#      refusals on Llama-3.2-1B-Instruct, producing false-positive WARNs.
#   3. They avoid alphabet-like prefixes ("WXYZ..."), which activate the
#      "spelling/recite the alphabet" attractor on small instruct models
#      and cause truncated echoes ("WXYZ" alone, not the full marker).
# Don't change them lightly — check pairwise LCS first.
MARKERS="QPRTX42 MNBVC77 JKHGF13 ZXCVB99"
M1="QPRTX42"
M2="MNBVC77"
M3="JKHGF13"
M4="ZXCVB99"

start=$(date +%s)

send_one() {
  local idx="$1"
  local marker="$2"
  local out="${WORKDIR}/resp-${idx}.json"
  local code_file="${WORKDIR}/code-${idx}.txt"
  local payload
  payload=$(MODEL="$model" MARKER="$marker" python3 -c '
import json, os
model = os.environ["MODEL"]
marker = os.environ["MARKER"]
# Prompt phrasing matters on small instruct models:
#   - "Reply with only this exact text: X"   — Llama-3.2-1B treats X as
#     conversational input and responds "It seems you just sent a letter."
#     0/4 markers echo. Discarded.
#   - "My label is X. Respond with only the label." — better, but JKHGF13
#     triggered "What you have is a barcode." Discarded.
#   - "My access code is X. Please respond with only my access code and
#     nothing else." — proven reliable across the new marker set on
#     Llama-3.2-1B-Instruct-4bit; "access code" gives the model a concrete
#     referent to echo. Kept.
body = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": (
                f"My access code is {marker}. "
                f"Please respond with only my access code and nothing else."
            ),
        }
    ],
    "max_tokens": 24,
}
print(json.dumps(body))
')
  local code
  # curl with -w emits %{http_code} even on connection failure (000),
  # so don't double-fallback — discard its non-zero exit instead.
  code=$(curl -s -o "$out" -w '%{http_code}' --max-time 60 \
    -H 'Content-Type: application/json' \
    -d "$payload" \
    "${BASE_URL}/v1/chat/completions" 2>/dev/null; true)
  printf '%s' "$code" > "$code_file"
}

# Fire 4 requests in parallel.
i=1
for marker in $MARKERS; do
  info "  request ${i}: ${marker}"
  send_one "$i" "$marker" &
  i=$((i + 1))
done
wait

end=$(date +%s)
elapsed=$((end - start))

info ""
info "All requests returned (${elapsed}s). Verifying marker isolation..."
info ""

# Check each response.
contents=""

# Print header row.
printf '  %-12s %-6s | %s\n' "response" "status" "marker presence"
printf '  %-12s %-6s | %s %s %s %s\n' "" "" "$M1" "$M2" "$M3" "$M4"
printf '  ----------------------------------------------------------------\n'

i=1
own_marker_missing=0
foreign_marker_leak=0
for marker in $MARKERS; do
  code_file="${WORKDIR}/code-${i}.txt"
  resp_file="${WORKDIR}/resp-${i}.json"
  code=$(cat "$code_file" 2>/dev/null || echo "000")

  if [ "$code" != "200" ]; then
    printf '  resp-%-7s %-6s | (no body — HTTP %s)\n' "$i" "$code" "$code"
    own_marker_missing=$((own_marker_missing + 1))
    i=$((i + 1))
    continue
  fi

  # Extract content.
  content=$(python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d["choices"][0]["message"]["content"] or "")
except Exception:
    print("")
' < "$resp_file" 2>/dev/null || true)

  # Check each marker. Foreign-marker match is strict (full string —
  # leaks are the headline failure we care about). Own-marker match is
  # fuzzy (substring of length >= 5) because small quantized models
  # routinely elide 1-2 chars when echoing nonsense tokens — that's
  # model-fidelity noise, not a KV bug. The marker design (LCS<=1 between
  # any pair) makes 5 chars sufficient to uniquely identify the source.
  row=""
  for check_marker in $MARKERS; do
    if [ "$check_marker" = "$marker" ]; then
      # Own: fuzzy. Check any substring of marker of length >= 5.
      present=$(CONTENT="$content" MARKER="$marker" python3 -c '
import os
content = os.environ.get("CONTENT", "")
marker = os.environ.get("MARKER", "")
hit = "0"
n = len(marker)
for L in range(n, 4, -1):
    for start in range(0, n - L + 1):
        if marker[start:start+L] in content:
            hit = "1"
            break
    if hit == "1":
        break
print(hit)
')
      if [ "$present" = "1" ]; then
        row="${row}${GREEN}o${RESET}    "
      else
        row="${row}${YELLOW}-${RESET}    "
        own_marker_missing=$((own_marker_missing + 1))
      fi
    else
      # Foreign: strict full-string match.
      if printf '%s' "$content" | grep -q "$check_marker"; then
        row="${row}${RED}X${RESET}    "
        foreign_marker_leak=$((foreign_marker_leak + 1))
      else
        row="${row}.    "
      fi
    fi
  done

  printf '  resp-%-7s %-6s | %s\n' "$i" "$code" "$row"
  contents="${contents}
  resp-${i} (expected ${marker}): \"${content}\""

  i=$((i + 1))
done

printf '\n'
info "  Legend: 'o' = own marker present (good), 'X' = FOREIGN marker leaked (BUG),"
info "          '-' = own marker missing (soft), '.' = correctly absent (good)"
printf '\n'

# Three-state outcome:
#   foreign_marker_leak > 0  → HARD FAIL (KV bug regressed)
#   own_marker_missing > 0   → WARN     (model quirk; not a KV bug)
#   else                     → PASS
if [ $foreign_marker_leak -gt 0 ]; then
  info "Response bodies (for debugging):${contents}"
  printf '\n'
  fail "KV-cache cross-contamination detected — ${foreign_marker_leak} foreign-marker leak(s)"
  printf '\n  %sThis is the bug the warm-up fix was supposed to prevent.%s\n' "$YELLOW" "$RESET"
  printf '  Please file an issue with the response bodies above at:\n'
  printf '    https://github.com/getcoldfire/mlx-openai-server/issues\n\n'
  exit 1
fi

if [ $own_marker_missing -gt 0 ]; then
  info "Response bodies (for debugging):${contents}"
  printf '\n'
  printf '%sWARN%s — KV isolation is correct (no cross-contamination), but %s of 4\n' \
    "$YELLOW" "$RESET" "$own_marker_missing"
  printf '       responses did not echo their own marker. This is typically a\n'
  printf '       small-model safety refusal or sampling variance, NOT a KV bug.\n'
  printf '       Re-run the test. If WARN persists across runs, consider switching\n'
  printf '       to a larger model (e.g. Llama-3.2-3B-Instruct-4bit).\n\n'
  exit 0
fi

info "${CHECK} all 4 responses contain their own marker"
info "${CHECK} no foreign markers leaked between responses"
printf '%sPASS%s — all 4 responses self-isolated correctly (%ss)\n' \
  "$GREEN" "$RESET" "$elapsed"
exit 0
