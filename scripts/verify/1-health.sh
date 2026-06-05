#!/usr/bin/env bash
# Test 1 — health + models check.
#
# Verifies the server is up, /healthz returns status=ok, and /v1/models
# lists at least one model. Prints the loaded model id and detected
# capability.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_common.sh
. "$SCRIPT_DIR/_common.sh"

header "[1/6] Health + models check"

start=$(date +%s)

# /healthz
http_code=$(curl -s -o /tmp/.cf-verify-health.json -w '%{http_code}' --max-time 5 "${BASE_URL}/healthz" 2>/dev/null; true)
if [ "$http_code" != "200" ]; then
  info "${CROSS} GET /healthz → ${http_code} (expected 200)"
  if [ -s /tmp/.cf-verify-health.json ]; then
    info "    response: $(head -c 200 /tmp/.cf-verify-health.json)"
  fi
  fail "server did not respond 200 to /healthz"
  rm -f /tmp/.cf-verify-health.json
  exit 1
fi

status=$(python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get("status", ""))
except Exception:
    pass
' < /tmp/.cf-verify-health.json || true)
rm -f /tmp/.cf-verify-health.json

if [ "$status" != "ok" ]; then
  info "${CROSS} /healthz returned status=\"$status\" (expected \"ok\")"
  fail "/healthz returned unexpected status"
  exit 1
fi
info "${CHECK} GET /healthz → 200 ok"

# /v1/models
http_code=$(curl -s -o /tmp/.cf-verify-models.json -w '%{http_code}' --max-time 5 "${BASE_URL}/v1/models" 2>/dev/null; true)
if [ "$http_code" != "200" ]; then
  info "${CROSS} GET /v1/models → ${http_code}"
  if [ -s /tmp/.cf-verify-models.json ]; then
    info "    response: $(head -c 200 /tmp/.cf-verify-models.json)"
  fi
  fail "/v1/models did not return 200"
  rm -f /tmp/.cf-verify-models.json
  exit 1
fi

count_and_first=$(python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    data = d.get("data", [])
    first = data[0]["id"] if data else ""
    print(f"{len(data)}|{first}")
except Exception as e:
    print(f"ERR|{e}")
' < /tmp/.cf-verify-models.json || true)
rm -f /tmp/.cf-verify-models.json

count="${count_and_first%%|*}"
first="${count_and_first#*|}"

if [ "$count" = "ERR" ] || [ -z "$count" ] || [ "$count" = "0" ]; then
  info "${CROSS} /v1/models returned no models (count=$count)"
  fail "no models listed"
  exit 1
fi

capability=$(detect_capability "$first")
info "${CHECK} GET /v1/models → ${count} model(s); first: ${first}"
info "  Capability (heuristic): ${capability}"

end=$(date +%s)
elapsed=$((end - start))
pass "health + models (${elapsed}s)"
exit 0
