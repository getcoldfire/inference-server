#!/usr/bin/env bash
# Shared helpers for the coldfire-inference-server verification suite.
#
# Sourced by run.sh and each individual test script. Provides:
#   - Host/port detection and BASE_URL
#   - Color helpers (tty-aware, NO_COLOR-aware)
#   - pass/fail/skip/info print helpers
#   - require_server   — bail out fast with a friendly hint if no server
#   - detect_model     — fetch the first model id from /v1/models
#   - detect_capability — return "chat" or "embeddings" based on the model id
#
# Target: bash 3.2 (macOS default). No mapfile, no readarray, no ${var,,}.

# Host / port (env-overridable; run.sh may override via positional args)
COLDFIRE_HOST="${COLDFIRE_HOST:-127.0.0.1}"
COLDFIRE_PORT="${COLDFIRE_PORT:-8080}"
BASE_URL="http://${COLDFIRE_HOST}:${COLDFIRE_PORT}"

# Color setup — only when stdout is a tty and NO_COLOR is not set.
# BLUE, CHECK, CROSS are used by scripts that source this file; export
# makes them visible to sourcing scripts AND tells shellcheck they're
# intentionally externally-visible (avoids SC2034 "appears unused").
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  export RED=$'\033[0;31m'
  export GREEN=$'\033[0;32m'
  export YELLOW=$'\033[0;33m'
  export BLUE=$'\033[0;34m'
  export BOLD=$'\033[1m'
  export RESET=$'\033[0m'
else
  export RED=''
  export GREEN=''
  export YELLOW=''
  export BLUE=''
  export BOLD=''
  export RESET=''
fi

# Tick / cross marks (plain ASCII fallback already since these are basic UTF-8)
export CHECK="${GREEN}✓${RESET}"
export CROSS="${RED}✗${RESET}"

info() {
  printf '  %s\n' "$*"
}

pass() {
  printf '%sPASS%s — %s\n' "$GREEN" "$RESET" "$*"
}

fail() {
  printf '%sFAIL%s — %s\n' "$RED" "$RESET" "$*"
}

skip() {
  printf '%sSKIP%s — %s\n' "$YELLOW" "$RESET" "$*"
}

# Print a one-line header in cyan/blue.
header() {
  printf '%s%s%s\n' "$BOLD" "$*" "$RESET"
}

# require_server: curl /healthz with a short timeout. If non-200, print
# a friendly error including the launch hint and exit 2.
require_server() {
  local http_code
  http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "${BASE_URL}/healthz" 2>/dev/null; true)
  if [ "$http_code" != "200" ]; then
    printf '%sNo server detected at %s%s\n' "$RED" "$BASE_URL" "$RESET" >&2
    printf '\n' >&2
    printf '  Launch one in another terminal:\n' >&2
    printf '\n' >&2
    printf '    coldfire-inference-server launch \\\n' >&2
    printf '      --host %s \\\n' "$COLDFIRE_HOST" >&2
    printf '      --port %s \\\n' "$COLDFIRE_PORT" >&2
    printf '      --model-path mlx-community/Llama-3.2-1B-Instruct-4bit\n' >&2
    printf '\n' >&2
    printf '  Or override host/port via env:\n' >&2
    printf '    COLDFIRE_HOST=...  COLDFIRE_PORT=...  %s\n' "$0" >&2
    exit 2
  fi
}

# detect_model: print the first model id from /v1/models, or empty string on failure.
detect_model() {
  local body
  body=$(curl -s --max-time 3 "${BASE_URL}/v1/models" 2>/dev/null || true)
  if [ -z "$body" ]; then
    return 1
  fi
  printf '%s' "$body" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    data = d.get("data", [])
    if data:
        print(data[0].get("id", ""))
except Exception:
    pass
' 2>/dev/null || true
}

# detect_capability: print "chat" or "embeddings" based on the loaded model id.
# Heuristic: if the model id (case-insensitive) contains "embed" or "nomic",
# treat it as embeddings; otherwise chat. Not bulletproof — good enough for
# the menu and per-test skip logic.
detect_capability() {
  local model="$1"
  # Lowercase via tr (bash 3.2 has no ${var,,}).
  local lower
  lower=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')
  case "$lower" in
    *embed*|*nomic*|*bert*)
      printf 'embeddings'
      ;;
    *)
      printf 'chat'
      ;;
  esac
}
