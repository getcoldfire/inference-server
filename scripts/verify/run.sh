#!/usr/bin/env bash
# coldfire-mlx-server verification suite — entry point.
#
# Modes:
#   ./run.sh                       interactive menu
#   ./run.sh --all                 run every applicable test, exit non-zero if any fail
#   ./run.sh --test N              run a single test (1-6)
#   ./run.sh --port 9090 ...       override host/port
#   ./run.sh --help                usage
#
# Bash 3.2 compatible. No mapfile / readarray / ${var,,}.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default host/port; --port and --host override before sourcing common.
ARG_HOST=""
ARG_PORT=""
MODE="menu"
SINGLE_TEST=""

usage() {
  cat <<'USAGE'
Usage: run.sh [--host HOST] [--port PORT] [--all | --test N] [--help]

  Interactive verification suite for coldfire-mlx-server.

  Options:
    --host HOST   Override server host (default: 127.0.0.1; or $COLDFIRE_HOST)
    --port PORT   Override server port (default: 8080;       or $COLDFIRE_PORT)
    --all         Run every applicable test non-interactively
    --test N      Run a single test (1-6) non-interactively
    --help        Show this message

  Tests:
    1. Health + models                    (~5 sec)
    2. Basic chat completion              (~10 sec)
    3. Streaming completion               (~10 sec)
    4. Concurrent KV-isolation            (~30 sec)  — the bug test
    5. Clean shutdown (SIGTERM < 5s)
    6. Embeddings (requires embedding server)
USAGE
}

# Parse args.
while [ $# -gt 0 ]; do
  case "$1" in
    --host)
      ARG_HOST="$2"
      shift 2
      ;;
    --port)
      ARG_PORT="$2"
      shift 2
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --test)
      MODE="single"
      SINGLE_TEST="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# Export overrides so _common.sh picks them up.
if [ -n "$ARG_HOST" ]; then
  export COLDFIRE_HOST="$ARG_HOST"
fi
if [ -n "$ARG_PORT" ]; then
  export COLDFIRE_PORT="$ARG_PORT"
fi

# shellcheck source=_common.sh
. "$SCRIPT_DIR/_common.sh"

# Helper: run one test script by number.
run_test() {
  local n="$1"
  local script
  case "$n" in
    1) script="$SCRIPT_DIR/1-health.sh" ;;
    2) script="$SCRIPT_DIR/2-chat.sh" ;;
    3) script="$SCRIPT_DIR/3-streaming.sh" ;;
    4) script="$SCRIPT_DIR/4-concurrent-kv-isolation.sh" ;;
    5) script="$SCRIPT_DIR/5-shutdown.sh" ;;
    6) script="$SCRIPT_DIR/6-embeddings.sh" ;;
    *)
      echo "Unknown test number: $n" >&2
      return 2
      ;;
  esac
  if [ ! -x "$script" ]; then
    echo "Missing or non-executable: $script" >&2
    return 2
  fi
  "$script"
}

# --all mode: run every applicable test based on capability.
run_all() {
  require_server
  local model
  model=$(detect_model)
  local capability
  capability=$(detect_capability "$model")

  printf '\n%sRunning all applicable tests against %s%s\n' "$BOLD" "$BASE_URL" "$RESET"
  printf '  Model:      %s\n' "$model"
  printf '  Capability: %s\n\n' "$capability"

  local failed=0
  local total=0
  local rc

  # Test 1 always runs.
  total=$((total + 1))
  printf '\n=== [1/6] Health + models ===\n'
  run_test 1
  rc=$?
  if [ $rc -ne 0 ]; then failed=$((failed + 1)); fi

  if [ "$capability" = "chat" ]; then
    for n in 2 3 4; do
      total=$((total + 1))
      printf '\n=== [%s/6] ===\n' "$n"
      run_test "$n"
      rc=$?
      if [ $rc -ne 0 ]; then failed=$((failed + 1)); fi
    done
  else
    printf '\n=== Skipping tests 2-4 (chat-only; this server is "%s") ===\n' "$capability"
  fi

  if [ "$capability" = "embeddings" ]; then
    total=$((total + 1))
    printf '\n=== [6/6] Embeddings ===\n'
    run_test 6
    rc=$?
    if [ $rc -ne 0 ]; then failed=$((failed + 1)); fi
  else
    printf '\n=== Skipping test 6 (embeddings-only; this server is "%s") ===\n' "$capability"
  fi

  # Note: test 5 (shutdown) is destructive — only run last in --all mode.
  total=$((total + 1))
  printf '\n=== [5/6] Clean shutdown ===\n'
  run_test 5
  rc=$?
  if [ $rc -ne 0 ]; then failed=$((failed + 1)); fi

  printf '\n%s' "$BOLD"
  if [ $failed -eq 0 ]; then
    printf '%sAll %s tests passed.%s\n' "$GREEN" "$total" "$RESET"
    return 0
  else
    printf '%s%s of %s tests failed.%s\n' "$RED" "$failed" "$total" "$RESET"
    return 1
  fi
}

# Interactive menu.
show_menu() {
  require_server
  local model capability
  model=$(detect_model)
  capability=$(detect_capability "$model")

  while :; do
    printf '\n'
    printf -- '-----------------------------------------------------------\n'
    printf '  %scoldfire-mlx-server v0.1.0 — verification suite%s\n' "$BOLD" "$RESET"
    printf -- '-----------------------------------------------------------\n'
    printf '\n'
    printf '  Detected server: %s\n' "$BASE_URL"
    printf '  Loaded model:    %s\n' "$model"
    printf '  Capability:      %s\n' "$capability"
    printf '\n'
    printf '  Choose:\n'
    printf '    1. Health + models                      (~5 sec)\n'
    printf '    2. Basic chat completion                (~10 sec)\n'
    printf '    3. Streaming completion                 (~10 sec)\n'
    printf '    4. Concurrent KV-isolation              (~30 sec)  *** the bug test\n'
    printf '    5. Clean shutdown (SIGTERM exit < 5s)\n'
    printf '    6. Embeddings (requires embedding server)\n'
    printf '    A. Run all applicable\n'
    printf '    Q. Quit\n'
    printf '\n'
    printf '  > '
    local choice
    if ! read -r choice; then
      printf '\n'
      return 0
    fi
    case "$choice" in
      1) run_test 1 ;;
      2)
        if [ "$capability" != "chat" ]; then
          skip "test 2 requires a chat model (current capability: $capability)"
        else
          run_test 2
        fi
        ;;
      3)
        if [ "$capability" != "chat" ]; then
          skip "test 3 requires a chat model (current capability: $capability)"
        else
          run_test 3
        fi
        ;;
      4)
        if [ "$capability" != "chat" ]; then
          skip "test 4 requires a chat model (current capability: $capability)"
        else
          run_test 4
        fi
        ;;
      5) run_test 5 ;;
      6)
        if [ "$capability" != "embeddings" ]; then
          skip "test 6 requires a separate embedding server (current capability: $capability)"
          info "launch one with: coldfire-mlx-server launch --port 8081 \\"
          info "    --model-path nomic-ai/nomic-embed-text-v1.5 --model-type embeddings"
          info "then re-run with: COLDFIRE_PORT=8081 $0"
        else
          run_test 6
        fi
        ;;
      A|a) run_all ;;
      Q|q|"") return 0 ;;
      *) printf '  Unknown choice: %s\n' "$choice" ;;
    esac
  done
}

case "$MODE" in
  all)
    run_all
    exit $?
    ;;
  single)
    require_server
    run_test "$SINGLE_TEST"
    exit $?
    ;;
  menu)
    show_menu
    ;;
esac
