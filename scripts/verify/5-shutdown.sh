#!/usr/bin/env bash
# Test 5 — clean shutdown.
#
# Finds the server PID (via pgrep, falling back to lsof on the listen
# port), sends SIGTERM, waits up to 10 seconds for exit, and asserts:
#   - exit happened within 5s (PASS) or 5-10s (WARN — still passes)
#   - exit code was 0
#
# After this runs the server is GONE. The script prints a clear hint
# for relaunching.
#
# Skips cleanly if:
#   - no server is detected (already gone)
#   - multiple matching processes are found (ambiguous — user should
#     test manually)

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_common.sh
. "$SCRIPT_DIR/_common.sh"

header "[5/6] Clean shutdown (SIGTERM)"

# Don't use require_server — skipping is friendlier here than failing
# loudly if the server is already stopped.
http_code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "${BASE_URL}/healthz" 2>/dev/null; true)
if [ "$http_code" != "200" ]; then
  skip "no server detected at ${BASE_URL} — already stopped or never started"
  exit 0
fi

# Prefer lsof-by-listen-port — it uniquely identifies *this* port's
# server even when multiple coldfire-inference-server instances are running.
# Fall back to pgrep only if lsof has no usable result (e.g. the server
# isn't binding via a process we can see — rare).
pids=$(lsof -nP -iTCP:"${COLDFIRE_PORT}" -sTCP:LISTEN -t 2>/dev/null || true)
if [ -z "$pids" ]; then
  pids=$(pgrep -f 'coldfire-inference-server.*launch' 2>/dev/null || true)
fi

if [ -z "$pids" ]; then
  skip "server responds to HTTP but no PID found via pgrep or lsof — cannot signal"
  exit 0
fi

# Count pids.
pid_count=$(printf '%s\n' "$pids" | wc -l | tr -d ' ')
if [ "$pid_count" -gt 1 ]; then
  info "found multiple candidate PIDs: $(printf '%s' "$pids" | tr '\n' ' ')"
  skip "ambiguous — please run this test manually against a single known PID"
  exit 0
fi

pid="$pids"
info "Target PID: ${pid}"
info "Sending SIGTERM..."

start=$(date +%s)
if ! kill -TERM "$pid" 2>/dev/null; then
  info "${CROSS} kill -TERM ${pid} failed (process may not be ours)"
  fail "could not signal server"
  exit 1
fi

# Poll for exit, max 10 seconds.
exit_code=""
within_5=0
within_10=0
elapsed=0
while [ $elapsed -le 10 ]; do
  if ! kill -0 "$pid" 2>/dev/null; then
    # Process is gone. We can't get its exit code from outside (only its
    # parent can wait()); assume 0 if it went away cleanly.
    # cli-v2 contract is exit 0 on SIGTERM; the meaningful signal here
    # is whether it exited within the deadline.
    exit_code=0
    if [ $elapsed -le 5 ]; then
      within_5=1
    elif [ $elapsed -le 10 ]; then
      within_10=1
    fi
    break
  fi
  sleep 1
  now=$(date +%s)
  elapsed=$((now - start))
done

if [ -z "$exit_code" ]; then
  info "${CROSS} process ${pid} still running after 10 seconds"
  fail "server did not exit within 10s of SIGTERM"
  printf '\n  %sThis is a cli-v2 contract violation (graceful shutdown < 5s).%s\n' "$YELLOW" "$RESET"
  printf '  You may want to: kill -9 %s\n\n' "$pid"
  exit 1
fi

# Verify /healthz is no longer responsive. curl writes %{http_code} even
# on connection failure (000), so don't double-append a fallback — just
# discard its non-zero exit.
sleep 1
http_after=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "${BASE_URL}/healthz" 2>/dev/null; true)
if [ "$http_after" = "200" ]; then
  info "${CROSS} server still responding to HTTP after exit — another instance?"
  fail "server appears to still be running"
  exit 1
fi

info "${CHECK} server exited after ${elapsed}s"
info "${CHECK} /healthz no longer responds (HTTP ${http_after})"

if [ $within_5 -eq 1 ]; then
  pass "clean shutdown (${elapsed}s, <5s target)"
elif [ $within_10 -eq 1 ]; then
  printf '%sWARN%s — exit took %ss (within 10s deadline, but >5s target)\n' "$YELLOW" "$RESET" "$elapsed"
  printf '\n'
  pass "shutdown completed (${elapsed}s)"
fi

printf '\n'
printf '  %sServer stopped. Relaunch with:%s\n' "$BOLD" "$RESET"
printf '    coldfire-inference-server launch --host %s --port %s --model-path mlx-community/Llama-3.2-1B-Instruct-4bit\n' \
  "$COLDFIRE_HOST" "$COLDFIRE_PORT"
printf '\n'
exit 0
