#!/bin/bash
#
# generate_notices.sh — write NOTICES.txt for the release tarball.
#
# Called from .github/workflows/release.yml after the venv exists and
# pip-licenses is installed. Emits a vertical plain-text listing with the
# full license body per package, which is what we ship in the Homebrew
# bottle at $(formula_prefix)/share/doc/coldfire-inference-server/NOTICES.txt
# and what `coldfire-inference-server --licenses` reads at runtime.
#
# pip-licenses is invoked through the venv binary so this works whether
# or not the caller has activated the venv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Defensive — pip-licenses should already be installed by the workflow,
# but install on demand so the script also works locally.
.venv/bin/pip install pip-licenses --quiet

.venv/bin/pip-licenses \
  --from=mixed \
  --format=plain-vertical \
  --with-license-file \
  > NOTICES.txt

echo "Wrote NOTICES.txt ($(wc -l < NOTICES.txt) lines)"
