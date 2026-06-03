# License Policy

This project is MIT-licensed. To preserve that, every direct and transitive runtime
dependency must declare a permissive license. The CI gate at `tools/license_check.py`
runs on every PR and blocks merges that introduce copyleft.

## Allowed
MIT, BSD (2/3-clause), Apache-2.0, ISC, PSF, Zlib, CC0, Unlicense, HPND, MPL-2.0.

## Blocked
GPL (any), AGPL, SSPL, CC-BY-NC, Commons Clause. **No exceptions.**

## Review-required
LGPL. Adding an LGPL dep requires explicit approval in `tools/license_overrides.json`
with a written rationale. No overrides exist today.

## Adding a dependency
1. Add to `pyproject.toml` and recompile `requirements.lock` via `pip-compile`.
2. Run `python tools/license_check.py` locally.
3. If a new license appears, either add it to `tools/allowed_licenses.json` (if it's
   already-allowed but the exact identifier wasn't matched) or replace the dep.

## Why
Coldfire distributes this software to contributor-owned Macs at scale via Homebrew.
GPL contamination would force the entire Coldfire CLI to GPL — incompatible with the
commercial distribution model. See `docs/superpowers/specs/2026-05-31-mlx-openai-server-license-audit.md`
in the private Coldfire monorepo for the full audit.
