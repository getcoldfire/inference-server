# Releasing `coldfire-inference-server`

This is the operator-facing release runbook. Anything that needs to happen between "main is green" and "users can `brew upgrade`" lives here.

The pipeline has two halves:
1. **Repo side** (this repo) — tag, push, CI builds the GitHub Release with `NOTICES.txt` and `requirements.lock` attached.
2. **Tap side** (`getcoldfire/homebrew-coldfire`) — bump the formula by hand (URL, sha256, version, and the `NOTICES.txt` resource sha256).

Both halves are required. If you only do half 1, users won't see the new version because they install via Homebrew.

---

## 1. Pre-release (local)

All steps run from a clean checkout of this repo with the dev venv active.

### 1a. Bump the version

Edit `pyproject.toml`:

```toml
[project]
version = "0.X.Y"   # was 0.A.B
```

Match Semantic Versioning. We're pre-1.0, so:
- patch bump → bugfixes, no API changes
- minor bump → new feature, additive endpoint, new embedding family
- major bump → not yet

### 1b. Update `CHANGELOG.md`

Append a new top-level section dated to today, summarising user-visible changes. Reference issue/PR numbers where relevant. Keep it short — the GitHub Release notes get auto-generated from the commit log between tags, this file is the curated narrative.

### 1c. Diff `NOTICES.txt` vs. the last release

Catches new transitive dependencies *before* the release workflow blocks on them:

```bash
bash tools/generate_notices.sh
diff NOTICES.txt <(curl -sL https://github.com/getcoldfire/inference-server/releases/download/v<prev>/NOTICES.txt)
```

Any new dependency that appears here gets a row in the diff. Inspect it. If a new dep landed in `requirements.lock` since last release, check:
- License is permissive (MIT/BSD/Apache-2/ISC/PSF/MPL-2 are fine; anything copyleft fails the gate)
- It's actually used and not pulled in by accident

### 1d. Run the full release gate

```bash
make release-check
```

This runs:
- `make lint` — ruff + mypy
- `make test` — unit tests (must be all green or matching the documented baseline)
- `make license-check` — `tools/license_check.py` against `tools/allowed_licenses.json`
- `make test-smoke` — short integration suite that boots a real server (macOS-only)
- `make test-soak` — the 1.5h+ soak suite (1h streaming, 100 sequential, 8-stream concurrent, etc.)

The soak suite is the long pole — block out time for it. CI does not run it; this is the only safety net before tagging.

For dev iteration, `make release-check-quick` skips the soak. Don't tag on quick alone.

### 1e. Commit the version bump + changelog

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "release: v0.X.Y"
git push origin main
```

Wait for CI (`ci.yml` + `linters.yml`) to go green on the commit before tagging.

## 2. Release (CI-driven)

### 2a. Tag and push

```bash
git tag v0.X.Y
git push origin v0.X.Y
```

That triggers `.github/workflows/release.yml` on a macOS runner. The workflow:

1. Creates a venv on the runner, installs `requirements.lock` + `pip-licenses` + the package.
2. Runs `tools/generate_notices.sh` to produce `NOTICES.txt`.
3. Runs `make license-check` as the release gate — fails the workflow if a copyleft license slipped in since the last CI run.
4. Creates a GitHub Release for the tag, attaches `NOTICES.txt` and `requirements.lock`, and auto-generates release notes from the commit log since the previous tag.

### 2b. Verify the GitHub Release

Open the release page in the browser. Check:

- `NOTICES.txt` is attached and downloads cleanly
- `requirements.lock` is attached
- Auto-generated notes look sensible
- The source tarball (`.tar.gz`) is listed under "Assets" (GitHub generates this automatically from the tag)

If anything is wrong — wrong tag, missing asset, license gate failure surfaced too late — delete the GitHub Release **and** the tag (locally and on origin), fix the underlying issue, and re-tag.

## 3. Homebrew bump (manual)

The release workflow does NOT touch the tap. This step is by hand.

### 3a. Fetch SHA256 for the source tarball

```bash
curl -sL \
  "https://github.com/getcoldfire/inference-server/archive/refs/tags/v0.X.Y.tar.gz" \
  -o /tmp/coldfire-inference-server-v0.X.Y.tar.gz
shasum -a 256 /tmp/coldfire-inference-server-v0.X.Y.tar.gz
```

### 3b. Fetch SHA256 for the NOTICES.txt release asset

```bash
curl -sL \
  "https://github.com/getcoldfire/inference-server/releases/download/v0.X.Y/NOTICES.txt" \
  -o /tmp/NOTICES-v0.X.Y.txt
shasum -a 256 /tmp/NOTICES-v0.X.Y.txt
```

### 3c. Edit `Formula/coldfire-inference-server.rb`

In the [`homebrew-coldfire`](https://github.com/getcoldfire/homebrew-coldfire) tap, update four fields:

```ruby
url "https://github.com/getcoldfire/inference-server/archive/refs/tags/v0.X.Y.tar.gz"
sha256 "<source tarball sha256 from 3a>"

resource "notices" do
  url "https://github.com/getcoldfire/inference-server/releases/download/v0.X.Y/NOTICES.txt"
  sha256 "<notices sha256 from 3b>"
end
```

The formula does not carry a separate `version` field — Homebrew derives it from the `url` tag.

### 3d. PR + merge

```bash
cd /path/to/homebrew-coldfire
git checkout -b bump-coldfire-inference-server-v0.X.Y
git add Formula/coldfire-inference-server.rb
git commit -m "coldfire-inference-server v0.X.Y"
git push origin bump-coldfire-inference-server-v0.X.Y
gh pr create --fill
```

Merge once CI is happy. Homebrew tap auto-audit catches the most common errors (mismatched sha, broken URL).

## 4. User upgrade path

After the tap PR merges, users get the new version with:

```bash
brew update
brew upgrade coldfire-inference-server
```

`brew update` is required — without it, Homebrew uses the cached formula and never sees the new tag.

A fresh install picks up the latest formula automatically:

```bash
brew tap getcoldfire/coldfire
brew install coldfire-inference-server
```

## 5. Troubleshooting

### `jiter` warning on install

Homebrew prints a benign warning during install along the lines of:

```
Warning: Failed to detect dylib id ... for .../jiter/jiter.abi3.so
```

This is harmless. `jiter` is a binary wheel that ships a `.so` Homebrew tries to relocate using `install_name_tool`; the dylib has no id set, so the rewrite is a no-op. Nothing to fix.

### Release workflow fails at "License gate"

A new transitive dependency picked up a non-allowlisted license. Look at the workflow log — `tools/license_check.py` prints exactly which package + which license. Options:
- If the license is actually fine but new (e.g. `Apache 2.0` not yet in the allowlist), add it to `tools/allowed_licenses.json`, commit, re-tag.
- If the license is a real copyleft hit, the dependency cannot ship. Pin to a prior version or remove.

### The release ran but `coldfire-inference-server --licenses` prints "NOTICES.txt not bundled in this build"

The NOTICES.txt resource sha256 in the formula is stale or the resource block was forgotten on the tap bump. The CLI looks for the file at `share/doc/coldfire-inference-server/NOTICES.txt` (the path the formula's `install` block writes to). Re-bump the formula with the correct sha and a clean reinstall (`brew reinstall coldfire-inference-server`).

### Soak tests are flaky on my local machine

`make test-soak` is sensitive to background load. Close other Metal-using apps before running it (Chrome with WebGL tabs is the usual culprit). If a single test fails intermittently and the rest pass, you can re-run just that file (`pytest tests/slow/test_X.py -m slow`) before deciding whether it's a real regression.
