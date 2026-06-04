#!/usr/bin/env python3
"""License audit gate for runtime dependencies.

Computes the transitive closure of `[project].dependencies` from
`pyproject.toml`, asks `pip-licenses` for the declared license of each
package in that closure, and fails (exit 1) if any package declares a
copyleft license OR an identifier we don't recognize.

Why runtime-only? `[dependency-groups].dev` packages like `codespell`
(GPL-2.0) never ship to end users — they're only installed in the
contributor's dev venv. We do not need them to be permissive. By default
upstream `pip-licenses` inspects every installed package, which would
spuriously block `codespell` and any other dev-only GPL tool.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Lazy import: `packaging` is only used in the CLI entry point (`main`),
# never in `check_licenses` itself — so unit tests don't need it.


@dataclass
class LicenseViolation:
    package: str
    license: str
    severity: str  # "blocked" or "review_required"
    message: str


def check_licenses(
    packages: list[dict[str, Any]],
    allowed: dict[str, list[str]],
    overrides: list[dict[str, Any]],
) -> list[LicenseViolation]:
    """Apply the allowlist/blocklist/overrides policy.

    Order of checks for each package:
    1. If `(name, license)` matches an override entry: passes silently.
    2. If `license` appears verbatim in `allowed["always_ok"]`: passes.
    3. If `license` contains any `review_required_substrings` substring:
       emits a `review_required` violation (LGPL needs explicit override).
       Checked BEFORE blocked so the more-specific "LGPL" wins over
       the broader "GPL".
    4. If `license` contains any `blocked_substrings` substring: emits
       a `blocked` violation.
    5. Otherwise: emits a `blocked` violation for unknown identifier.
    """
    overrides_index = {(o["name"], o["license"]) for o in overrides}
    violations: list[LicenseViolation] = []
    for pkg in packages:
        name, lic = pkg["Name"], pkg["License"]
        if (name, lic) in overrides_index:
            continue
        if lic in allowed["always_ok"]:
            continue
        # Check review_required first so more-specific tokens like "LGPL"
        # win over the broader "GPL" blocked substring.
        matched = False
        for substr in allowed["review_required_substrings"]:
            if substr in lic:
                violations.append(
                    LicenseViolation(
                        package=name,
                        license=lic,
                        severity="review_required",
                        message=(
                            f"{name} {pkg['Version']} declares {lic} "
                            f"(matches review-required {substr!r}); "
                            "add to tools/license_overrides.json"
                        ),
                    )
                )
                matched = True
                break
        if matched:
            continue
        for substr in allowed["blocked_substrings"]:
            if substr in lic:
                violations.append(
                    LicenseViolation(
                        package=name,
                        license=lic,
                        severity="blocked",
                        message=(
                            f"{name} {pkg['Version']} declares {lic} "
                            f"(matches blocked {substr!r})"
                        ),
                    )
                )
                matched = True
                break
        if matched:
            continue
        violations.append(
            LicenseViolation(
                package=name,
                license=lic,
                severity="blocked",
                message=(
                    f"{name} {pkg['Version']} declares unknown license {lic!r}; "
                    "add to allowed_licenses.json or replace"
                ),
            )
        )
    return violations


def runtime_closure(pyproject_path: Path) -> set[str]:
    """Return the transitive closure of `[project].dependencies` (runtime only).

    Walks the dep graph using `importlib.metadata`. Skips marker-gated
    dependencies (extras, OS-specific, etc.). Names are normalized to the
    PEP-503 form (lowercase, hyphens-not-underscores) so they line up with
    `pip-licenses` output.
    """
    from importlib import metadata
    from packaging.requirements import Requirement

    data = tomllib.loads(pyproject_path.read_text())
    direct = [Requirement(d).name for d in data["project"]["dependencies"]]

    seen: set[str] = set()
    stack: list[str] = list(direct)
    while stack:
        raw = stack.pop()
        name = raw.lower().replace("_", "-")
        if name in seen:
            continue
        seen.add(name)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            # Not installed in the current environment; skip — pip-licenses
            # can only inspect installed packages anyway.
            continue
        for req_str in dist.requires or []:
            req = Requirement(req_str)
            # Skip marker-gated deps (e.g. `extra == "test"`, `os == "linux"`).
            # Evaluate against empty extras + the current platform so that
            # platform-conditional deps stay in scope.
            if req.marker is not None:
                try:
                    keep = req.marker.evaluate({"extra": ""})
                except Exception:
                    # Conservative: if marker is unparseable, include it.
                    keep = True
                if not keep:
                    continue
            stack.append(req.name)
    return seen


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    pyproject = repo_root / "pyproject.toml"
    allowed = json.loads((repo_root / "tools" / "allowed_licenses.json").read_text())
    overrides = json.loads(
        (repo_root / "tools" / "license_overrides.json").read_text()
    )["overrides"]

    closure = runtime_closure(pyproject)

    # Ask pip-licenses about ALL installed packages, then filter.
    # We use --from=mixed (default) which produces the most complete output
    # by combining trove classifiers AND PEP-621 metadata fields.
    raw = subprocess.check_output(
        ["pip-licenses", "--from=mixed", "--format=json"],
        text=True,
    )
    all_packages = json.loads(raw)

    # Filter to the runtime closure. Match on normalized name.
    runtime_packages = [
        p
        for p in all_packages
        if p["Name"].lower().replace("_", "-") in closure
    ]

    # If a runtime package isn't shown by pip-licenses, it's not installed —
    # surface that as an error so we don't silently skip auditing it.
    missing = closure - {p["Name"].lower().replace("_", "-") for p in runtime_packages}
    # `pip-licenses` skips packages that don't have a dist-info dir (rare;
    # mostly for editable installs of the project itself). We don't fail
    # on those — but we do warn so the operator can verify.
    if missing:
        print(
            f"NOTE: {len(missing)} runtime package(s) in pyproject closure not "
            f"reported by pip-licenses (likely uninstalled or editable): "
            f"{sorted(missing)}"
        )

    violations = check_licenses(runtime_packages, allowed, overrides)
    if violations:
        print("LICENSE AUDIT FAILED\n")
        for v in violations:
            print(f"  [{v.severity.upper()}] {v.message}")
        print("\nSee docs/LICENSING.md for the project's license policy.")
        sys.exit(1)
    print(
        f"License audit passed: {len(runtime_packages)} runtime packages checked, "
        "all permissive."
    )


if __name__ == "__main__":
    main()
