#!/usr/bin/env python3
"""License audit gate. Fails CI if any dep declares a copyleft license."""
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LicenseViolation:
    package: str
    license: str
    severity: str  # "blocked" or "review_required"
    message: str

def check_licenses(packages, allowed, overrides):
    overrides_index = {(o["name"], o["license"]) for o in overrides}
    violations = []
    for pkg in packages:
        name, lic = pkg["Name"], pkg["License"]
        if (name, lic) in overrides_index:
            continue
        if lic in allowed["always_ok"]:
            continue
        # Check review_required first so more-specific tokens like "LGPL" win
        # over the broader "GPL" blocked substring.
        for substr in allowed["review_required_substrings"]:
            if substr in lic:
                violations.append(LicenseViolation(
                    package=name, license=lic, severity="review_required",
                    message=f"{name} {pkg['Version']} declares {lic} (matches review-required '{substr}'); add to tools/license_overrides.json"
                ))
                break
        else:
            for substr in allowed["blocked_substrings"]:
                if substr in lic:
                    violations.append(LicenseViolation(
                        package=name, license=lic, severity="blocked",
                        message=f"{name} {pkg['Version']} declares {lic} (matches blocked '{substr}')"
                    ))
                    break
            else:
                violations.append(LicenseViolation(
                    package=name, license=lic, severity="blocked",
                    message=f"{name} {pkg['Version']} declares unknown license {lic!r}; add to allowed_licenses.json or replace"
                ))
    return violations

def main():
    allowed = json.loads(Path("tools/allowed_licenses.json").read_text())
    overrides = json.loads(Path("tools/license_overrides.json").read_text())["overrides"]
    raw = subprocess.check_output(
        ["pip-licenses", "--from=mixed", "--format=json"],
        text=True,
    )
    packages = json.loads(raw)
    violations = check_licenses(packages, allowed, overrides)
    if violations:
        print("LICENSE AUDIT FAILED\n")
        for v in violations:
            print(f"  [{v.severity.upper()}] {v.message}")
        print("\nSee docs/LICENSING.md for the project's license policy.")
        sys.exit(1)
    print(f"License audit passed: {len(packages)} packages checked, all permissive.")

if __name__ == "__main__":
    main()
