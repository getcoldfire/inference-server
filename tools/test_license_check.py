import json
import pytest
from pathlib import Path
from license_check import check_licenses, LicenseViolation

ALLOWED = json.loads(Path("tools/allowed_licenses.json").read_text())

def test_mit_pkg_passes():
    pkgs = [{"Name": "requests", "Version": "2.31.0", "License": "Apache-2.0"}]
    assert check_licenses(pkgs, ALLOWED, overrides=[]) == []

def test_gpl_pkg_blocked():
    pkgs = [{"Name": "mlx-embeddings", "Version": "0.0.5", "License": "GPL-3.0"}]
    violations = check_licenses(pkgs, ALLOWED, overrides=[])
    assert len(violations) == 1
    assert violations[0].severity == "blocked"
    assert "mlx-embeddings" in violations[0].message

def test_lgpl_pkg_requires_review():
    pkgs = [{"Name": "soxr", "Version": "0.5.0", "License": "LGPL-2.1-or-later"}]
    violations = check_licenses(pkgs, ALLOWED, overrides=[])
    assert len(violations) == 1
    assert violations[0].severity == "review_required"

def test_lgpl_pkg_with_override_passes():
    pkgs = [{"Name": "soxr", "Version": "0.5.0", "License": "LGPL-2.1-or-later"}]
    overrides = [{"name": "soxr", "license": "LGPL-2.1-or-later", "reviewer": "test", "date": "2026-06-03", "rationale": "needed transitive"}]
    assert check_licenses(pkgs, ALLOWED, overrides=overrides) == []

def test_unknown_license_blocks():
    pkgs = [{"Name": "weirdpkg", "Version": "1.0", "License": "Custom-EULA"}]
    violations = check_licenses(pkgs, ALLOWED, overrides=[])
    assert violations[0].severity == "blocked"
