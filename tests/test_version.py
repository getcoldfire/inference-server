"""Regression tests for ``app.version``.

These exist because v0.4.0 shipped with ``PACKAGE_NAME`` set to the
upstream Python distribution name (``mlx-openai-server``), which no
longer matched the renamed pyproject ``[project].name``. The
brew-installed venv contained no distribution under that name, so
``importlib.metadata.version(PACKAGE_NAME)`` raised
``PackageNotFoundError`` and the fallback to read ``pyproject.toml``
also failed (pyproject is not packaged into site-packages). Net effect:
every ``import app`` crashed.

The tests below pin ``PACKAGE_NAME`` to the pyproject project name so
future renames cannot silently regress.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from app.version import PACKAGE_NAME


def _pyproject_project() -> dict:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        return tomllib.load(handle)["project"]


def test_package_name_matches_pyproject_project_name() -> None:
    assert PACKAGE_NAME == _pyproject_project()["name"]
