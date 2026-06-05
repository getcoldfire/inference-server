# Makefile for coldfire-mlx-server.
#
# All targets invoke tools from the local .venv so callers don't need to
# `source` anything. Override PYTHON=python on CI runners that don't use a
# repo-local venv: `make PYTHON=python lint`.
PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy

# Put the venv on PATH so tools/license_check.py finds `pip-licenses`
# (it spawns it via subprocess). Targets that override PYTHON= can also
# override VENV_BIN= to point elsewhere.
VENV_BIN ?= $(CURDIR)/.venv/bin
export PATH := $(VENV_BIN):$(PATH)

.PHONY: run install lint test test-smoke test-soak license-check release-check-quick release-check

run:
	mlx-server launch \
	--model-path mlx-community/Qwen3-1.7B-4bit \
	--model-type lm \
	--max-concurrency 1 \
	--queue-timeout 300 \
	--queue-size 100

install:
	$(PYTHON) -m pip install -e .

lint:
	$(RUFF) check .
	$(RUFF) format --check .
	$(MYPY) app/ tools/

test:
	$(PYTEST) tests/ -m "not integration and not slow and not smoke" -v

test-smoke:
	$(PYTEST) tests/integration/ -m smoke -v

test-soak:
	$(PYTEST) tests/slow/ -m slow -v

license-check:
	$(PYTHON) tools/license_check.py

# release-check-quick: all gates except the 1.5h soak. Use during dev iteration.
release-check-quick: lint test license-check test-smoke
	@echo "Quick release gates passed. Run 'make release-check' for the full pre-tag gate including 1.5h soak."

# release-check: full pre-tag gate; runs the 1.5h+ soak suite. Use ONLY before tagging.
release-check: release-check-quick test-soak
	@echo "Release check passed — safe to tag."
