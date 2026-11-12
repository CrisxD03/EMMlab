# emmlab task runner (make replaces `just`, which is not installed here).
# pnpm store / caches live in TMPDIR so sandboxed shells can install packages.
PNPM ?= pnpm
PNPM_FLAGS ?= --store-dir $(or $(TMPDIR),/tmp)/pnpm-store
WEB := web

.PHONY: help sync check check-all py-check web-check e2e build serve smoke test test-slow

help:
	@echo "make sync       install Python (uv) and web (pnpm) dependencies"
	@echo "make check      G1: ruff, basedpyright, fast pytest, web tsc+biome+vitest"
	@echo "make check-all  check + slow statistical tests"
	@echo "make e2e        G2: Playwright end-to-end (needs local port binding + Chrome)"
	@echo "make build      build the web app into src/emmlab/server/static"
	@echo "make serve      run the app at http://127.0.0.1:8765"
	@echo "make smoke      G3: clean-clone quickstart"

sync:
	uv sync
	@if [ -d $(WEB) ]; then $(PNPM) -C $(WEB) install $(PNPM_FLAGS); fi

py-check:
	uv run ruff check src tests
	uv run basedpyright
	uv run pytest -q

web-check:
	@if [ -d $(WEB) ]; then $(PNPM) -C $(WEB) run check && $(PNPM) -C $(WEB) test; fi

check: py-check web-check

test:
	uv run pytest -q

test-slow:
	uv run pytest -q -m slow

check-all: check test-slow

build:
	$(PNPM) -C $(WEB) run build

serve: build
	uv run lab serve --port 8765

e2e: build
	$(PNPM) -C $(WEB) run e2e

smoke:
	bash scripts/smoke_clean_clone.sh
