.PHONY: install test test-unit sample lint lock

install:
	uv sync --frozen --extra dev --extra reframe

test:
	uv run --frozen pytest -q

test-unit:
	uv run --frozen pytest -q -m "not integration"

sample:
	uv run --frozen python scripts/make_sample.py sample/podcast

lock:
	uv lock
