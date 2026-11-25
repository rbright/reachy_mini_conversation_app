default:
    @just --list

# Install project and dev dependencies from uv.lock.
install:
    uv sync --locked --group dev

fmt:
    uv run --locked ruff format .

fmt-check:
    uv run --locked ruff format --check .

lint:
    uv run --locked ruff check .

typecheck:
    uv run --locked mypy

test:
    uv run --locked pytest -q

lock-check:
    uv lock --check

# Run every check that CI gates on.
check: lock-check fmt-check lint typecheck test
