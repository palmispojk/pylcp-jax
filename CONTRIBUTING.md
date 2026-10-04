# Contributing to pylcp_jax

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
```

## Code quality

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [pyright](https://github.com/microsoft/pyright) for type checking.

### Linting

```bash
uv run ruff check pylcp_jax
```

### Formatting

```bash
uv run ruff format pylcp_jax
```

All code should be formatted with `ruff format` before committing. The formatter uses a line length of 100 (configured in `pyproject.toml`).

### Type checking

```bash
uv run pyright
```

Pyright is configured in basic mode and checks the `pylcp_jax/` directory. The `gratings` module is excluded from type checking due to an optional `numba` dependency.

## Running tests

```bash
uv run pytest
```

GPU tests are skipped automatically when no CUDA device is available.

## Style notes

- **Line length**: 100 characters.
- **Imports**: sorted by `ruff` (isort rules). `jax.config.update()` calls before `jax.numpy` imports are expected and exempt from import-order checks.
- **Variable names**: single-letter names like `l`, `I`, `O` are standard physics notation and allowed.
- **Docstrings**: NumPy style. Use `r"""` for docstrings containing LaTeX math (e.g. `\rho`, `\mu`).
- **Lambdas**: short lambda expressions are acceptable for simple physics formulas.