Installation instructions
=========================

Prerequisites
-------------

``pylcp-jax`` requires Python >= 3.11 (tested on 3.11 to 3.14). It uses
`JAX <https://github.com/jax-ml/jax>`_ and
`Diffrax <https://github.com/patrick-kidger/diffrax>`_ as its numerical backend.
The package is installed as ``pylcp-jax`` and imported as ``pylcp_jax``.

Install from PyPI
-----------------

::

  pip install pylcp-jax

``pylcp-jax`` is a separate package from the original ``pylcp``; it does not
replace or conflict with it.

Install from source
-------------------

Using `uv <https://docs.astral.sh/uv/>`_ (recommended)::

  git clone https://github.com/palmispojk/pylcp-jax/
  cd pylcp-jax
  uv sync

Or using pip::

  git clone https://github.com/palmispojk/pylcp-jax/
  cd pylcp-jax
  pip install .

GPU support
-----------

For GPU-accelerated simulations (requires a CUDA-capable GPU):

Using uv::

  uv sync --extra cuda

Using pip::

  pip install "pylcp-jax[cuda]"

or, from a source checkout, ``pip install ".[cuda]"``.

This installs JAX with CUDA 12 support. GPU tests in the test suite are
automatically skipped when no GPU is detected.

Development setup
-----------------

To install development and documentation dependencies::

  uv sync --group dev --group docs

Run the test suite with::

  uv run pytest

See `CONTRIBUTING.md <https://github.com/palmispojk/pylcp-jax/blob/master/CONTRIBUTING.md>`_
for code style and linting instructions.