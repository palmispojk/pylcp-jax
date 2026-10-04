"""Pytest conftest — runs before any test module is imported.

- Enables 64-bit floating point in JAX (jax_enable_x64) so that all
  tests run with double precision.
- Sets XLA_FLAGS to create 4 virtual CPU devices so that multi-device
  sharding tests (test_multi_device_sharding.py) work even when collected
  alongside other test files.  This has no effect on single-device tests;
  they continue to use jax.devices('cpu')[0] as before.
- Provides shared fixtures (zero_B, single_beam, symmetric_beams, ham)
  used across multiple test modules.

The XLA flag is only added when no --xla_force_host_platform_device_count
is already present in XLA_FLAGS, so user-specified values are respected.
"""

import os
import warnings

# XLA_FLAGS must be set before JAX is imported.
_existing = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _existing:
    os.environ["XLA_FLAGS"] = f"{_existing} --xla_force_host_platform_device_count=4".strip()

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import pylcp_jax.hamiltonians as hamiltonians
from pylcp_jax.fields import constantMagneticField, laserBeams
from pylcp_jax.hamiltonian import hamiltonian

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

HAS_GPU = jax.default_backend() == "gpu"
if not HAS_GPU:
    warnings.warn(
        "No CUDA GPU detected by JAX — all GPU tests will be skipped. "
        "Install jax[cuda12] to enable them.",
        stacklevel=1,
    )

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="No GPU available")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_ham(gamma=1.0, k=1.0, mass=1.0):
    """Minimal F=0 -> F'=1 Hamiltonian (1 ground + 3 excited states)."""
    H0_g, mu_g = hamiltonians.singleF(F=0, gF=0)
    H0_e, mu_e = hamiltonians.singleF(F=1, gF=1)
    d_q = hamiltonians.dqij_two_bare_hyperfine(0, 1)
    return hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q, mass=mass, gamma=gamma, k=k)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zero_B():
    return constantMagneticField(jnp.array([0.0, 0.0, 0.0]))


@pytest.fixture
def single_beam():
    """One sigma+ beam along +z, on resonance, weak saturation."""
    return laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.1, "delta": 0.0}])


@pytest.fixture
def symmetric_beams():
    """Two counter-propagating sigma+/sigma- beams along z, equal intensity."""
    return laserBeams(
        [
            {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.5, "delta": -1.0},
            {"kvec": [0.0, 0.0, -1.0], "pol": -1, "s": 0.5, "delta": -1.0},
        ]
    )


@pytest.fixture
def ham():
    return make_ham()
