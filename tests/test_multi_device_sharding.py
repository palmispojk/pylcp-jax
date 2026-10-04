"""
Tests for multi-device sharding logic in integration_tools_gpu.

When <= 1 real GPU is available, this module simulates a multi-GPU node by
setting XLA_FLAGS to create 4 virtual CPU devices.  The sharding, padding,
and GSPMD partitioning logic is identical to real multi-GPU execution — only
the device type differs.

These tests verify that:
  - _shard_batch correctly distributes arrays and pads when needed
  - solve_ivp_random produces identical results whether sharded or not
  - solve_ivp_dense produces identical results whether sharded or not
  - Chunked (batched) execution with sharding gives correct results

The XLA_FLAGS env var must be set *before* JAX initialises.  This is
handled by tests/conftest.py, which creates 4 virtual CPU devices for
the entire test session.  If running standalone without conftest, use::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 pytest tests/test_multi_device_sharding.py
"""

import math
import warnings
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pylcp_jax.integration_tools_gpu as itg
from pylcp_jax.integration_tools_gpu import (
    _gpu_devices,
    _shard_batch,
    solve_ivp_dense,
    solve_ivp_random,
)

# ---------------------------------------------------------------------------
# Detect whether we're running on simulated or real multi-GPU
# ---------------------------------------------------------------------------

_real_gpus = _gpu_devices()
HAS_MULTI_GPU = len(_real_gpus) > 1

# Use real GPUs if available, otherwise fall back to simulated CPU devices.
if HAS_MULTI_GPU:
    DEVICES = _real_gpus
else:
    DEVICES = jax.devices("cpu")
    if len(DEVICES) < 2:
        pytest.skip(
            "Could not create multiple virtual devices — XLA_FLAGS may not "
            "have been set before JAX initialised.",
            allow_module_level=True,
        )
    warnings.warn(
        f"No multi-GPU detected — simulating {len(DEVICES)} devices via "
        f"virtual CPU backends (XLA_FLAGS).  Sharding logic is identical to "
        f"real multi-GPU; only the device platform differs.",
        stacklevel=1,
    )

N_DEVICES = len(DEVICES)


@pytest.fixture(autouse=True)
def _patch_gpu_devices():
    """Make _gpu_devices() return DEVICES so solve_ivp_* use multi-device sharding."""
    with mock.patch.object(itg, "_gpu_devices", return_value=DEVICES):
        yield


# ---------------------------------------------------------------------------
# Shared ODE definitions (same as in test_integration_tools_gpu.py)
# ---------------------------------------------------------------------------


def exp_decay(t, y, args):
    return -y


def harmonic(t, y, args):
    return jnp.array([-y[1], y[0]])


def dummy_random(t, y, dt, key, args=None):
    return y, jnp.int32(0), jnp.float64(dt), key


def always_scatter(t, y, dt, key, args=None):
    key, _ = jax.random.split(key)
    return y, jnp.int32(1), jnp.float64(dt), key


# ---------------------------------------------------------------------------
# TestShardBatch — unit tests for _shard_batch
# ---------------------------------------------------------------------------


class TestShardBatch:
    """Verify _shard_batch padding, shapes, and device placement."""

    def test_single_device_passthrough(self):
        """With 1 device, array is returned unchanged."""
        arr = jnp.ones((5, 3))
        out, orig_N = _shard_batch(arr, DEVICES[:1])
        assert orig_N == 5
        assert out.shape == (5, 3)
        np.testing.assert_array_equal(np.array(out), np.array(arr))

    def test_evenly_divisible_no_padding(self):
        """N divisible by n_devices: no padding needed."""
        N = N_DEVICES * 3
        arr = jnp.arange(N * 2, dtype=jnp.float64).reshape(N, 2)
        out, orig_N = _shard_batch(arr, DEVICES)
        assert orig_N == N
        assert out.shape == (N, 2)

    def test_padding_when_not_divisible(self):
        """N not divisible by n_devices: array is padded with zeros."""
        N = N_DEVICES * 3 + 1
        arr = jnp.ones((N, 2))
        out, orig_N = _shard_batch(arr, DEVICES)
        assert orig_N == N
        padded_N = out.shape[0]
        assert padded_N % N_DEVICES == 0
        assert padded_N >= N
        # Original data preserved.
        np.testing.assert_array_equal(np.array(out[:N]), np.array(arr))
        # Padding is zeros.
        if padded_N > N:
            np.testing.assert_array_equal(np.array(out[N:]), np.zeros((padded_N - N, 2)))

    def test_3d_array(self):
        """Sharding works on higher-dimensional arrays."""
        N = N_DEVICES * 2
        arr = jnp.ones((N, 4, 3))
        out, orig_N = _shard_batch(arr, DEVICES)
        assert orig_N == N
        assert out.shape == (N, 4, 3)

    def test_output_is_sharded(self):
        """Output array should be placed across multiple devices."""
        N = N_DEVICES * 2
        arr = jnp.ones((N, 2))
        out, _ = _shard_batch(arr, DEVICES)
        # A sharded array has a non-trivial sharding attribute.
        sharding = out.sharding
        assert len(sharding.device_set) == N_DEVICES


# ---------------------------------------------------------------------------
# TestMultiDeviceDense — solve_ivp_dense with sharding
# ---------------------------------------------------------------------------


class TestMultiDeviceDense:
    """Verify solve_ivp_dense gives correct results with multi-device sharding.

    The autouse _patch_gpu_devices fixture makes _gpu_devices() return DEVICES,
    so solve_ivp_dense automatically shards across the simulated devices."""

    def test_exponential_decay_accuracy(self):
        """Sharded solve matches analytical solution."""
        N = N_DEVICES * 2
        y0 = jnp.ones((N, 1)) * 2.0
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 2.0], y0, n_points=101)
        expected = 2.0 * jnp.exp(-ts)
        for i in range(N):
            np.testing.assert_allclose(
                np.array(ys[i, :, 0]), np.array(expected), atol=1e-4, err_msg=f"Atom {i} failed"
            )

    def test_matches_single_device(self):
        """Sharded and single-device runs produce identical results."""
        N = N_DEVICES * 2
        y0 = jnp.arange(1, N + 1, dtype=jnp.float64).reshape(N, 1)
        n_points = 51

        # Single-device reference (bypass multi-device by patching back).
        with mock.patch.object(itg, "_gpu_devices", return_value=[]):
            ts_ref, ys_ref = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=n_points)

        # Multi-device (autouse fixture active).
        ts_shard, ys_shard = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=n_points)
        np.testing.assert_allclose(np.array(ts_ref), np.array(ts_shard), atol=1e-12)
        np.testing.assert_allclose(np.array(ys_ref), np.array(ys_shard), atol=1e-10)

    def test_uneven_batch_matches(self):
        """Batch not divisible by n_devices still gives correct results."""
        N = N_DEVICES * 2 + 1  # odd
        y0 = jnp.ones((N, 1)) * 3.0
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=21)
        # Should have exactly N atoms in output (padding stripped).
        assert ys.shape[0] == N
        expected = 3.0 * jnp.exp(-ts)
        for i in range(N):
            np.testing.assert_allclose(
                np.array(ys[i, :, 0]), np.array(expected), atol=1e-4, err_msg=f"Atom {i} failed"
            )

    def test_harmonic_energy_conservation(self):
        """Energy conservation holds across sharded devices."""
        N = N_DEVICES
        y0 = jnp.tile(jnp.array([[0.0, 1.0]]), (N, 1))
        ts, ys = solve_ivp_dense(
            harmonic, [0.0, 4 * jnp.pi], y0, n_points=201, rtol=1e-8, atol=1e-8
        )
        for i in range(N):
            energy = 0.5 * (ys[i, :, 0] ** 2 + ys[i, :, 1] ** 2)
            np.testing.assert_allclose(
                np.array(energy), 0.5, atol=1e-5, err_msg=f"Energy not conserved for atom {i}"
            )


# ---------------------------------------------------------------------------
# TestMultiDeviceRandom — solve_ivp_random with sharding
# ---------------------------------------------------------------------------


class TestMultiDeviceRandom:
    """Verify solve_ivp_random gives correct results with multi-device sharding.

    The autouse _patch_gpu_devices fixture makes _gpu_devices() return DEVICES,
    so solve_ivp_random automatically shards across the simulated devices."""

    def _keys(self, n, seed=42):
        return jax.random.split(jax.random.PRNGKey(seed), n)

    def test_exponential_decay_accuracy(self):
        """Sharded stochastic solve (no scatter) matches analytical solution."""
        N = N_DEVICES * 2
        y0 = jnp.ones((N, 1)) * 2.0
        keys = self._keys(N)
        sols = solve_ivp_random(
            exp_decay,
            dummy_random,
            [0.0, 1.0],
            y0,
            keys,
            n_points=20,
            max_steps=10000,
            rtol=1e-6,
            atol=1e-6,
        )
        for i in range(N):
            y_final = float(sols[i].y[0, -1])
            expected = 2.0 * math.exp(-1.0)
            assert y_final == pytest.approx(expected, rel=1e-3), (
                f"Atom {i}: {y_final} != {expected}"
            )

    def test_matches_single_device(self):
        """Sharded and single-device runs produce identical trajectories."""
        N = N_DEVICES * 2
        seed = 123
        y0 = jnp.arange(1, N + 1, dtype=jnp.float64).reshape(N, 1)
        keys = self._keys(N, seed=seed)

        # Single-device reference (bypass multi-device by patching back).
        with mock.patch.object(itg, "_gpu_devices", return_value=[]):
            sols_ref = solve_ivp_random(
                exp_decay, dummy_random, [0.0, 0.5], y0, keys, n_points=20, max_steps=5000
            )

        # Multi-device (autouse fixture active).
        sols_shard = solve_ivp_random(
            exp_decay, dummy_random, [0.0, 0.5], y0, keys, n_points=20, max_steps=5000
        )

        for i in range(N):
            np.testing.assert_allclose(
                np.array(sols_ref[i].y),
                np.array(sols_shard[i].y),
                atol=1e-10,
                err_msg=f"Atom {i} mismatch between single-device and sharded",
            )

    def test_uneven_batch_all_succeed(self):
        """Batch not divisible by n_devices: all atoms succeed."""
        N = N_DEVICES * 2 + 1
        y0 = jnp.ones((N, 1))
        keys = self._keys(N)
        sols = solve_ivp_random(
            exp_decay, dummy_random, [0.0, 0.5], y0, keys, n_points=20, max_steps=5000
        )
        for i in range(N):
            assert sols[i].success, f"Atom {i} did not succeed"

    def test_scatter_events_with_sharding(self):
        """always_scatter produces events on all sharded atoms."""
        N = N_DEVICES * 2
        y0 = jnp.ones((N, 1))
        keys = self._keys(N)
        sols = solve_ivp_random(
            exp_decay, always_scatter, [0.0, 0.5], y0, keys, n_points=20, max_steps=2000
        )
        for i in range(N):
            assert len(sols[i].t_random) > 0, f"Atom {i} had no scatter events"
            assert jnp.all(sols[i].n_random > 0)

    def test_chunked_sharding(self):
        """Explicit small batch_size forces chunked execution with sharding."""
        N = N_DEVICES * 4
        y0 = jnp.ones((N, 1)) * 2.0
        keys = self._keys(N)

        # Force batch_size smaller than N to trigger chunking.
        sols = solve_ivp_random(
            exp_decay,
            dummy_random,
            [0.0, 1.0],
            y0,
            keys,
            n_points=20,
            max_steps=10000,
            batch_size=N_DEVICES * 2,
        )
        assert len(sols) == N
        for i in range(N):
            y_final = float(sols[i].y[0, -1])
            expected = 2.0 * math.exp(-1.0)
            assert y_final == pytest.approx(expected, rel=1e-2), (
                f"Atom {i}: {y_final} != {expected}"
            )
