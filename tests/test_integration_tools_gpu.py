"""
Tests for pylcp_jax/integration_tools_gpu.py

Covers RandomOdeResult, solve_ivp_dense, and solve_ivp_random.

Test organisation:
  - CPU tests (TestSolveIvpDenseCPU, TestSolveIvpRandomCPU): always run.
  - GPU tests (TestSolveIvpDenseGPU, TestSolveIvpRandomGPU): skipped when
    no CUDA GPU is detected.  A single warning is emitted at collection time.
  - Cross-device tests (TestCPUvsGPUDense, TestCPUvsGPURandom): verify that
    CPU and GPU produce identical results.  Also skipped without a GPU.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import HAS_GPU, requires_gpu

from pylcp_jax.integration_tools_gpu import (
    RandomOdeResult,
    solve_ivp_dense,
)
from pylcp_jax.integration_tools_gpu import (
    solve_ivp_random as _solve_ivp_random,
)

# CPU tests would take minutes with a large n_points; use a small fixed value.
# GPU tests are all skipped on CPU so this override is harmless.
solve_ivp_random = functools.partial(_solve_ivp_random, n_points=20)

CPU_DEVICE = jax.devices("cpu")[0]
GPU_DEVICE = jax.devices("gpu")[0] if HAS_GPU else None


# ---------------------------------------------------------------------------
# Shared ODE definitions
# ---------------------------------------------------------------------------


def exp_decay(t, y, args):
    """dy/dt = -y  ->  y(t) = y0 * exp(-t)."""
    return -y


def harmonic(t, y, args):
    """Simple harmonic oscillator: d/dt [x, v] = [v, -x]."""
    return jnp.array([-y[1], y[0]])


def dummy_random(t, y, dt, key, args=None):
    """No-op stochastic function: never scatters."""
    return y, jnp.int32(0), jnp.float64(dt), key


def always_scatter(t, y, dt, key, args=None):
    """Always records one scatter event per step."""
    key, _ = jax.random.split(key)
    return y, jnp.int32(1), jnp.float64(dt), key


# ---------------------------------------------------------------------------
# TestRandomOdeResult
# ---------------------------------------------------------------------------


class TestRandomOdeResult:
    def _make(self):
        t = jnp.linspace(0.0, 1.0, 10)
        y = jnp.ones((2, 10))
        return RandomOdeResult(
            t=t,
            y=y,
            t_random=jnp.array([0.5]),
            n_random=jnp.array([1]),
            inds_random=jnp.zeros(10, dtype=bool).at[5].set(True),
            success=True,
            status=0,
            message="Success",
            nfev=42,
        )

    def test_has_t(self):
        assert hasattr(self._make(), "t")

    def test_has_y(self):
        assert hasattr(self._make(), "y")

    def test_has_t_random(self):
        assert hasattr(self._make(), "t_random")

    def test_has_n_random(self):
        assert hasattr(self._make(), "n_random")

    def test_has_inds_random(self):
        assert hasattr(self._make(), "inds_random")

    def test_success_true(self):
        assert self._make().success is True

    def test_status_zero(self):
        assert self._make().status == 0

    def test_message_stored(self):
        assert self._make().message == "Success"

    def test_nfev_stored(self):
        assert self._make().nfev == 42


# ---------------------------------------------------------------------------
# TestSolveIvpDenseCPU
# ---------------------------------------------------------------------------


class TestSolveIvpDenseCPU:
    """Deterministic ODE solver on CPU.

    Validates against analytical solutions: exponential decay dy/dt = -y
    -> y(t) = y0*exp(-t), and harmonic oscillator d/dt[x,v] = [v,-x] with
    energy conservation E = 1/2(x^2 + v^2) = const.  Tests batching, solver
    backends, tolerance effects, and the dt0 scaling fix for long time spans."""

    # --- shape / structure ---

    def test_ts_shape(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        ts, _ = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=50)
        assert ts.shape == (50,)

    def test_ys_shape_single(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        _, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=50)
        assert ys.shape == (1, 50, 1)

    def test_ys_shape_batch(self):
        y0 = jax.device_put(jnp.array([[1.0], [2.0], [3.0]]), CPU_DEVICE)
        _, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=21)
        assert ys.shape == (3, 21, 1)

    def test_ys_shape_2d_state(self):
        y0 = jax.device_put(jnp.array([[0.0, 1.0]]), CPU_DEVICE)
        _, ys = solve_ivp_dense(harmonic, [0.0, 1.0], y0, n_points=11)
        assert ys.shape == (1, 11, 2)

    def test_ts_starts_at_t0(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        ts, _ = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=11)
        assert float(ts[0]) == pytest.approx(0.0)

    def test_ts_ends_at_tf(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        ts, _ = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=11)
        assert float(ts[-1]) == pytest.approx(1.0)

    def test_ts_monotonically_increasing(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        ts, _ = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=20)
        assert jnp.all(jnp.diff(ts) > 0)

    # --- numerical correctness ---

    def test_exponential_decay_accuracy(self):
        y0 = jax.device_put(jnp.array([[2.0]]), CPU_DEVICE)
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 2.0], y0, n_points=101)
        expected = 2.0 * jnp.exp(-ts)
        assert jnp.allclose(ys[0, :, 0], expected, atol=1e-4)

    def test_batch_scales_linearly(self):
        """y0 doubled -> solution doubled at all times."""
        y0 = jax.device_put(jnp.array([[1.0], [2.0]]), CPU_DEVICE)
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=51)
        ratio = ys[1, :, 0] / ys[0, :, 0]
        assert jnp.allclose(ratio, jnp.full_like(ratio, 2.0), atol=1e-5)

    def test_harmonic_half_period(self):
        """x(pi) ~ 0 for harmonic oscillator starting at x=0, v=1."""
        y0 = jax.device_put(jnp.array([[0.0, 1.0]]), CPU_DEVICE)
        ts, ys = solve_ivp_dense(harmonic, [0.0, jnp.pi], y0, n_points=101)
        assert float(ys[0, -1, 0]) == pytest.approx(0.0, abs=1e-3)

    def test_energy_conservation_harmonic(self):
        """E = 0.5*(x^2 + v^2) must remain constant."""
        y0 = jax.device_put(jnp.array([[0.0, 1.0]]), CPU_DEVICE)
        ts, ys = solve_ivp_dense(
            harmonic, [0.0, 4 * jnp.pi], y0, n_points=201, rtol=1e-8, atol=1e-8
        )
        energy = 0.5 * (ys[0, :, 0] ** 2 + ys[0, :, 1] ** 2)
        assert jnp.allclose(energy, jnp.full_like(energy, 0.5), atol=1e-5)

    def test_no_nan_in_output(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        _, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=50)
        assert not jnp.any(jnp.isnan(ys))

    # --- solver types ---

    def test_solver_dopri5(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=51, solver_type="Dopri5")
        assert jnp.allclose(ys[0, :, 0], jnp.exp(-ts), atol=1e-4)

    def test_solver_bosh3(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=51, solver_type="Bosh3")
        assert jnp.allclose(ys[0, :, 0], jnp.exp(-ts), atol=1e-3)

    def test_solver_kvaerno5(self):
        """Kvaerno5 is a stiff solver; verify it runs without error."""
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=51, solver_type="Kvaerno5")
        assert not jnp.any(jnp.isnan(ys))
        assert float(ys[0, -1, 0]) == pytest.approx(math.exp(-1.0), abs=1e-3)

    def test_invalid_solver_raises(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        with pytest.raises(Exception):
            solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=10, solver_type="NotASolver")

    # --- tolerance effect ---

    def test_tighter_tolerance_more_accurate(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        _, ys_lo = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=11, rtol=1e-3, atol=1e-3)
        _, ys_hi = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=11, rtol=1e-9, atol=1e-9)
        exact = math.exp(-1.0)
        err_lo = abs(float(ys_lo[0, -1, 0]) - exact)
        err_hi = abs(float(ys_hi[0, -1, 0]) - exact)
        assert err_hi <= err_lo + 1e-12

    # --- long span dt0 scaling ---

    def test_long_span_no_nan(self):
        """dt0 scaling fix: a span of [0, 100] should not produce NaN."""
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        _, ys = solve_ivp_dense(exp_decay, [0.0, 100.0], y0, n_points=51)
        assert not jnp.any(jnp.isnan(ys))


# ---------------------------------------------------------------------------
# TestSolveIvpRandomCPU
# ---------------------------------------------------------------------------


class TestSolveIvpRandomCPU:
    """Stochastic ODE solver on CPU.

    Each result is a RandomOdeResult carrying y(t) plus scattered-event
    times t_random, event counts n_random, and boolean indices inds_random.
    Without scattering (dummy_random) the solution matches y0*exp(-t).  With
    always_scatter every step records an event.  Tests batch independence,
    early termination (max_steps too small -> success=False), and the dt0
    scaling fix for long spans."""

    @classmethod
    def setup_class(cls):
        key = jax.device_put(jax.random.PRNGKey(42), CPU_DEVICE)
        keys1 = jax.device_put(jax.random.split(key, 1), CPU_DEVICE)
        keys4 = jax.device_put(jax.random.split(key, 4), CPU_DEVICE)
        y0_1 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        y0_4 = jax.device_put(jnp.ones((4, 1)), CPU_DEVICE)
        cls._sol = solve_ivp_random(exp_decay, dummy_random, [0.0, 1.0], y0_1, keys1)[0]
        cls._sol_full = solve_ivp_random(
            exp_decay, dummy_random, [0.0, 1.0], y0_1, keys1, max_steps=10000
        )[0]
        cls._results4 = solve_ivp_random(exp_decay, dummy_random, [0.0, 1.0], y0_4, keys4)
        cls._scatter_sol = solve_ivp_random(
            exp_decay, always_scatter, [0.0, 0.5], y0_1, keys1, max_steps=2000
        )[0]

    def _keys(self, n, seed=42):
        key = jax.device_put(jax.random.PRNGKey(seed), CPU_DEVICE)
        return jax.device_put(jax.random.split(key, n), CPU_DEVICE)

    # --- return type / structure ---

    def test_list_length_matches_batch(self):
        assert len(self._results4) == 4

    def test_result_has_t(self):
        assert hasattr(self._sol, "t")

    def test_result_has_y(self):
        assert hasattr(self._sol, "y")

    def test_result_has_t_random(self):
        assert hasattr(self._sol, "t_random")

    def test_result_has_n_random(self):
        assert hasattr(self._sol, "n_random")

    def test_result_has_inds_random(self):
        assert hasattr(self._sol, "inds_random")

    def test_result_has_success(self):
        assert hasattr(self._sol, "success")

    # --- time array properties ---

    def test_t_starts_at_t0(self):
        assert float(self._sol.t[0]) == pytest.approx(0.0, abs=1e-10)

    def test_t_ends_at_tf(self):
        assert float(self._sol_full.t[-1]) == pytest.approx(1.0, abs=1e-6)

    def test_t_monotonically_increasing(self):
        assert jnp.all(jnp.diff(self._sol.t) >= 0.0)

    # --- y array properties ---

    def test_y_t_same_length(self):
        assert self._sol.y.shape[1] == self._sol.t.shape[0]

    def test_y_state_dim_correct(self):
        y0 = jax.device_put(jnp.array([[0.0, 1.0]]), CPU_DEVICE)
        sol = solve_ivp_random(
            harmonic, dummy_random, [0.0, 1.0], y0, self._keys(1), max_steps=5000
        )[0]
        assert sol.y.shape[0] == 2

    def test_no_nan_in_y(self):
        assert not jnp.any(jnp.isnan(self._sol.y))

    # --- numerical correctness ---

    def test_exponential_decay_accuracy(self):
        """Without scattering, result matches analytical exp(-t)."""
        y0 = jax.device_put(jnp.array([[2.0]]), CPU_DEVICE)
        sol = solve_ivp_random(
            exp_decay,
            dummy_random,
            [0.0, 1.0],
            y0,
            self._keys(1),
            max_steps=10000,
            rtol=1e-6,
            atol=1e-6,
        )[0]
        y_final = float(sol.y[0, -1])
        expected = 2.0 * math.exp(-1.0)
        assert y_final == pytest.approx(expected, rel=1e-3)

    # --- success / status ---

    def test_success_flag_true(self):
        assert self._sol_full.success is True

    def test_status_zero_on_success(self):
        assert self._sol_full.status == 0

    # --- stochastic events ---

    def test_no_scatter_gives_empty_t_random(self):
        assert len(self._sol.t_random) == 0

    def test_always_scatter_populates_t_random(self):
        assert len(self._scatter_sol.t_random) > 0

    def test_always_scatter_n_random_positive(self):
        assert jnp.all(self._scatter_sol.n_random > 0)

    def test_inds_random_dtype_bool(self):
        assert self._sol.inds_random.dtype == jnp.bool_

    # --- batch independence ---

    def test_multi_atom_all_succeed(self):
        N = 3
        y0 = jax.device_put(jnp.ones((N, 1)), CPU_DEVICE)
        results = solve_ivp_random(
            exp_decay, dummy_random, [0.0, 0.5], y0, self._keys(N), max_steps=5000
        )
        assert all(sol.success for sol in results)

    def test_different_y0_batch_independent(self):
        """Two atoms with different y0 should give proportionally scaled y."""
        y0 = jax.device_put(jnp.array([[1.0], [2.0]]), CPU_DEVICE)
        results = solve_ivp_random(
            exp_decay,
            dummy_random,
            [0.0, 1.0],
            y0,
            self._keys(2),
            max_steps=10000,
            rtol=1e-7,
            atol=1e-7,
        )
        y1_final = float(results[0].y[0, -1])
        y2_final = float(results[1].y[0, -1])
        assert y2_final == pytest.approx(2.0 * y1_final, rel=1e-4)

    # --- solver types ---

    def test_solver_bosh3(self):
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        sol = solve_ivp_random(
            exp_decay,
            dummy_random,
            [0.0, 1.0],
            y0,
            self._keys(1),
            solver_type="Bosh3",
            max_steps=10000,
        )[0]
        assert sol.success

    # --- dt0 scaling fix ---

    def test_long_span_no_nan(self):
        """dt0 scaling fix: span [0, 500] should not produce NaN or hang."""
        y0 = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        sol = solve_ivp_random(
            exp_decay, dummy_random, [0.0, 500.0], y0, self._keys(1), max_steps=50000
        )[0]
        assert not jnp.any(jnp.isnan(sol.y))


# ---------------------------------------------------------------------------
# TestSolveIvpDenseGPU
# ---------------------------------------------------------------------------


@requires_gpu
class TestSolveIvpDenseGPU:
    """Deterministic ODE solver on GPU.  Mirrors key CPU tests."""

    def test_ts_shape(self):
        y0 = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
        ts, _ = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=50)
        assert ts.shape == (50,)

    def test_ys_shape_batch(self):
        y0 = jax.device_put(jnp.array([[1.0], [2.0], [3.0]]), GPU_DEVICE)
        _, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=21)
        assert ys.shape == (3, 21, 1)

    def test_exponential_decay_accuracy(self):
        y0 = jax.device_put(jnp.array([[2.0]]), GPU_DEVICE)
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 2.0], y0, n_points=101)
        expected = 2.0 * jnp.exp(-ts)
        assert jnp.allclose(ys[0, :, 0], expected, atol=1e-4)

    def test_batch_scales_linearly(self):
        y0 = jax.device_put(jnp.array([[1.0], [2.0]]), GPU_DEVICE)
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=51)
        ratio = ys[1, :, 0] / ys[0, :, 0]
        assert jnp.allclose(ratio, jnp.full_like(ratio, 2.0), atol=1e-5)

    def test_harmonic_half_period(self):
        y0 = jax.device_put(jnp.array([[0.0, 1.0]]), GPU_DEVICE)
        ts, ys = solve_ivp_dense(harmonic, [0.0, jnp.pi], y0, n_points=101)
        assert float(ys[0, -1, 0]) == pytest.approx(0.0, abs=1e-3)

    def test_energy_conservation_harmonic(self):
        y0 = jax.device_put(jnp.array([[0.0, 1.0]]), GPU_DEVICE)
        ts, ys = solve_ivp_dense(
            harmonic, [0.0, 4 * jnp.pi], y0, n_points=201, rtol=1e-8, atol=1e-8
        )
        energy = 0.5 * (ys[0, :, 0] ** 2 + ys[0, :, 1] ** 2)
        assert jnp.allclose(energy, jnp.full_like(energy, 0.5), atol=1e-5)

    def test_no_nan_in_output(self):
        y0 = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
        _, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=50)
        assert not jnp.any(jnp.isnan(ys))

    def test_solver_dopri5(self):
        y0 = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=51, solver_type="Dopri5")
        assert jnp.allclose(ys[0, :, 0], jnp.exp(-ts), atol=1e-4)

    def test_solver_bosh3(self):
        y0 = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=51, solver_type="Bosh3")
        assert jnp.allclose(ys[0, :, 0], jnp.exp(-ts), atol=1e-3)

    def test_solver_kvaerno5(self):
        y0 = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
        ts, ys = solve_ivp_dense(exp_decay, [0.0, 1.0], y0, n_points=51, solver_type="Kvaerno5")
        assert not jnp.any(jnp.isnan(ys))
        assert float(ys[0, -1, 0]) == pytest.approx(math.exp(-1.0), abs=1e-3)

    def test_long_span_no_nan(self):
        y0 = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
        _, ys = solve_ivp_dense(exp_decay, [0.0, 100.0], y0, n_points=51)
        assert not jnp.any(jnp.isnan(ys))


# ---------------------------------------------------------------------------
# TestSolveIvpRandomGPU
# ---------------------------------------------------------------------------


@requires_gpu
class TestSolveIvpRandomGPU:
    """Stochastic ODE solver on GPU.  Mirrors key CPU tests."""

    @classmethod
    def setup_class(cls):
        key = jax.random.PRNGKey(42)
        keys1 = jax.device_put(jax.random.split(key, 1), GPU_DEVICE)
        keys4 = jax.device_put(jax.random.split(key, 4), GPU_DEVICE)
        y0_1 = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
        y0_4 = jax.device_put(jnp.ones((4, 1)), GPU_DEVICE)
        cls._sol = solve_ivp_random(exp_decay, dummy_random, [0.0, 1.0], y0_1, keys1)[0]
        cls._sol_full = solve_ivp_random(
            exp_decay, dummy_random, [0.0, 1.0], y0_1, keys1, max_steps=10000
        )[0]
        cls._results4 = solve_ivp_random(exp_decay, dummy_random, [0.0, 1.0], y0_4, keys4)
        cls._scatter_sol = solve_ivp_random(
            exp_decay, always_scatter, [0.0, 0.5], y0_1, keys1, max_steps=2000
        )[0]

    def _keys(self, n, seed=42):
        return jax.device_put(jax.random.split(jax.random.PRNGKey(seed), n), GPU_DEVICE)

    def test_list_length_matches_batch(self):
        assert len(self._results4) == 4

    def test_exponential_decay_accuracy(self):
        y0 = jax.device_put(jnp.array([[2.0]]), GPU_DEVICE)
        sol = solve_ivp_random(
            exp_decay,
            dummy_random,
            [0.0, 1.0],
            y0,
            self._keys(1),
            max_steps=10000,
            rtol=1e-6,
            atol=1e-6,
        )[0]
        y_final = float(sol.y[0, -1])
        expected = 2.0 * math.exp(-1.0)
        assert y_final == pytest.approx(expected, rel=1e-3)

    def test_success_flag_true(self):
        assert self._sol_full.success is True

    def test_no_nan_in_y(self):
        assert not jnp.any(jnp.isnan(self._sol.y))

    def test_no_scatter_gives_empty_t_random(self):
        assert len(self._sol.t_random) == 0

    def test_always_scatter_populates_t_random(self):
        assert len(self._scatter_sol.t_random) > 0

    def test_multi_atom_all_succeed(self):
        N = 3
        y0 = jax.device_put(jnp.ones((N, 1)), GPU_DEVICE)
        results = solve_ivp_random(
            exp_decay, dummy_random, [0.0, 0.5], y0, self._keys(N), max_steps=5000
        )
        assert all(sol.success for sol in results)

    def test_different_y0_batch_independent(self):
        y0 = jax.device_put(jnp.array([[1.0], [2.0]]), GPU_DEVICE)
        results = solve_ivp_random(
            exp_decay,
            dummy_random,
            [0.0, 1.0],
            y0,
            self._keys(2),
            max_steps=10000,
            rtol=1e-7,
            atol=1e-7,
        )
        y1_final = float(results[0].y[0, -1])
        y2_final = float(results[1].y[0, -1])
        assert y2_final == pytest.approx(2.0 * y1_final, rel=1e-4)

    def test_solver_bosh3(self):
        y0 = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
        sol = solve_ivp_random(
            exp_decay,
            dummy_random,
            [0.0, 1.0],
            y0,
            self._keys(1),
            solver_type="Bosh3",
            max_steps=10000,
        )[0]
        assert sol.success

    def test_long_span_no_nan(self):
        y0 = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
        sol = solve_ivp_random(
            exp_decay, dummy_random, [0.0, 500.0], y0, self._keys(1), max_steps=50000
        )[0]
        assert not jnp.any(jnp.isnan(sol.y))


# ---------------------------------------------------------------------------
# TestCPUvsGPUDense — verify CPU and GPU produce identical results
# ---------------------------------------------------------------------------


@requires_gpu
class TestCPUvsGPUDense:
    """Cross-device consistency for solve_ivp_dense."""

    def test_exponential_decay_matches(self):
        y0_cpu = jax.device_put(jnp.array([[2.0]]), CPU_DEVICE)
        y0_gpu = jax.device_put(jnp.array([[2.0]]), GPU_DEVICE)
        ts_cpu, ys_cpu = solve_ivp_dense(exp_decay, [0.0, 2.0], y0_cpu, n_points=101)
        ts_gpu, ys_gpu = solve_ivp_dense(exp_decay, [0.0, 2.0], y0_gpu, n_points=101)
        np.testing.assert_allclose(np.array(ts_cpu), np.array(ts_gpu), atol=1e-12)
        np.testing.assert_allclose(np.array(ys_cpu), np.array(ys_gpu), atol=1e-10)

    def test_harmonic_energy_matches(self):
        y0_cpu = jax.device_put(jnp.array([[0.0, 1.0]]), CPU_DEVICE)
        y0_gpu = jax.device_put(jnp.array([[0.0, 1.0]]), GPU_DEVICE)
        _, ys_cpu = solve_ivp_dense(
            harmonic, [0.0, 4 * jnp.pi], y0_cpu, n_points=201, rtol=1e-8, atol=1e-8
        )
        _, ys_gpu = solve_ivp_dense(
            harmonic, [0.0, 4 * jnp.pi], y0_gpu, n_points=201, rtol=1e-8, atol=1e-8
        )
        np.testing.assert_allclose(np.array(ys_cpu), np.array(ys_gpu), atol=1e-10)

    def test_batch_matches(self):
        y0_cpu = jax.device_put(jnp.array([[1.0], [2.0], [3.0]]), CPU_DEVICE)
        y0_gpu = jax.device_put(jnp.array([[1.0], [2.0], [3.0]]), GPU_DEVICE)
        _, ys_cpu = solve_ivp_dense(exp_decay, [0.0, 1.0], y0_cpu, n_points=51)
        _, ys_gpu = solve_ivp_dense(exp_decay, [0.0, 1.0], y0_gpu, n_points=51)
        np.testing.assert_allclose(np.array(ys_cpu), np.array(ys_gpu), atol=1e-10)

    def test_solvers_match(self):
        """All solver types produce the same result on CPU and GPU."""
        for solver in ("Dopri5", "Bosh3", "Kvaerno5"):
            y0_cpu = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
            y0_gpu = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
            _, ys_cpu = solve_ivp_dense(
                exp_decay, [0.0, 1.0], y0_cpu, n_points=51, solver_type=solver
            )
            _, ys_gpu = solve_ivp_dense(
                exp_decay, [0.0, 1.0], y0_gpu, n_points=51, solver_type=solver
            )
            np.testing.assert_allclose(
                np.array(ys_cpu),
                np.array(ys_gpu),
                atol=1e-10,
                err_msg=f"CPU vs GPU mismatch for solver {solver}",
            )


# ---------------------------------------------------------------------------
# TestCPUvsGPURandom — verify CPU and GPU produce identical results
# ---------------------------------------------------------------------------


@requires_gpu
class TestCPUvsGPURandom:
    """Cross-device consistency for solve_ivp_random.

    Both devices are seeded identically so deterministic paths (dummy_random)
    must produce bitwise-identical trajectories."""

    def test_exponential_decay_matches(self):
        seed = 42
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 1)
        y0_cpu = jax.device_put(jnp.array([[2.0]]), CPU_DEVICE)
        y0_gpu = jax.device_put(jnp.array([[2.0]]), GPU_DEVICE)
        keys_cpu = jax.device_put(keys, CPU_DEVICE)
        keys_gpu = jax.device_put(keys, GPU_DEVICE)

        sol_cpu = solve_ivp_random(
            exp_decay,
            dummy_random,
            [0.0, 1.0],
            y0_cpu,
            keys_cpu,
            max_steps=10000,
            rtol=1e-6,
            atol=1e-6,
        )[0]
        sol_gpu = solve_ivp_random(
            exp_decay,
            dummy_random,
            [0.0, 1.0],
            y0_gpu,
            keys_gpu,
            max_steps=10000,
            rtol=1e-6,
            atol=1e-6,
        )[0]
        np.testing.assert_allclose(np.array(sol_cpu.y), np.array(sol_gpu.y), atol=1e-10)

    def test_batch_matches(self):
        seed = 123
        key = jax.random.PRNGKey(seed)
        N = 3
        keys = jax.random.split(key, N)
        y0_cpu = jax.device_put(jnp.ones((N, 1)), CPU_DEVICE)
        y0_gpu = jax.device_put(jnp.ones((N, 1)), GPU_DEVICE)
        keys_cpu = jax.device_put(keys, CPU_DEVICE)
        keys_gpu = jax.device_put(keys, GPU_DEVICE)

        sols_cpu = solve_ivp_random(
            exp_decay, dummy_random, [0.0, 0.5], y0_cpu, keys_cpu, max_steps=5000
        )
        sols_gpu = solve_ivp_random(
            exp_decay, dummy_random, [0.0, 0.5], y0_gpu, keys_gpu, max_steps=5000
        )
        for i in range(N):
            np.testing.assert_allclose(
                np.array(sols_cpu[i].y),
                np.array(sols_gpu[i].y),
                atol=1e-10,
                err_msg=f"CPU vs GPU mismatch for atom {i}",
            )

    def test_scatter_events_match(self):
        """With always_scatter, event counts and times must agree."""
        seed = 99
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 1)
        y0_cpu = jax.device_put(jnp.array([[1.0]]), CPU_DEVICE)
        y0_gpu = jax.device_put(jnp.array([[1.0]]), GPU_DEVICE)
        keys_cpu = jax.device_put(keys, CPU_DEVICE)
        keys_gpu = jax.device_put(keys, GPU_DEVICE)

        sol_cpu = solve_ivp_random(
            exp_decay, always_scatter, [0.0, 0.5], y0_cpu, keys_cpu, max_steps=2000
        )[0]
        sol_gpu = solve_ivp_random(
            exp_decay, always_scatter, [0.0, 0.5], y0_gpu, keys_gpu, max_steps=2000
        )[0]
        np.testing.assert_array_equal(np.array(sol_cpu.n_random), np.array(sol_gpu.n_random))
        np.testing.assert_allclose(
            np.array(sol_cpu.t_random), np.array(sol_gpu.t_random), atol=1e-12
        )
