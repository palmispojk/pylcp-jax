"""
Tests for pylcp_jax/governingeq.py
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pylcp_jax.fields import constantMagneticField, laserBeam, laserBeams
from pylcp_jax.governingeq import governingeq

# ---------------------------------------------------------------------------
# Minimal concrete subclasses
# ---------------------------------------------------------------------------


class _LinearForceGovEq(governingeq):
    """F = -k*r  (restoring — simple harmonic trap)."""

    def __init__(self, laserBeams_arg, magField_arg, k=1.0, mass=1.0, **kwargs):
        super().__init__(laserBeams_arg, magField_arg, **kwargs)
        self.k = k
        self.mass = mass

    def find_equilibrium_force(self, **kwargs):
        return -self.k * self.r0


class _VelocityDampedGovEq(governingeq):
    """F = -k*r - b*v  (restoring + velocity damping)."""

    def __init__(self, laserBeams_arg, magField_arg, k=1.0, beta=2.0, mass=1.0, **kwargs):
        super().__init__(laserBeams_arg, magField_arg, **kwargs)
        self.k = k
        self._beta = beta
        self.mass = mass

    def find_equilibrium_force(self, **kwargs):
        return -self.k * self.r0 - self._beta * self.v0


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_beams():
    return laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": -2.0}])


@pytest.fixture
def zero_magfield():
    return constantMagneticField(jnp.array([0.0, 0.0, 0.0]))


@pytest.fixture
def linear_geq(simple_beams, zero_magfield):
    return _LinearForceGovEq(simple_beams, zero_magfield, k=1.0, mass=1.0)


@pytest.fixture
def damped_geq(simple_beams, zero_magfield):
    return _VelocityDampedGovEq(simple_beams, zero_magfield, k=1.0, beta=2.0, mass=1.0)


# ---------------------------------------------------------------------------
# TestGoverningEqInit
# ---------------------------------------------------------------------------


class TestGoverningEqInit:
    def test_list_of_beams_stored_under_g_to_e(self, zero_magfield):
        beam_list = [laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)]
        geq = _LinearForceGovEq(beam_list, zero_magfield)
        assert "g->e" in geq.laserBeams
        assert geq.laserBeams["g->e"].num_of_beams == 1

    def test_laserbeams_object_stored_under_g_to_e(self, simple_beams, zero_magfield):
        geq = _LinearForceGovEq(simple_beams, zero_magfield)
        assert "g->e" in geq.laserBeams
        assert geq.laserBeams["g->e"].num_of_beams == 1

    def test_dict_of_laserbeams_stored(self, simple_beams, zero_magfield):
        lb_dict = {"g->e": simple_beams}
        geq = _LinearForceGovEq(lb_dict, zero_magfield)
        assert "g->e" in geq.laserBeams

    def test_invalid_laserbeams_type_raises(self, zero_magfield):
        with pytest.raises(TypeError):
            _LinearForceGovEq("bad_input", zero_magfield)

    def test_invalid_dict_value_raises(self, zero_magfield):
        with pytest.raises(TypeError):
            _LinearForceGovEq({"g->e": "not_a_laserbeams"}, zero_magfield)

    def test_callable_magfield_wrapped(self, simple_beams):
        geq = _LinearForceGovEq(simple_beams, lambda R, t: jnp.array([0.0, 0.0, 0.0]))
        assert hasattr(geq.magField, "Field")

    def test_magfield_object_stored(self, simple_beams, zero_magfield):
        geq = _LinearForceGovEq(simple_beams, zero_magfield)
        assert hasattr(geq.magField, "Field")

    def test_array_magfield_wrapped(self, simple_beams):
        geq = _LinearForceGovEq(simple_beams, np.array([0.0, 1.0, 0.0]))
        assert hasattr(geq.magField, "Field")

    def test_invalid_magfield_type_raises(self, simple_beams):
        with pytest.raises(TypeError):
            _LinearForceGovEq(simple_beams, "bad_field")

    def test_wrong_accel_size_raises(self, simple_beams, zero_magfield):
        with pytest.raises(ValueError):
            _LinearForceGovEq(simple_beams, zero_magfield, a=jnp.array([0.0, 0.0]))

    def test_default_accel_is_zero(self, linear_geq):
        assert jnp.allclose(linear_geq.constant_accel, jnp.zeros(3))

    def test_custom_accel_stored(self, simple_beams, zero_magfield):
        g = jnp.array([0.0, 0.0, -9.8])
        geq = _LinearForceGovEq(simple_beams, zero_magfield, a=g)
        assert jnp.allclose(geq.constant_accel, g)

    def test_initial_sol_is_none(self, linear_geq):
        assert linear_geq.sol is None

    def test_profile_is_empty_dict(self, linear_geq):
        assert linear_geq.profile == {}

    def test_r0_defaults_to_origin(self, linear_geq):
        assert jnp.allclose(linear_geq.r0, jnp.zeros(3))

    def test_v0_defaults_to_zero(self, linear_geq):
        assert jnp.allclose(linear_geq.v0, jnp.zeros(3))


# ---------------------------------------------------------------------------
# TestSetPositionVelocity
# ---------------------------------------------------------------------------


class TestSetPositionVelocity:
    def test_set_initial_position_updates_r0(self, linear_geq):
        r_new = jnp.array([1.0, 2.0, 3.0])
        linear_geq.set_initial_position(r_new)
        assert jnp.allclose(linear_geq.r0, r_new)

    def test_set_initial_velocity_updates_v0(self, linear_geq):
        v_new = jnp.array([0.1, 0.2, 0.3])
        linear_geq.set_initial_velocity(v_new)
        assert jnp.allclose(linear_geq.v0, v_new)

    def test_set_initial_position_resets_sol(self, linear_geq):
        linear_geq.sol = "some_solution"
        linear_geq.set_initial_position(jnp.zeros(3))
        assert linear_geq.sol is None

    def test_set_initial_velocity_resets_sol(self, linear_geq):
        linear_geq.sol = "some_solution"
        linear_geq.set_initial_velocity(jnp.zeros(3))
        assert linear_geq.sol is None

    def test_set_position_and_velocity_together(self, linear_geq):
        r = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 1.0, 0.0])
        linear_geq.set_initial_position_and_velocity(r, v)
        assert jnp.allclose(linear_geq.r0, r)
        assert jnp.allclose(linear_geq.v0, v)

    def test_r0_stored_as_jax_array(self, linear_geq):
        linear_geq.set_initial_position(np.array([1.0, 2.0, 3.0]))
        assert isinstance(linear_geq.r0, jnp.ndarray)

    def test_v0_stored_as_jax_array(self, linear_geq):
        linear_geq.set_initial_velocity(np.array([0.1, 0.2, 0.3]))
        assert isinstance(linear_geq.v0, jnp.ndarray)


# ---------------------------------------------------------------------------
# TestFindEquilibriumPosition
# ---------------------------------------------------------------------------


class TestFindEquilibriumPosition:
    """Find the position r⃗_eq where the net force vanishes: F⃗(r⃗_eq) = 0⃗.

    Uses root-finding (Brent's method for 1D, general solver for multi-axis).
    For the test's linear restoring force F = -kr, the equilibrium is at r = 0."""

    def test_single_axis_finds_zero(self, linear_geq):
        # F = -k*r, equilibrium at r=0 for any k>0
        r_eq = linear_geq.find_equilibrium_position([0], bracket=[-2.0, 2.0], method="brentq")
        assert float(r_eq[0]) == pytest.approx(0.0, abs=1e-6)

    def test_two_axes_find_zero(self, linear_geq):
        r_eq = linear_geq.find_equilibrium_position([0, 1], x0=[0.5, 0.5])
        assert float(r_eq[0]) == pytest.approx(0.0, abs=1e-4)
        assert float(r_eq[1]) == pytest.approx(0.0, abs=1e-4)

    def test_r_eq_stored_on_object(self, linear_geq):
        linear_geq.find_equilibrium_position([2], bracket=[-2.0, 2.0], method="brentq")
        assert linear_geq.r_eq is not None
        assert linear_geq.r_eq.shape == (3,)


# ---------------------------------------------------------------------------
# TestTrappingFrequencies
# ---------------------------------------------------------------------------


class TestTrappingFrequencies:
    """Trapping frequency ω from the force gradient: ω = √(−∂F/∂r / m).

    For a harmonic potential F = -kr this gives ω = √(k/m).  Verifies
    correct scaling (k=1,m=1 → ω=1; k=4,m=1 → ω=2) and that an
    isotropic force gives identical frequencies on all three axes."""

    def test_single_axis_returns_scalar(self, linear_geq):
        omega = linear_geq.trapping_frequencies([0], eps=0.001)
        assert isinstance(omega, float)

    def test_omega_positive_for_restoring_force(self, linear_geq):
        omega = linear_geq.trapping_frequencies([0, 1, 2], eps=0.001)
        assert jnp.all(omega > 0)

    def test_omega_correct_value(self, linear_geq):
        # With k=1, mass=1: omega = sqrt(k/mass) = 1.0
        omega = linear_geq.trapping_frequencies([0], eps=1e-4)
        assert omega == pytest.approx(1.0, rel=1e-3)

    def test_all_three_axes_same_for_isotropic(self, linear_geq):
        omega = linear_geq.trapping_frequencies([0, 1, 2], eps=1e-4)
        assert float(omega[0]) == pytest.approx(float(omega[1]), rel=1e-3)
        assert float(omega[0]) == pytest.approx(float(omega[2]), rel=1e-3)

    def test_omega_scales_with_k(self, simple_beams, zero_magfield):
        k = 4.0
        geq = _LinearForceGovEq(simple_beams, zero_magfield, k=k, mass=1.0)
        omega = geq.trapping_frequencies([0], eps=1e-4)
        # omega = sqrt(k/mass) = sqrt(4) = 2.0
        assert omega == pytest.approx(2.0, rel=1e-3)


# ---------------------------------------------------------------------------
# TestDampingCoeff
# ---------------------------------------------------------------------------


class TestDampingCoeff:
    """Velocity damping coefficient β = −∂F/∂v.

    For the damped model F = -kr - βv, this gives β = 2.0.  A purely
    position-dependent force (no velocity dependence) has β = 0.
    Positive β means the force opposes motion — essential for laser cooling."""

    def test_single_axis_returns_scalar(self, damped_geq):
        beta = damped_geq.damping_coeff([0], eps=0.001)
        assert isinstance(beta, float)

    def test_beta_correct_value(self, damped_geq):
        # With beta=2.0: beta_coeff = -dF/dv = 2.0
        beta = damped_geq.damping_coeff([0], eps=1e-4)
        assert beta == pytest.approx(2.0, rel=1e-3)

    def test_no_velocity_damping_gives_zero(self, linear_geq):
        # _LinearForceGovEq has no velocity dependence → beta = 0
        beta = linear_geq.damping_coeff([0], eps=1e-4)
        assert beta == pytest.approx(0.0, abs=1e-6)

    def test_all_three_axes(self, damped_geq):
        beta = damped_geq.damping_coeff([0, 1, 2], eps=1e-4)
        assert beta.shape == (3,)
        for i in range(3):
            assert float(beta[i]) == pytest.approx(2.0, rel=1e-3)
