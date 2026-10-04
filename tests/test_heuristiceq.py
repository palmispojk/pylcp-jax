"""
Tests for pylcp_jax/heuristiceq.py
"""

import jax.numpy as jnp
import numpy as np
import pytest

from pylcp_jax.fields import constantMagneticField, laserBeams
from pylcp_jax.heuristiceq import heuristiceq

# ---------------------------------------------------------------------------
# Local fixtures (shared ones live in conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def nonzero_B():
    return constantMagneticField(jnp.array([0.0, 0.0, 1.0]))


@pytest.fixture
def heq(single_beam, zero_B):
    return heuristiceq(single_beam, zero_B, mass=1.0, gamma=1.0, k=1.0)


@pytest.fixture
def heq_sym(symmetric_beams, zero_B):
    return heuristiceq(symmetric_beams, zero_B, mass=1.0, gamma=1.0, k=1.0)


# ---------------------------------------------------------------------------
# TestHeuristiceqInit
# ---------------------------------------------------------------------------


class TestHeuristiceqInit:
    def test_mass_stored(self, heq):
        assert heq.mass == pytest.approx(1.0)

    def test_gamma_stored(self, heq):
        assert heq.gamma == pytest.approx(1.0)

    def test_k_stored(self, heq):
        assert heq.k == pytest.approx(1.0)

    def test_profile_starts_empty(self, heq):
        assert heq.profile == {}

    def test_sol_starts_none(self, heq):
        assert heq.sol is None

    def test_r0_defaults_to_origin(self, heq):
        assert jnp.allclose(heq.r0, jnp.zeros(3))

    def test_v0_defaults_to_zero(self, heq):
        assert jnp.allclose(heq.v0, jnp.zeros(3))

    def test_invalid_key_raises(self, zero_B):
        bad_beams = {
            "a->b": laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.1, "delta": 0.0}])
        }
        with pytest.raises(KeyError):
            heuristiceq(bad_beams, zero_B, mass=1.0)

    def test_list_input_promoted(self, zero_B):
        beams_list = [{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.1, "delta": 0.0}]
        h = heuristiceq(beams_list, zero_B, mass=1.0)
        assert "g->e" in h.laserBeams

    def test_nonunit_mass(self, zero_B, single_beam):
        h = heuristiceq(single_beam, zero_B, mass=87.0)
        assert h.mass == pytest.approx(87.0)


# ---------------------------------------------------------------------------
# TestScatteringRate
# ---------------------------------------------------------------------------


class TestScatteringRate:
    """Heuristic two-level scattering rate: R = (Γ/2)·s/(1 + s + (2δ_eff/Γ)²).

    Here s is the saturation parameter and δ_eff = δ − k⃗·v⃗ is the
    effective detuning including the Doppler shift.  Rate must be
    non-negative, increase with intensity s, and decrease when the
    Doppler shift pushes δ_eff away from resonance."""

    def test_returns_correct_shape_single_beam(self, heq):
        R = jnp.zeros(3)
        V = jnp.zeros(3)
        rates = heq.scattering_rate(R, V, t=0.0)
        assert rates.shape == (1,)

    def test_returns_correct_shape_two_beams(self, heq_sym):
        R = jnp.zeros(3)
        V = jnp.zeros(3)
        rates = heq_sym.scattering_rate(R, V, t=0.0)
        assert rates.shape == (2,)

    def test_rates_non_negative(self, heq):
        R = jnp.zeros(3)
        V = jnp.zeros(3)
        rates = heq.scattering_rate(R, V, t=0.0)
        assert jnp.all(rates >= 0.0)

    def test_return_kvecs_gives_tuple(self, heq):
        result = heq.scattering_rate(jnp.zeros(3), jnp.zeros(3), t=0.0, return_kvecs=True)
        assert len(result) == 2
        rates, kvecs = result
        assert kvecs.shape == (1, 3)

    def test_resonant_rate_positive(self, heq):
        """At resonance and zero velocity the beam should produce a positive rate."""
        rates = heq.scattering_rate(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert float(rates[0]) > 0.0

    def test_rate_increases_with_intensity(self, zero_B):
        beams_lo = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.01, "delta": 0.0}])
        beams_hi = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": 0.0}])
        h_lo = heuristiceq(beams_lo, zero_B, mass=1.0, gamma=1.0, k=1.0)
        h_hi = heuristiceq(beams_hi, zero_B, mass=1.0, gamma=1.0, k=1.0)
        R_lo = h_lo.scattering_rate(jnp.zeros(3), jnp.zeros(3), t=0.0)
        R_hi = h_hi.scattering_rate(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert float(R_hi[0]) > float(R_lo[0])

    def test_doppler_shift_reduces_rate(self, zero_B):
        """Atom moving in +z with -z detuning → beam is blue-shifted → farther from resonance."""
        beams = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.5, "delta": -0.5}])
        h = heuristiceq(beams, zero_B, mass=1.0, gamma=1.0, k=1.0)
        v_zero = jnp.zeros(3)
        # v_pos: atom moves in +z, Doppler shifts toward resonance
        v_pos = jnp.array([0.0, 0.0, 1.0])
        R_zero = h.scattering_rate(jnp.zeros(3), v_zero, t=0.0)
        R_pos = h.scattering_rate(jnp.zeros(3), v_pos, t=0.0)
        # +z beam with delta=-0.5, v_z>0 shifts to -1.5
        assert float(R_pos[0]) < float(R_zero[0])

    def test_nonzero_B_does_not_crash(self, nonzero_B, single_beam):
        h = heuristiceq(single_beam, nonzero_B, mass=1.0, gamma=1.0, k=1.0)
        rates = h.scattering_rate(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert not jnp.any(jnp.isnan(rates))


# ---------------------------------------------------------------------------
# TestForce
# ---------------------------------------------------------------------------


class TestForce:
    """Radiation pressure force F⃗ = ℏk⃗ · R per beam.

    A single +z beam gives force only in +z.  Two symmetric
    counter-propagating beams give zero net force at v⃗ = 0⃗.
    Force scales with intensity (higher s → larger |F⃗|)."""

    def test_F_shape(self, heq):
        F, _ = heq.force(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert F.shape == (3,)

    def test_F_laser_key_present(self, heq):
        _, F_laser = heq.force(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert "g->e" in F_laser

    def test_F_laser_shape(self, heq):
        _, F_laser = heq.force(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert F_laser["g->e"].shape == (3, 1)  # 1 beam

    def test_single_beam_positive_z_force(self, heq):
        """+z beam → net force in +z direction."""
        F, _ = heq.force(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert float(F[2]) > 0.0

    def test_single_beam_no_transverse_force(self, heq):
        """A pure +z beam should produce no x or y force."""
        F, _ = heq.force(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert float(F[0]) == pytest.approx(0.0, abs=1e-12)
        assert float(F[1]) == pytest.approx(0.0, abs=1e-12)

    def test_symmetric_beams_zero_z_force(self, heq_sym):
        """Two equal counter-propagating beams → net z-force = 0 at v=0."""
        F, _ = heq_sym.force(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert float(F[2]) == pytest.approx(0.0, abs=1e-12)

    def test_force_scales_with_intensity(self, zero_B):
        beams_lo = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.1, "delta": 0.0}])
        beams_hi = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 2.0, "delta": 0.0}])
        h_lo = heuristiceq(beams_lo, zero_B, mass=1.0, gamma=1.0, k=1.0)
        h_hi = heuristiceq(beams_hi, zero_B, mass=1.0, gamma=1.0, k=1.0)
        F_lo, _ = h_lo.force(jnp.zeros(3), jnp.zeros(3), t=0.0)
        F_hi, _ = h_hi.force(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert float(F_hi[2]) > float(F_lo[2])

    def test_no_nan_in_force(self, heq):
        F, _ = heq.force(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert not jnp.any(jnp.isnan(F))


# ---------------------------------------------------------------------------
# TestFindEquilibriumForce
# ---------------------------------------------------------------------------


class TestFindEquilibriumForce:
    """Evaluate force at the current position r⃗₀ and velocity v⃗₀.

    Symmetric beams give zero force at the origin.  A Doppler-shifted
    atom (v⃗₀ ≠ 0⃗) scatters less from a co-propagating beam, reducing
    the force compared to v⃗₀ = 0⃗."""

    def test_returns_shape_3(self, heq):
        F = heq.find_equilibrium_force()
        assert F.shape == (3,)

    def test_single_beam_positive_z(self, heq):
        F = heq.find_equilibrium_force()
        assert float(F[2]) > 0.0

    def test_return_details_gives_tuple(self, heq):
        result = heq.find_equilibrium_force(return_details=True)
        assert len(result) == 3  # (F, F_laser, R_rates)

    def test_return_details_R_shape(self, heq):
        _, _, R_rates = heq.find_equilibrium_force(return_details=True)
        assert R_rates.shape == (1,)  # 1 beam

    def test_symmetric_beams_zero_z_force(self, heq_sym):
        F = heq_sym.find_equilibrium_force()
        assert float(F[2]) == pytest.approx(0.0, abs=1e-12)

    def test_no_nan(self, heq):
        F = heq.find_equilibrium_force()
        assert not jnp.any(jnp.isnan(F))

    def test_respects_r0_v0(self, zero_B):
        """Force at non-zero v0 should differ from v0=0 (Doppler shift)."""
        beams = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": 0.0}])
        h_rest = heuristiceq(beams, zero_B, mass=1.0, gamma=1.0, k=1.0)
        h_moving = heuristiceq(
            beams, zero_B, mass=1.0, gamma=1.0, k=1.0, v0=jnp.array([0.0, 0.0, 2.0])
        )
        F_rest = h_rest.find_equilibrium_force()
        F_moving = h_moving.find_equilibrium_force()
        # Moving in +z shifts the +z beam off resonance → less force
        assert float(F_moving[2]) < float(F_rest[2])


# ---------------------------------------------------------------------------
# TestGenerateForceProfile
# ---------------------------------------------------------------------------


class TestGenerateForceProfile:
    def test_1d_F_shape(self, heq):
        R = np.zeros((3, 10))
        V = np.zeros((3, 10))
        fp = heq.generate_force_profile(R, V)
        assert fp.F.shape == (3, 10)

    def test_2d_F_shape(self, heq):
        R = np.zeros((3, 5, 4))
        V = np.zeros((3, 5, 4))
        fp = heq.generate_force_profile(R, V)
        assert fp.F.shape == (3, 5, 4)

    def test_f_key_present(self, heq):
        R = np.zeros((3, 3))
        V = np.zeros((3, 3))
        fp = heq.generate_force_profile(R, V)
        assert "g->e" in fp.f

    def test_profile_stored_by_name(self, heq):
        R = np.zeros((3, 3))
        V = np.zeros((3, 3))
        heq.generate_force_profile(R, V, name="test")
        assert "test" in heq.profile

    def test_profile_name_auto_increments(self, heq):
        R = np.zeros((3, 3))
        V = np.zeros((3, 3))
        heq.generate_force_profile(R, V)
        heq.generate_force_profile(R, V)
        assert "0" in heq.profile
        assert "1" in heq.profile

    def test_neq_is_none(self, heq):
        """heuristiceq has no internal state, so Neq should be None."""
        R = np.zeros((3, 3))
        V = np.zeros((3, 3))
        fp = heq.generate_force_profile(R, V)
        assert fp.Neq is None

    def test_uniform_grid_positive_z_force(self, heq):
        """Every grid point should have positive z-force for a +z beam."""
        R = np.zeros((3, 5))
        V = np.zeros((3, 5))
        fp = heq.generate_force_profile(R, V)
        assert jnp.all(fp.F[2] > 0.0)


# ---------------------------------------------------------------------------
# Test1DMOTForceProfile – regression tests for magnetic field gradient
# ---------------------------------------------------------------------------


class Test1DMOTForceProfile:
    """1D magneto-optical trap with linear B-field gradient B⃗ = −αr⃗.

    The Zeeman shift makes σ⁺/σ⁻ beams address different m_F transitions
    depending on position, producing a restoring force: F(x>0) < 0 and
    F(x<0) > 0.  By symmetry F(0) = 0 and F(x) = −F(−x) (antisymmetric).
    Force must be non-zero at the resonance position x_res ≈ δ/α."""

    @pytest.fixture
    def mot_heq(self):
        from pylcp_jax.fields import magField

        alpha = 1.0
        beams = laserBeams(
            [
                {"kvec": [1.0, 0.0, 0.0], "pol": -1, "s": 1.0, "delta": -4.0},
                {"kvec": [-1.0, 0.0, 0.0], "pol": -1, "s": 1.0, "delta": -4.0},
            ]
        )
        B = magField(lambda R: -alpha * R)
        return heuristiceq(beams, B, mass=1.0, gamma=1.0, k=1.0)

    def test_force_profile_no_nan(self, mot_heq):
        """Force profile with a B-field gradient must not contain NaN."""
        x = np.linspace(-10, 10, 21)
        R = np.array([x, np.zeros_like(x), np.zeros_like(x)])
        V = np.zeros_like(R)
        fp = mot_heq.generate_force_profile(R, V)
        assert not np.any(np.isnan(fp.F))

    def test_force_at_origin_is_zero(self, mot_heq):
        """By symmetry the force at x=0 (where B=0) must vanish."""
        mot_heq.set_initial_position_and_velocity(jnp.zeros(3), jnp.zeros(3))
        F = mot_heq.find_equilibrium_force()
        assert float(F[0]) == pytest.approx(0.0, abs=1e-10)

    def test_force_is_restoring(self, mot_heq):
        """For x>0 the force must point in -x (restoring), and vice versa."""
        x = np.linspace(-10, 10, 21)
        R = np.array([x, np.zeros_like(x), np.zeros_like(x)])
        V = np.zeros_like(R)
        fp = mot_heq.generate_force_profile(R, V)
        # Exclude the origin
        pos_mask = x > 1.0
        neg_mask = x < -1.0
        assert np.all(fp.F[0, pos_mask] < 0.0), "Force should be negative for x>0"
        assert np.all(fp.F[0, neg_mask] > 0.0), "Force should be positive for x<0"

    def test_force_is_antisymmetric(self, mot_heq):
        """F(x) ≈ -F(-x) for the symmetric 1D MOT."""
        x = np.linspace(-10, 10, 21)
        R = np.array([x, np.zeros_like(x), np.zeros_like(x)])
        V = np.zeros_like(R)
        fp = mot_heq.generate_force_profile(R, V)
        F_x = np.array(fp.F[0])
        assert np.allclose(F_x, -F_x[::-1], atol=1e-10)

    def test_force_nonzero_away_from_origin(self, mot_heq):
        """Force must be non-zero at the resonance position."""
        mot_heq.set_initial_position_and_velocity(jnp.array([4.0, 0.0, 0.0]), jnp.zeros(3))
        F = mot_heq.find_equilibrium_force()
        assert abs(float(F[0])) > 1e-4


# ---------------------------------------------------------------------------
# TestRandomRecoilKickDistribution
# ---------------------------------------------------------------------------


class TestRandomRecoilKickDistribution:
    """Verify random_recoil kicks use two independent random unit vectors.

    The sum of two independent random unit vectors has a magnitude that
    varies between 0 and 2.  A single vector scaled by 2 would always
    have magnitude exactly 2.
    """

    def test_kick_magnitude_varies(self, zero_B, single_beam):
        """Kick magnitudes must not all be identical (rules out fixed * 2)."""
        import jax

        heq = heuristiceq(single_beam, zero_B, mass=1.0, gamma=1.0, k=1.0)

        free_axes = jnp.array([1.0, 1.0, 1.0])
        mass = heq.mass

        def _random_unit_vector(key):
            key_phi, key_z = jax.random.split(key)
            phi = 2.0 * jnp.pi * jax.random.uniform(key_phi)
            z = 2.0 * jax.random.uniform(key_z) - 1.0
            r_xy = jnp.sqrt(1.0 - z**2)
            return jnp.array([r_xy * jnp.cos(phi), r_xy * jnp.sin(phi), z]) * free_axes

        # Directly test the random_recoil_fn inside evolve_motion_batch
        # by running many short evolve_motion calls and collecting kicks
        magnitudes = []
        for i in range(200):
            key = jax.random.PRNGKey(i)
            key, key_dice, key_v1, key_v2 = jax.random.split(key, 4)
            vec1 = _random_unit_vector(key_v1)
            vec2 = _random_unit_vector(key_v2)
            kick = heq.k / mass * (vec1 + vec2)
            magnitudes.append(float(jnp.linalg.norm(kick)))

        # If kicks are vec1+vec2, magnitudes vary; if *2, they're all identical
        assert np.std(magnitudes) > 1e-6, (
            "All kick magnitudes are identical — likely using a single "
            "random vector * 2 instead of two independent random vectors"
        )


# ---------------------------------------------------------------------------
# TestEvolveMotion
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEvolveMotion:
    """Single-atom trajectory integration via scipy."""

    @pytest.fixture
    def heq_mot(self, zero_B):
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": 0.0},
            ]
        )
        h = heuristiceq(beams, zero_B, mass=1.0, gamma=1.0, k=1.0)
        h.set_initial_position_and_velocity(jnp.zeros(3), jnp.zeros(3))
        return h

    def test_returns_solution(self, heq_mot):
        sol = heq_mot.evolve_motion([0, 1.0])
        assert hasattr(sol, "t")
        assert hasattr(sol, "r")
        assert hasattr(sol, "v")

    def test_sol_stored_on_instance(self, heq_mot):
        heq_mot.evolve_motion([0, 1.0])
        assert heq_mot.sol is not None

    def test_position_changes_with_force(self, heq_mot):
        """A +z beam should accelerate the atom in +z."""
        sol = heq_mot.evolve_motion([0, 5.0], max_step=0.5)
        z_final = float(sol.r[2, -1])
        assert z_final > 0.0, "Atom should have moved in +z"

    def test_velocity_increases_along_beam(self, heq_mot):
        sol = heq_mot.evolve_motion([0, 5.0], max_step=0.5)
        vz_final = float(sol.v[2, -1])
        assert vz_final > 0.0, "Atom should have positive z-velocity"

    def test_transverse_stays_zero(self, heq_mot):
        """No transverse force → x,y should stay at zero."""
        sol = heq_mot.evolve_motion([0, 5.0], max_step=0.5)
        assert np.allclose(sol.r[0], 0.0, atol=1e-10)
        assert np.allclose(sol.r[1], 0.0, atol=1e-10)

    def test_freeze_axis(self, heq_mot):
        """Freezing z should prevent motion even with a +z beam."""
        sol = heq_mot.evolve_motion([0, 5.0], freeze_axis=[False, False, True], max_step=0.5)
        assert np.allclose(sol.r[2], 0.0, atol=1e-10)
        assert np.allclose(sol.v[2], 0.0, atol=1e-10)

    def test_no_nan(self, heq_mot):
        sol = heq_mot.evolve_motion([0, 2.0])
        assert not np.any(np.isnan(sol.r))
        assert not np.any(np.isnan(sol.v))


# ---------------------------------------------------------------------------
# TestEvolveMotionBatch
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEvolveMotionBatch:
    """Batched atom trajectory integration via JAX/diffrax."""

    @pytest.fixture
    def heq_batch(self, zero_B):
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": 0.0},
            ]
        )
        h = heuristiceq(beams, zero_B, mass=1.0, gamma=1.0, k=1.0)
        h.set_initial_position_and_velocity(jnp.zeros(3), jnp.zeros(3))
        return h

    def test_returns_list_of_sols(self, heq_batch):
        sols = heq_batch.evolve_motion_batch([0, 1.0], n_points=11)
        assert isinstance(sols, list)
        assert len(sols) == 1  # default: single atom from r0/v0

    def test_sol_has_r_and_v(self, heq_batch):
        sols = heq_batch.evolve_motion_batch([0, 1.0], n_points=11)
        assert hasattr(sols[0], "r")
        assert hasattr(sols[0], "v")

    def test_multiple_atoms(self, heq_batch):
        y0 = jnp.zeros((3, 6))  # 3 atoms, state = [v(3), r(3)]
        sols = heq_batch.evolve_motion_batch([0, 1.0], y0_batch=y0, n_points=11)
        assert len(sols) == 3

    def test_atom_accelerates_in_z(self, heq_batch):
        sols = heq_batch.evolve_motion_batch([0, 5.0], n_points=51)
        z_final = float(sols[0].r[2, -1])
        assert z_final > 0.0

    def test_freeze_axis_batch(self, heq_batch):
        sols = heq_batch.evolve_motion_batch(
            [0, 5.0], freeze_axis=[False, False, True], n_points=51
        )
        assert np.allclose(sols[0].r[2], 0.0, atol=1e-10)

    def test_random_recoil_runs(self, heq_batch):
        """random_recoil=True should run without error."""
        sols = heq_batch.evolve_motion_batch([0, 2.0], random_recoil=True, n_points=21)
        assert len(sols) == 1
        assert not np.any(np.isnan(sols[0].r))

    def test_sols_stored_on_instance(self, heq_batch):
        heq_batch.evolve_motion_batch([0, 1.0], n_points=11)
        assert heq_batch.sols is not None
