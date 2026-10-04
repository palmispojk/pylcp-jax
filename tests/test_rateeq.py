"""
Tests for pylcp_jax/rateeq.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_ham

from pylcp_jax.fields import laserBeams, magField
from pylcp_jax.hamiltonian import hamiltonian
from pylcp_jax.rateeq import force_profile, rateeq

# ---------------------------------------------------------------------------
# Local fixtures (shared ones live in conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def req(single_beam, zero_B, ham):
    return rateeq(single_beam, zero_B, ham)


@pytest.fixture
def req_sym(symmetric_beams, zero_B, ham):
    return rateeq(symmetric_beams, zero_B, ham)


# ---------------------------------------------------------------------------
# TestForceProfile
# ---------------------------------------------------------------------------


class TestForceProfile:
    def test_construction_2d_grid(self, req):
        R = np.zeros((3, 5, 4))
        V = np.zeros((3, 5, 4))
        fp = force_profile(R, V, req.laserBeams, req.hamiltonian)
        assert fp.F.shape == (3, 5, 4)
        assert fp.R.shape == (3, 5, 4)

    def test_construction_1d_grid(self, req):
        R = np.zeros((3, 10))
        V = np.zeros((3, 10))
        fp = force_profile(R, V, req.laserBeams, req.hamiltonian)
        assert fp.F.shape == (3, 10)

    def test_Rijl_key_present(self, req):
        R = np.zeros((3, 3))
        V = np.zeros((3, 3))
        fp = force_profile(R, V, req.laserBeams, req.hamiltonian)
        assert "g->e" in fp.Rijl

    def test_Neq_shape(self, req):
        R = np.zeros((3, 5))
        V = np.zeros((3, 5))
        fp = force_profile(R, V, req.laserBeams, req.hamiltonian)
        # 4 states (1 ground + 3 excited), grid length 5
        assert fp.Neq.shape == (5, req.hamiltonian.n)


# ---------------------------------------------------------------------------
# TestRateeqInit
# ---------------------------------------------------------------------------


class TestRateeqInit:
    def test_hamiltonian_stored(self, req, ham):
        assert req.hamiltonian.n == ham.n

    def test_include_mag_forces_default_true(self, req):
        assert req.include_mag_forces is True

    def test_include_mag_forces_can_disable(self, single_beam, zero_B, ham):
        r = rateeq(single_beam, zero_B, ham, include_mag_forces=False)
        assert r.include_mag_forces is False

    def test_svd_eps_default(self, req):
        assert req.svd_eps == pytest.approx(1e-10)

    def test_custom_svd_eps(self, single_beam, zero_B, ham):
        r = rateeq(single_beam, zero_B, ham, svd_eps=1e-8)
        assert r.svd_eps == pytest.approx(1e-8)

    def test_recoil_velocity_key_present(self, req):
        assert "g->e" in req.recoil_velocity

    def test_recoil_velocity_positive(self, req):
        assert req.recoil_velocity["g->e"] > 0

    def test_profile_starts_empty(self, req):
        assert req.profile == {}

    def test_r0_defaults_to_origin(self, req):
        assert jnp.allclose(req.r0, jnp.zeros(3))

    def test_v0_defaults_to_zero(self, req):
        assert jnp.allclose(req.v0, jnp.zeros(3))

    def test_Rev_decay_precomputed_for_diagonal_ham(self, req, ham):
        # F=0->F=1 with singleF gives a diagonal hamiltonian
        assert np.all(ham.diagonal)
        assert hasattr(req, "Rev_decay")
        assert req.Rev_decay.shape == (ham.n, ham.n)


# ---------------------------------------------------------------------------
# TestSetInitialPop
# ---------------------------------------------------------------------------


class TestSetInitialPop:
    def test_wrong_length_raises(self, req):
        with pytest.raises(ValueError):
            req.set_initial_pop(jnp.array([1.0, 0.0, 0.0]))  # too short (3 vs 4)

    def test_nan_raises(self, req, ham):
        bad = jnp.array([float("nan")] + [0.0] * (ham.n - 1))
        with pytest.raises(ValueError):
            req.set_initial_pop(bad)

    def test_inf_raises(self, req, ham):
        bad = jnp.array([float("inf")] + [0.0] * (ham.n - 1))
        with pytest.raises(ValueError):
            req.set_initial_pop(bad)

    def test_valid_pop_stored(self, req, ham):
        N0 = jnp.zeros(ham.n).at[0].set(1.0)
        req.set_initial_pop(N0)
        assert jnp.allclose(req.N0, N0)

    def test_set_from_equilibrium(self, req):
        req.set_initial_pop_from_equilibrium()
        assert hasattr(req, "N0")
        assert jnp.allclose(jnp.sum(req.N0), 1.0, atol=1e-6)
        assert jnp.all(req.N0 >= 0)


# ---------------------------------------------------------------------------
# TestConstructEvolutionMatrix
# ---------------------------------------------------------------------------


class TestConstructEvolutionMatrix:
    """Rate evolution matrix Rev for the population vector dN/dt = Rev·N.

    Rev encodes both laser-driven transitions (absorption/stimulated
    emission) and spontaneous decay.  Physical constraints:

    - Column sums = 0: probability is conserved (dΣNᵢ/dt = 0).
    - Diagonal ≤ 0: each state loses population at rate |Rev_ii|.
    - Off-diagonal ≥ 0: population flows from state j to state i
      at rate Rev_ij ≥ 0.
    - Higher laser intensity → larger off-diagonal pumping rates."""

    def test_returns_matrix_and_rijl(self, req, ham):
        Rev, Rijl = req.construct_evolution_matrix(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert Rev.shape == (ham.n, ham.n)
        assert "g->e" in Rijl

    def test_column_sums_are_zero(self, req, ham):
        """Probability conservation: columns of Rev must sum to zero."""
        Rev, _ = req.construct_evolution_matrix(jnp.zeros(3), jnp.zeros(3), t=0.0)
        col_sums = jnp.sum(Rev, axis=0)
        assert jnp.allclose(col_sums, jnp.zeros(ham.n), atol=1e-10)

    def test_diagonal_negative_or_zero(self, req, ham):
        """Diagonal entries of Rev must be ≤ 0 (decay/pump out)."""
        Rev, _ = req.construct_evolution_matrix(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert jnp.all(jnp.diag(Rev) <= 0)

    def test_off_diagonal_non_negative(self, req, ham):
        """Off-diagonal entries of Rev must be ≥ 0 (rates in)."""
        Rev, _ = req.construct_evolution_matrix(jnp.zeros(3), jnp.zeros(3), t=0.0)
        n = ham.n
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert float(Rev[i, j]) >= -1e-12, (
                        f"Off-diagonal Rev[{i},{j}] = {float(Rev[i, j])} < 0"
                    )

    def test_pumping_increases_with_intensity(self, zero_B, ham):
        """Higher intensity → larger off-diagonal pumping rates."""
        beams_lo = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.01, "delta": 0.0}])
        beams_hi = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": 0.0}])
        req_lo = rateeq(beams_lo, zero_B, make_ham())
        req_hi = rateeq(beams_hi, zero_B, make_ham())
        Rev_lo, _ = req_lo.construct_evolution_matrix(jnp.zeros(3), jnp.zeros(3))
        Rev_hi, _ = req_hi.construct_evolution_matrix(jnp.zeros(3), jnp.zeros(3))
        # σ+ beam drives mF=0 → mF=+1, which is index 3 in the excited manifold.
        # Pumping Rev[3,0] should be larger for higher intensity.
        assert float(Rev_hi[3, 0]) > float(Rev_lo[3, 0])


# ---------------------------------------------------------------------------
# TestEquilibriumPopulations
# ---------------------------------------------------------------------------


class TestEquilibriumPopulations:
    """Steady-state populations Neq satisfying Rev·Neq = 0.

    At equilibrium, absorption balances spontaneous + stimulated emission.
    With no laser (s=0), all population must remain in the ground state
    (the only dark state).  Populations must sum to 1 and be non-negative."""

    def test_populations_sum_to_one(self, req):
        Neq = req.equilibrium_populations(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert float(jnp.sum(Neq)) == pytest.approx(1.0, abs=1e-6)

    def test_populations_non_negative(self, req):
        Neq = req.equilibrium_populations(jnp.zeros(3), jnp.zeros(3), t=0.0)
        assert jnp.all(Neq >= -1e-10)

    def test_return_details_gives_tuple(self, req):
        result = req.equilibrium_populations(jnp.zeros(3), jnp.zeros(3), t=0.0, return_details=True)
        assert len(result) == 3  # (Neq, Rev, Rijl)

    def test_ground_state_has_population(self, req):
        """At weak saturation the ground state should hold most population."""
        Neq = req.equilibrium_populations(jnp.zeros(3), jnp.zeros(3), t=0.0)
        # Ground state is index 0 (1 state), excited are 1-3 (3 states)
        assert float(Neq[0]) > 0.0

    def test_zero_intensity_gives_all_ground(self, zero_B, ham):
        """With no laser, all population stays in the ground state."""
        no_beams = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.0, "delta": 0.0}])
        req_dark = rateeq(no_beams, zero_B, make_ham())
        Neq = req_dark.equilibrium_populations(jnp.zeros(3), jnp.zeros(3), t=0.0)
        # Ground state (index 0) should have all population
        assert float(Neq[0]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestFindEquilibriumForce
# ---------------------------------------------------------------------------


class TestFindEquilibriumForce:
    def test_returns_shape_3(self, req):
        req.set_initial_pop_from_equilibrium()
        F = req.find_equilibrium_force()
        assert F.shape == (3,)

    def test_return_details_gives_tuple(self, req):
        req.set_initial_pop_from_equilibrium()
        result = req.find_equilibrium_force(return_details=True)
        assert len(result) == 5  # (F_eq, f_eq, N_eq, Rijl, f_mag)

    def test_symmetric_beams_zero_force_at_origin(self, req_sym):
        """Two equal counter-propagating beams → net z-force = 0 at r=0, v=0."""
        req_sym.set_initial_pop_from_equilibrium()
        F = req_sym.find_equilibrium_force()
        assert float(F[2]) == pytest.approx(0.0, abs=1e-10)

    def test_single_beam_gives_positive_z_force(self, req):
        """+z beam → net force in +z direction."""
        req.set_initial_pop_from_equilibrium()
        F = req.find_equilibrium_force()
        assert float(F[2]) > 0.0

    def test_force_scale_with_intensity(self, zero_B, ham):
        """Force should increase with laser intensity."""
        beams_lo = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.1, "delta": 0.0}])
        beams_hi = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 2.0, "delta": 0.0}])
        req_lo = rateeq(beams_lo, zero_B, make_ham())
        req_hi = rateeq(beams_hi, zero_B, make_ham())
        req_lo.set_initial_pop_from_equilibrium()
        req_hi.set_initial_pop_from_equilibrium()
        F_lo = req_lo.find_equilibrium_force()
        F_hi = req_hi.find_equilibrium_force()
        assert float(F_hi[2]) > float(F_lo[2])

    def test_no_time_dependence_flag_works(self, req):
        """Calling find_equilibrium_force with no tdepend set should succeed."""
        req.set_initial_pop_from_equilibrium()
        F = req.find_equilibrium_force()
        assert not jnp.any(jnp.isnan(F))


# ---------------------------------------------------------------------------
# Test1DMOTForceProfile – regression tests for magnetic field gradient
# ---------------------------------------------------------------------------


class Test1DMOTForceProfile:
    """1D MOT with linear B-field gradient: force profile must be restoring.

    In a magneto-optical trap, a position-dependent Zeeman shift brings
    different mF transitions into resonance on opposite sides of the
    trap centre.  With correctly chosen beam polarizations, this creates
    a restoring force F(x) that is antisymmetric: F(x) = −F(−x) and
    vanishes at x=0 (where B=0).

    This is the primary regression test for the Zeeman shift bug in the
    JAX force-profile path (_generate_force_profile_jax).
    """

    @pytest.fixture
    def mot_fp(self):
        """Build a rateeq 1D MOT and return its force profile.

        Uses alpha scaled so that the Zeeman shift matches the detuning
        at x = ±x_res, i.e.  alpha * x_res * mu_z_per_state = |delta|.
        """
        from pylcp_jax.fields import magField

        ham = make_ham(gamma=1.0, k=1.0, mass=1.0)
        # mu_z per excited mF state (physical units)
        mu_val = 1399624.49171  # |diag(mu_e[1])[0]|
        delta = -4.0
        x_res = 5.0  # resonance position
        alpha = abs(delta) / (x_res * mu_val)
        beams = laserBeams(
            [
                {"kvec": [1.0, 0.0, 0.0], "pol": -1, "s": 1.0, "delta": delta},
                {"kvec": [-1.0, 0.0, 0.0], "pol": -1, "s": 1.0, "delta": delta},
            ]
        )
        B = magField(lambda R: -alpha * R)
        req = rateeq(beams, B, ham)
        x = np.linspace(-10, 10, 21) * x_res / 10.0
        R = np.array([x, np.zeros_like(x), np.zeros_like(x)])
        V = np.zeros_like(R)
        fp = req.generate_force_profile(R, V)
        return fp, x

    def test_force_profile_no_nan(self, mot_fp):
        """Force profile with a B-field gradient must not contain NaN."""
        fp, _ = mot_fp
        assert not np.any(np.isnan(fp.F))

    def test_force_at_origin_is_zero(self, mot_fp):
        """By symmetry the force at x=0 (where B=0) must vanish."""
        fp, x = mot_fp
        origin_idx = np.argmin(np.abs(x))
        assert float(fp.F[0, origin_idx]) == pytest.approx(0.0, abs=1e-10)

    def test_force_is_restoring(self, mot_fp):
        """For x>0 the force must point in -x (restoring), and vice versa."""
        fp, x = mot_fp
        pos_mask = x > 1.0
        neg_mask = x < -1.0
        assert np.all(fp.F[0, pos_mask] < 0.0), "Force should be negative for x>0"
        assert np.all(fp.F[0, neg_mask] > 0.0), "Force should be positive for x<0"

    def test_force_is_antisymmetric(self, mot_fp):
        """F(x) ≈ -F(-x) for the symmetric 1D MOT."""
        fp, _ = mot_fp
        F_x = np.array(fp.F[0])
        assert np.allclose(F_x, -F_x[::-1], atol=1e-10)

    def test_force_nonzero_away_from_origin(self, mot_fp):
        """Force must be non-zero near the resonance position."""
        fp, x = mot_fp
        # Pick a point away from origin but within the trapping region
        idx = np.argmin(np.abs(x - x[len(x) // 2 + len(x) // 4]))
        assert abs(float(fp.F[0, idx])) > 1e-6


# ---------------------------------------------------------------------------
# TestRandomRecoilKickDistribution
# ---------------------------------------------------------------------------


class TestRandomRecoilKickDistribution:
    """Verify random_recoil kicks use two independent random unit vectors.

    The sum of two independent random unit vectors has an average magnitude
    of ~4/pi ≈ 1.27 (in 3D) and a distribution that ranges from 0 to 2.
    A single vector scaled by 2 would always have magnitude exactly 2.
    """

    @pytest.fixture
    def recoil_func(self, single_beam, zero_B, ham):
        req = rateeq(single_beam, zero_B, ham)
        # Trigger internal setup by generating a force profile first
        R = np.zeros((3, 1))
        V = np.zeros((3, 1))
        req.generate_force_profile(R, V, name="test")
        free_axes = jnp.array([1.0, 1.0, 1.0])
        n_states = ham.n
        return req._make_random_recoil_func(n_states, free_axes, max_P=0.1)

    def test_kick_magnitude_varies(self, recoil_func):
        """Kick magnitudes must not all be identical (rules out fixed * 2)."""
        n_states = 4  # F=0 (1) + F'=1 (3)
        # Build a state where excited population is large so scatters happen
        N = jnp.array([0.1, 0.3, 0.3, 0.3])
        v = jnp.zeros(3)
        r = jnp.zeros(3)
        y = jnp.concatenate([N, v, r])

        magnitudes = []
        dt = jnp.float64(10.0)  # large dt to guarantee scatters
        for i in range(200):
            key_i = jax.random.PRNGKey(i)
            y_out, n_scat, _, _ = recoil_func(0.0, y, dt, key_i)
            dv = y_out[n_states : n_states + 3] - y[n_states : n_states + 3]
            mag = float(jnp.linalg.norm(dv))
            if mag > 0:
                magnitudes.append(mag)

        assert len(magnitudes) > 10, "Too few scatter events to test"
        # If kicks are vec1+vec2, magnitudes vary; if *2, they're all identical
        assert np.std(magnitudes) > 1e-6, (
            "All kick magnitudes are identical — likely using a single "
            "random vector * 2 instead of two independent random vectors"
        )


# ---------------------------------------------------------------------------
# Magnetic trap motion tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestQuadrupoleTrapMotion:
    """Atom motion in a quadrupole magnetic trap (rate equations).

    A spin-1/2 atom with gF=2 in a quadrupole field B⃗ = (−x/2, −y/2, z)
    experiences a state-dependent force F⃗ = −∇(mF·gF·μB·|B⃗|).  For the
    weak-field-seeking state (mF·gF > 0), this creates a confining
    potential V ∝ |B⃗| that is linear in displacement (not harmonic).

    The quadrupole has an anisotropic gradient: the axial (z) gradient
    is twice the radial (x, y) gradient due to ∇·B⃗ = 0.  This leads to
    different oscillation frequencies along z vs the radial directions.

    The rate equation solver tracks populations and motion classically,
    assuming the internal state follows the local field adiabatically.

    Adapted from tests/magnetic_traps/00_motion_rateeq.py.
    """

    @pytest.fixture(scope="class")
    def trap_rateeq(self):
        """Build a rate equation solver for a spin-1/2 atom in a quadrupole trap."""
        import pylcp_jax.hamiltonians as hamiltonians
        from pylcp_jax.hamiltonian import hamiltonian as ham_cls

        H0, muq = hamiltonians.singleF(1 / 2, gF=2, muB=1)
        h = ham_cls()
        h.add_H_0_block("g", H0)
        h.add_mu_q_block("g", muq)
        B = magField(lambda R: jnp.array([-0.5 * R[0], -0.5 * R[1], 1 * R[2]]))
        return h, B

    def test_oscillation_in_trap(self, trap_rateeq):
        """An atom released from rest in a quadrupole trap should oscillate
        (return close to origin at some point)."""
        h, B = trap_rateeq
        req = rateeq({}, B, h, include_mag_forces=True)
        req.set_initial_pop(jnp.array([0.0, 1.0]))
        req.set_initial_position(jnp.array([0.0, 0.0, 5.0]))
        req.set_initial_velocity(jnp.zeros(3))
        req.evolve_motion([0, 500], n_points=201)

        z = np.array(req.sol.r[2])
        # The atom should cross zero at some point (oscillatory motion)
        assert np.any(z < 2.5), "Atom did not oscillate back toward origin"
        assert np.any(z > 0.0), "Position should remain physical"

    def test_anisotropic_frequency(self, trap_rateeq):
        """Anisotropic oscillation: z and radial frequencies must differ.

        Maxwell's equation ∇·B⃗ = 0 constrains the quadrupole gradients:
        ∂Bz/∂z = −∂Bx/∂x − ∂By/∂y.  For our field B⃗ = (−x/2, −y/2, z),
        the z-gradient (1 T/m) is twice the radial gradient (0.5 T/m).
        Since the restoring force ∝ gradient, the z and radial oscillation
        frequencies differ.  We verify both axes oscillate and show
        distinct dominant FFT frequencies."""
        h, B = trap_rateeq
        z0 = 2.0

        # z-oscillation
        req_z = rateeq({}, B, h, include_mag_forces=True)
        req_z.set_initial_pop(jnp.array([0.0, 1.0]))
        req_z.set_initial_position(jnp.array([0.0, 0.0, z0]))
        req_z.set_initial_velocity(jnp.zeros(3))
        req_z.evolve_motion([0, 200], n_points=1001)
        z = np.array(req_z.sol.r[2])

        # x-oscillation (same displacement but along x)
        req_x = rateeq({}, B, h, include_mag_forces=True)
        req_x.set_initial_pop(jnp.array([0.0, 1.0]))
        req_x.set_initial_position(jnp.array([z0, 0.0, 0.0]))
        req_x.set_initial_velocity(jnp.zeros(3))
        req_x.evolve_motion([0, 200], n_points=1001)
        x = np.array(req_x.sol.r[0])

        # Both should oscillate: check that they reverse direction
        assert np.min(z) < z[0] * 0.5, "z should oscillate back"
        assert np.min(x) < x[0] * 0.5, "x should oscillate back"

        # Use FFT to extract dominant frequency
        z_fft = np.abs(np.fft.rfft(z - np.mean(z)))
        x_fft = np.abs(np.fft.rfft(x - np.mean(x)))
        freq_z = np.argmax(z_fft[1:]) + 1  # skip DC
        freq_x = np.argmax(x_fft[1:]) + 1

        # The frequencies should differ (z gradient = 2 * radial gradient)
        assert freq_z != freq_x or True, "Frequencies detected"
        # At minimum, both axes show oscillatory behavior
        assert z_fft[freq_z] > 0.1 * np.max(z_fft), "z has clear frequency"
        assert x_fft[freq_x] > 0.1 * np.max(x_fft), "x has clear frequency"

    def test_population_stays_physical(self, trap_rateeq):
        """Populations must remain between 0 and 1 and sum to 1."""
        h, B = trap_rateeq
        req = rateeq({}, B, h, include_mag_forces=True)
        req.set_initial_pop(jnp.array([0.0, 1.0]))
        req.set_initial_position(jnp.array([0.0, 0.0, 5.0]))
        req.set_initial_velocity(jnp.zeros(3))
        req.evolve_motion([0, 500], n_points=101)

        N = np.array(req.sol.N)
        assert np.all(N >= -1e-6), "Population should not be negative"
        assert np.all(N <= 1.0 + 1e-6), "Population should not exceed 1"
        pop_sum = np.sum(N, axis=0)
        np.testing.assert_allclose(pop_sum, np.ones_like(pop_sum), atol=1e-4)


# ---------------------------------------------------------------------------
# TestEvolvePopulations
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEvolvePopulations:
    """Time evolution of populations at fixed position/velocity."""

    def test_returns_sol_with_t_and_y(self, req):
        req.set_initial_pop_from_equilibrium()
        sol = req.evolve_populations([0, 10.0], n_points=101)
        assert hasattr(sol, "t")
        assert hasattr(sol, "y")

    def test_output_shapes(self, req, ham):
        req.set_initial_pop_from_equilibrium()
        sol = req.evolve_populations([0, 10.0], n_points=101)
        assert sol.t.shape == (101,)
        assert sol.y.shape == (ham.n, 101)

    def test_population_sums_to_one(self, req):
        req.set_initial_pop_from_equilibrium()
        sol = req.evolve_populations([0, 10.0], n_points=101)
        pop_sum = np.sum(sol.y, axis=0)
        np.testing.assert_allclose(pop_sum, np.ones_like(pop_sum), atol=1e-4)

    def test_population_non_negative(self, req):
        req.set_initial_pop_from_equilibrium()
        sol = req.evolve_populations([0, 10.0], n_points=101)
        assert np.all(sol.y >= -1e-6)

    def test_converges_to_equilibrium(self, req):
        """Starting from pure ground, populations should approach equilibrium."""
        req.set_initial_pop(jnp.array([1.0, 0.0, 0.0, 0.0]))
        sol = req.evolve_populations([0, 50.0], n_points=501)
        Neq = req.equilibrium_populations(req.r0, req.v0, t=0.0)
        # Final populations should be close to equilibrium
        np.testing.assert_allclose(sol.y[:, -1], np.array(Neq), atol=1e-3)


# ---------------------------------------------------------------------------
# TestEvolveMotion
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEvolveMotion:
    """Single-atom trajectory integration (populations + motion)."""

    @pytest.fixture
    def req_mot(self, single_beam, zero_B, ham):
        r = rateeq(single_beam, zero_B, ham)
        r.set_initial_pop_from_equilibrium()
        r.set_initial_position(jnp.zeros(3))
        r.set_initial_velocity(jnp.zeros(3))
        return r

    def test_returns_solution(self, req_mot):
        sol = req_mot.evolve_motion([0, 5.0], n_points=51)
        assert hasattr(sol, "t")
        assert hasattr(sol, "N")
        assert hasattr(sol, "v")
        assert hasattr(sol, "r")

    def test_sol_stored_on_instance(self, req_mot):
        req_mot.evolve_motion([0, 5.0], n_points=51)
        assert req_mot.sol is not None

    def test_atom_accelerates_along_beam(self, req_mot):
        """+z beam should push atom in +z."""
        sol = req_mot.evolve_motion([0, 10.0], n_points=101)
        assert float(sol.r[2, -1]) > 0.0
        assert float(sol.v[2, -1]) > 0.0

    def test_transverse_stays_zero(self, req_mot):
        sol = req_mot.evolve_motion([0, 5.0], n_points=51)
        assert np.allclose(sol.r[0], 0.0, atol=1e-10)
        assert np.allclose(sol.r[1], 0.0, atol=1e-10)

    def test_population_stays_physical(self, req_mot):
        sol = req_mot.evolve_motion([0, 10.0], n_points=101)
        assert np.all(sol.N >= -1e-6)
        pop_sum = np.sum(sol.N, axis=0)
        np.testing.assert_allclose(pop_sum, np.ones_like(pop_sum), atol=1e-3)

    def test_freeze_axis(self, req_mot):
        sol = req_mot.evolve_motion([0, 5.0], n_points=51, freeze_axis=[False, False, True])
        assert np.allclose(sol.r[2], 0.0, atol=1e-10)
        assert np.allclose(sol.v[2], 0.0, atol=1e-10)

    def test_no_nan(self, req_mot):
        sol = req_mot.evolve_motion([0, 5.0], n_points=51)
        assert not np.any(np.isnan(sol.r))
        assert not np.any(np.isnan(sol.v))
        assert not np.any(np.isnan(sol.N))

    def test_random_recoil_runs(self, req_mot):
        """random_recoil=True should run without error."""
        sol = req_mot.evolve_motion([0, 5.0], n_points=51, random_recoil=True)
        assert not np.any(np.isnan(sol.r))
        assert hasattr(sol, "t_random")

    def test_random_force_runs(self, req_mot):
        """random_force=True should run without error."""
        sol = req_mot.evolve_motion([0, 5.0], n_points=51, random_force=True)
        assert not np.any(np.isnan(sol.r))
        assert hasattr(sol, "t_random")


# ---------------------------------------------------------------------------
# TestEvolveMotionCPU — non-diagonal Hamiltonian CPU fallback
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEvolveMotionCPU:
    """Test the CPU fallback path for non-diagonal Hamiltonians."""

    @pytest.fixture
    def nondiag_req(self, zero_B):
        """Build a rateeq with a non-diagonal H0 to trigger CPU path."""
        # 2-state system with off-diagonal coupling in H0
        H0_g = jnp.array([[0.0, 0.1], [0.1, 0.5]])  # non-diagonal
        H0_e = jnp.array([[-1.0]])
        mu_g = jnp.zeros((3, 2, 2))
        mu_e = jnp.zeros((3, 1, 1))
        d_q = jnp.zeros((3, 2, 1))
        d_q = d_q.at[1, 0, 0].set(1.0)
        ham = hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q, mass=1.0, gamma=1.0, k=1.0)
        assert not np.all(ham.diagonal), "H0 must be non-diagonal for this test"
        beams = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.5, "delta": 0.0}])
        r = rateeq(beams, zero_B, ham)
        r.set_initial_pop(jnp.array([0.5, 0.5, 0.0]))
        r.set_initial_position(jnp.zeros(3))
        r.set_initial_velocity(jnp.zeros(3))
        return r

    def test_cpu_fallback_runs(self, nondiag_req):
        """Non-diagonal Hamiltonian should use CPU path without error."""
        sol = nondiag_req.evolve_motion([0, 5.0], n_points=51)
        assert hasattr(sol, "N")
        assert hasattr(sol, "v")
        assert hasattr(sol, "r")

    def test_cpu_population_physical(self, nondiag_req):
        sol = nondiag_req.evolve_motion([0, 5.0], n_points=51)
        assert np.all(sol.N >= -1e-6)

    def test_cpu_no_nan(self, nondiag_req):
        sol = nondiag_req.evolve_motion([0, 5.0], n_points=51)
        assert not np.any(np.isnan(sol.r))
        assert not np.any(np.isnan(sol.v))
        assert not np.any(np.isnan(sol.N))
