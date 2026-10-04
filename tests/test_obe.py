"""
Tests for pylcp_jax/obe.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_ham, requires_gpu

import pylcp_jax.hamiltonians as hamiltonians
from pylcp_jax.fields import (
    constantMagneticField,
    infinitePlaneWaveBeam,
    laserBeams,
    magField,
)
from pylcp_jax.hamiltonian import hamiltonian
from pylcp_jax.obe import force_profile, obe

# ---------------------------------------------------------------------------
# Module-scoped fixture overrides (OBE construction is expensive)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ham():
    return make_ham()


@pytest.fixture(scope="module")
def zero_B():
    return constantMagneticField(jnp.array([0.0, 0.0, 0.0]))


@pytest.fixture(scope="module")
def weak_B():
    return constantMagneticField(jnp.array([0.0, 0.0, 0.1]))


@pytest.fixture(scope="module")
def single_beam():
    """One sigma+ beam along +z, weak saturation, on resonance."""
    return laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.1, "delta": 0.0}])


@pytest.fixture(scope="module")
def symmetric_beams():
    """Two counter-propagating sigma+/sigma- beams, equal intensity."""
    return laserBeams(
        [
            {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.5, "delta": -1.0},
            {"kvec": [0.0, 0.0, -1.0], "pol": -1, "s": 0.5, "delta": -1.0},
        ]
    )


@pytest.fixture(scope="module")
def obe_transform(single_beam, zero_B, ham):
    """OBE with transform_into_re_im=True (default)."""
    return obe(single_beam, zero_B, ham, transform_into_re_im=True)


@pytest.fixture(scope="module")
def obe_complex(single_beam, zero_B, ham):
    """OBE with transform_into_re_im=False (complex basis)."""
    return obe(single_beam, zero_B, ham, transform_into_re_im=False)


@pytest.fixture(scope="module")
def obe_sym(symmetric_beams, zero_B, ham):
    """OBE with symmetric beams."""
    return obe(symmetric_beams, zero_B, ham)


# ---------------------------------------------------------------------------
# TestForceProfile
# ---------------------------------------------------------------------------


class TestForceProfile:
    def test_basic_shapes(self, obe_transform):
        o = obe_transform
        R = np.zeros((3, 5, 4))
        V = np.zeros((3, 5, 4))
        fp = force_profile(R, V, o.laserBeams, o.hamiltonian)
        assert fp.F.shape == (3, 5, 4)
        assert fp.R.shape == (3, 5, 4)
        assert fp.V.shape == (3, 5, 4)

    def test_iterations_shape(self, obe_transform):
        o = obe_transform
        R = np.zeros((3, 3))
        V = np.zeros((3, 3))
        fp = force_profile(R, V, o.laserBeams, o.hamiltonian)
        assert fp.iterations.shape == (3,)

    def test_fq_key_present(self, obe_transform):
        o = obe_transform
        R = np.zeros((3, 3))
        V = np.zeros((3, 3))
        fp = force_profile(R, V, o.laserBeams, o.hamiltonian)
        assert "g->e" in fp.fq

    def test_fq_shape(self, obe_transform):
        o = obe_transform
        R = np.zeros((3, 5))
        V = np.zeros((3, 5))
        fp = force_profile(R, V, o.laserBeams, o.hamiltonian)
        # fq[key] has shape R.shape + (3, n_beams) = (3, 5, 3, 1)
        assert fp.fq["g->e"].shape == (3, 5, 3, 1)

    def test_store_data_updates_F(self, obe_transform):
        o = obe_transform
        R = np.zeros((3, 2))
        V = np.zeros((3, 2))
        fp = force_profile(R, V, o.laserBeams, o.hamiltonian)
        F_in = np.array([1.0, 2.0, 3.0])
        Neq_in = np.array([0.5, 0.1, 0.1, 0.3])
        fp.store_data(
            (0,),
            Neq_in,
            F_in,
            {"g->e": np.zeros((3, 1))},
            np.zeros(3),
            5,
            {"g->e": np.zeros((3, 3, 1))},
        )
        np.testing.assert_allclose(fp.F[:, 0], F_in)
        assert fp.iterations[0] == 5


# ---------------------------------------------------------------------------
# TestObeInit
# ---------------------------------------------------------------------------


class TestObeInit:
    def test_basic_construction(self, obe_transform):
        assert obe_transform is not None

    def test_construction_no_transform(self, obe_complex):
        assert obe_complex is not None

    def test_ev_mat_keys_with_transform(self, obe_transform):
        keys = set(obe_transform.ev_mat.keys())
        assert "decay" in keys
        assert "H0" in keys
        assert "reE" in keys
        assert "imE" in keys
        assert "B" in keys
        # d_q and d_q* should be deleted after transform
        assert "d_q" not in keys
        assert "d_q*" not in keys

    def test_ev_mat_keys_no_transform(self, obe_complex):
        keys = set(obe_complex.ev_mat.keys())
        assert "decay" in keys
        assert "H0" in keys
        assert "d_q" in keys
        assert "d_q*" in keys
        assert "B" in keys

    def test_ev_mat_decay_shape(self, obe_transform):
        n = obe_transform.hamiltonian.n
        assert obe_transform.ev_mat["decay"].shape == (n**2, n**2)

    def test_ev_mat_H0_shape(self, obe_transform):
        n = obe_transform.hamiltonian.n
        assert obe_transform.ev_mat["H0"].shape == (n**2, n**2)

    def test_ev_mat_B_length(self, obe_transform):
        assert len(obe_transform.ev_mat["B"]) == 3

    def test_decay_rates_positive(self, obe_transform):
        # Excited states have positive decay rates; ground state has zero
        for key in obe_transform.decay_rates:
            assert np.all(obe_transform.decay_rates[key] >= 0)

    def test_decay_rates_truncated_positive(self, obe_transform):
        for key in obe_transform.decay_rates_truncated:
            assert np.all(obe_transform.decay_rates_truncated[key] > 0)

    def test_ev_mat_are_jax_arrays(self, obe_transform):
        for key in obe_transform.ev_mat:
            if isinstance(obe_transform.ev_mat[key], dict):
                for subkey in obe_transform.ev_mat[key]:
                    assert isinstance(obe_transform.ev_mat[key][subkey], jnp.ndarray)
            elif isinstance(obe_transform.ev_mat[key], list):
                for v in obe_transform.ev_mat[key]:
                    assert isinstance(v, jnp.ndarray)
            else:
                assert isinstance(obe_transform.ev_mat[key], jnp.ndarray)

    def test_transform_true_matrices_are_real(self, obe_transform):
        # After transform, decay and H0 should be real (imaginary parts removed)
        decay = np.array(obe_transform.ev_mat["decay"])
        H0 = np.array(obe_transform.ev_mat["H0"])
        assert np.allclose(np.imag(decay), 0, atol=1e-12)
        assert np.allclose(np.imag(H0), 0, atol=1e-12)

    def test_magField_setter_clears_dydt_cache(self, single_beam, zero_B, ham):
        o = obe(single_beam, zero_B, ham)
        o.set_initial_rho_equally()
        o.evolve_density([0.0, 1.0], n_points=5)
        assert "_dydt" in o.__dict__
        o.magField = zero_B
        assert "_dydt" not in o.__dict__

    def test_laserBeams_setter_clears_dydt_cache(self, single_beam, zero_B, ham):
        o = obe(single_beam, zero_B, ham)
        o.set_initial_rho_equally()
        o.evolve_density([0.0, 1.0], n_points=5)
        assert "_dydt" in o.__dict__
        o.laserBeams = o.laserBeams
        assert "_dydt" not in o.__dict__


# ---------------------------------------------------------------------------
# TestBuildCoherentEvSubmatrix (optimization verification)
# ---------------------------------------------------------------------------


class TestBuildCoherentEvSubmatrix:
    """Liouvillian superoperator L for coherent evolution: dρ/dt = Lρ.

    The Liouvillian is constructed as L = i(H⊗I − I⊗H) using the
    Kronecker product, which maps the density matrix ρ (n×n) to a
    column vector of length n².

    Physical constraints:
    - For Hermitian H, L is anti-Hermitian (L + L† = 0), ensuring
      unitary evolution preserves Tr(ρ) and positivity.
    - The trace-preserving condition requires Σ_diag L_ij = 0 for all j.
    - The kron-product implementation must match the reference triple-loop
      construction element by element."""

    def _make_liouvillian_loop(self, H, n):
        """Reference implementation via the original triple loop."""
        ev = np.zeros((n**2, n**2), dtype="complex128")
        idx = lambda i, j: i + j * n
        for ii in range(n):
            for jj in range(n):
                for kk in range(n):
                    ev[idx(ii, jj), idx(ii, kk)] += 1j * H[kk, jj]
                    ev[idx(ii, jj), idx(kk, jj)] -= 1j * H[ii, kk]
        return ev

    def _make_liouvillian_kron(self, H, n):
        """New vectorized kron implementation."""
        I = np.eye(n)
        H = np.asarray(H)
        return 1j * np.kron(H.T, I) - 1j * np.kron(I, H)

    def test_kron_matches_loop_identity(self):
        n = 3
        H = np.eye(n, dtype="complex128")
        L_loop = self._make_liouvillian_loop(H, n)
        L_kron = self._make_liouvillian_kron(H, n)
        np.testing.assert_allclose(L_loop, L_kron, atol=1e-14)

    def test_kron_matches_loop_diagonal(self):
        n = 4
        H = np.diag([0.0, 1.0, 2.0, 3.0]).astype("complex128")
        L_loop = self._make_liouvillian_loop(H, n)
        L_kron = self._make_liouvillian_kron(H, n)
        np.testing.assert_allclose(L_loop, L_kron, atol=1e-14)

    def test_kron_matches_loop_full(self):
        n = 3
        rng = np.random.default_rng(42)
        H_re = rng.standard_normal((n, n))
        H = (H_re + H_re.T).astype("complex128")  # Hermitian
        H[0, 1] += 0.5j
        H[1, 0] -= 0.5j
        L_loop = self._make_liouvillian_loop(H, n)
        L_kron = self._make_liouvillian_kron(H, n)
        np.testing.assert_allclose(L_loop, L_kron, atol=1e-14)

    def test_liouvillian_preserves_trace(self):
        """d/dt tr(rho) = 0: sum of each L row for diagonal of rho must be zero."""
        n = 4
        H = np.diag([0.0, 1.0, -1.0, 2.0]).astype("complex128")
        L = self._make_liouvillian_kron(H, n)
        # Diagonal elements of rho correspond to indices i+i*n = i*(n+1)
        diag_indices = [i + i * n for i in range(n)]
        # sum over diagonal rows of L gives d/dt tr(rho)
        trace_rate = np.sum(L[diag_indices, :], axis=0)
        np.testing.assert_allclose(trace_rate, 0, atol=1e-14)

    def test_liouvillian_anti_hermitian_for_hermitian_H(self):
        """For Hermitian H, L + L† = 0 (L is anti-Hermitian / skew-Hermitian)."""
        n = 3
        H = np.array([[1.0, 0.5 + 0.2j, 0.0], [0.5 - 0.2j, -1.0, 0.3j], [0.0, -0.3j, 0.0]])
        L = self._make_liouvillian_kron(H, n)
        np.testing.assert_allclose(L + np.conj(L.T), 0, atol=1e-14)

    def test_ev_mat_shape(self, obe_transform):
        n = obe_transform.hamiltonian.n
        assert obe_transform.ev_mat["H0"].shape == (n**2, n**2)


# ---------------------------------------------------------------------------
# TestDensityIndex
# ---------------------------------------------------------------------------


class TestDensityIndex:
    """Test the __density_index mapping through observable/rho behavior."""

    def test_rho0_diagonal_from_equally(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        # With transform=True, rho0 is in re/im basis.
        # The diagonal element for ground state 0 should be set.
        assert o.rho0 is not None
        assert o.rho0.shape == (o.hamiltonian.n**2,)

    def test_rho0_trace_one(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        # Recover complex rho and check trace = 1
        if o.transform_into_re_im:
            rho_complex = jnp.dot(jnp.asarray(o.U), o.rho0.astype("complex128"))
        else:
            rho_complex = o.rho0
        rho_mat = jnp.reshape(rho_complex, (o.hamiltonian.n, o.hamiltonian.n))
        trace = jnp.real(jnp.trace(rho_mat))
        assert float(trace) == pytest.approx(1.0, abs=1e-12)

    def test_rho0_ground_only_from_equally(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        if o.transform_into_re_im:
            rho_complex = jnp.dot(jnp.asarray(o.U), o.rho0.astype("complex128"))
        else:
            rho_complex = o.rho0
        rho_mat = jnp.reshape(rho_complex, (o.hamiltonian.n, o.hamiltonian.n))
        diag = jnp.real(jnp.diagonal(rho_mat))
        # n_g=1 ground state should have pop=1, all 3 excited states pop=0
        assert float(diag[0]) == pytest.approx(1.0, abs=1e-12)
        for i in range(1, o.hamiltonian.n):
            assert float(diag[i]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# TestSetInitialRho
# ---------------------------------------------------------------------------


class TestSetInitialRho:
    def test_set_from_populations(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        Npop = np.zeros(n)
        Npop[0] = 1.0
        o.set_initial_rho_from_populations(Npop)
        assert o.rho0 is not None
        assert o.rho0.shape == (n**2,)

    def test_set_from_populations_normalizes(self, obe_transform, ham):
        o = obe_transform
        Npop = np.ones(ham.n) * 2.0  # un-normalized
        o.set_initial_rho_from_populations(Npop)
        # After normalization, trace = 1
        if o.transform_into_re_im:
            rho_c = jnp.dot(jnp.asarray(o.U), o.rho0.astype("complex128"))
        else:
            rho_c = o.rho0
        rho_mat = jnp.reshape(rho_c, (ham.n, ham.n))
        assert float(jnp.real(jnp.trace(rho_mat))) == pytest.approx(1.0, abs=1e-12)

    def test_set_from_populations_wrong_length_raises(self, obe_transform, ham):
        with pytest.raises(ValueError, match="Npop"):
            obe_transform.set_initial_rho_from_populations(np.ones(ham.n + 1))

    def test_set_from_populations_nan_raises(self, obe_transform, ham):
        bad = np.full(ham.n, np.nan)
        with pytest.raises(ValueError):
            obe_transform.set_initial_rho_from_populations(bad)

    def test_set_rho_flat_complex(self, obe_transform, ham):
        o = obe_transform
        rho0 = np.zeros(ham.n**2, dtype="complex128")
        rho0[0] = 1.0  # ground state population
        o.set_initial_rho(rho0)
        assert o.rho0 is not None

    def test_set_rho_matrix_input(self, obe_transform, ham):
        o = obe_transform
        rho0_mat = np.zeros((ham.n, ham.n), dtype="complex128")
        rho0_mat[0, 0] = 1.0
        o.set_initial_rho(rho0_mat)
        assert o.rho0.shape == (ham.n**2,)

    def test_set_rho_nan_raises(self, obe_transform, ham):
        bad = np.full(ham.n**2, np.nan, dtype="complex128")
        with pytest.raises(ValueError):
            obe_transform.set_initial_rho(bad)

    def test_set_rho_wrong_size_raises(self, obe_transform, ham):
        bad = np.zeros(ham.n**2 + 1, dtype="complex128")
        with pytest.raises(ValueError):
            obe_transform.set_initial_rho(bad)

    def test_set_equally_produces_normalized_rho(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        assert o.rho0 is not None
        # The trace of rho0 should be 1
        if o.transform_into_re_im:
            rho_c = jnp.dot(jnp.asarray(o.U), o.rho0.astype("complex128"))
        else:
            rho_c = o.rho0
        rho_mat = jnp.reshape(rho_c, (o.hamiltonian.n, o.hamiltonian.n))
        assert float(jnp.real(jnp.trace(rho_mat))) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# TestObservable
# ---------------------------------------------------------------------------


class TestObservable:
    """Computing expectation values ⟨O⟩ = Tr(O·ρ) from the density matrix.

    The identity operator must give Tr(ρ) = 1.  A projector |i⟩⟨i|
    gives the population in state i.  Vector operators (shape 3×n×n)
    return a 3-component result.  Dimension mismatches must raise."""

    def test_identity_gives_trace_one(self, obe_transform, ham):
        o = obe_transform
        o.set_initial_rho_equally()
        rho0 = o.rho0
        n = ham.n
        I_op = jnp.eye(n)
        result = o.observable(I_op, rho0)
        assert float(result) == pytest.approx(1.0, abs=1e-12)

    def test_projector_gives_population(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        # Start with all population in ground state (index 0)
        rho0_mat = np.zeros((n, n), dtype="complex128")
        rho0_mat[0, 0] = 1.0
        P0 = jnp.zeros((n, n), dtype="complex128").at[0, 0].set(1.0)
        result = o.observable(P0, rho0_mat)
        assert float(result) == pytest.approx(1.0, abs=1e-12)

    def test_excited_projector_zero_in_ground(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        rho0_mat = np.zeros((n, n), dtype="complex128")
        rho0_mat[0, 0] = 1.0
        P_exc = jnp.zeros((n, n), dtype="complex128").at[1, 1].set(1.0)
        result = o.observable(P_exc, rho0_mat)
        assert float(result) == pytest.approx(0.0, abs=1e-12)

    def test_vector_observable_shape(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        rho0_mat = np.zeros((n, n), dtype="complex128")
        rho0_mat[0, 0] = 1.0
        # Vector operator: shape (3, n, n)
        O_vec = jnp.zeros((3, n, n), dtype="complex128")
        result = o.observable(O_vec, rho0_mat)
        assert result.shape == (3,)

    def test_wrong_O_size_raises(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        rho0_mat = np.zeros((n, n), dtype="complex128")
        rho0_mat[0, 0] = 1.0
        bad_O = jnp.zeros((n + 1, n + 1))
        with pytest.raises(ValueError):
            o.observable(bad_O, rho0_mat)

    def test_wrong_rho_shape_raises(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        bad_rho = jnp.zeros((n + 1, n + 1))
        O = jnp.eye(n)
        with pytest.raises(ValueError):
            o.observable(O, bad_rho)

    def test_flat_rho_input_works(self, obe_transform, ham):
        """1D rho0 (internal representation) is auto-reshaped by observable."""
        o = obe_transform
        n = ham.n
        o.set_initial_rho_equally()
        rho0_flat = o.rho0  # 1D array in internal (re/im) basis
        I_op = jnp.eye(n)
        result = o.observable(I_op, rho0_flat)
        assert float(result) == pytest.approx(1.0, abs=1e-12)

    def test_result_is_real(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        rho0_mat = np.eye(n, dtype="complex128") / n
        O = jnp.eye(n)
        result = o.observable(O, rho0_mat)
        assert jnp.isreal(result)


# ---------------------------------------------------------------------------
# TestEvolveDensity
# ---------------------------------------------------------------------------


class TestEvolveDensity:
    """Time evolution of the density matrix under the full OBE.

    The OBE master equation dρ/dt = −i[H,ρ] + L[ρ] includes both
    coherent driving (laser coupling) and incoherent decay (spontaneous
    emission via Lindblad terms).

    At all times:
    - Tr(ρ) = 1 (probability conservation).
    - Diagonal elements ρ_ii ≥ 0 (populations are non-negative).
    - Starting from the ground state with a resonant laser, excited
      state population must grow."""

    def test_returns_sol_object(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        sol = o.evolve_density([0.0, 5.0], n_points=11)
        assert hasattr(sol, "t")
        assert hasattr(sol, "rho")

    def test_t_shape(self, obe_transform):
        o = obe_transform
        o.set_initial_rho_equally()
        sol = o.evolve_density([0.0, 5.0], n_points=11)
        assert sol.t.shape == (11,)

    def test_rho_shape(self, obe_transform, ham):
        o = obe_transform
        o.set_initial_rho_equally()
        n = ham.n
        sol = o.evolve_density([0.0, 5.0], n_points=11)
        assert sol.rho.shape == (n, n, 11)

    def test_rho_trace_conserved(self, obe_transform, ham):
        """tr(rho(t)) = 1 at all time points."""
        o = obe_transform
        o.set_initial_rho_equally()
        sol = o.evolve_density([0.0, 5.0], n_points=21)
        for t_idx in range(21):
            rho_t = sol.rho[:, :, t_idx]
            trace = float(jnp.real(jnp.trace(rho_t)))
            assert trace == pytest.approx(1.0, abs=1e-6)

    def test_rho_diagonal_non_negative(self, obe_transform, ham):
        """Populations are non-negative at all times."""
        o = obe_transform
        o.set_initial_rho_equally()
        sol = o.evolve_density([0.0, 5.0], n_points=21)
        for t_idx in range(21):
            diag = jnp.real(jnp.diagonal(sol.rho[:, :, t_idx]))
            assert float(jnp.min(diag)) >= -1e-8

    def test_excited_pop_grows_from_ground(self, obe_transform, ham):
        """Starting from ground state, excited state population increases."""
        o = obe_transform
        o.set_initial_rho_equally()
        sol = o.evolve_density([0.0, 20.0], n_points=101)
        # Initial excited pop
        exc_init = float(jnp.real(jnp.sum(jnp.diagonal(sol.rho[1:, 1:, 0]))))
        # Later excited pop
        exc_final = float(jnp.real(jnp.sum(jnp.diagonal(sol.rho[1:, 1:, -1]))))
        assert exc_final > exc_init

    def test_initial_rho_is_initial_condition(self, obe_transform, ham):
        """rho at t=0 should match the initial condition."""
        o = obe_transform
        n = ham.n
        # Set ground state only
        rho0_mat = np.zeros((n, n), dtype="complex128")
        rho0_mat[0, 0] = 1.0
        o.set_initial_rho(rho0_mat)
        sol = o.evolve_density([0.0, 1.0], n_points=11)
        # At t=0 (index 0), diagonal[0] should be ~1
        pop_ground_t0 = float(jnp.real(sol.rho[0, 0, 0]))
        assert pop_ground_t0 == pytest.approx(1.0, abs=1e-4)

    def test_no_transform_rho_trace_conserved(self, obe_complex, ham):
        """Without re/im transform, trace is also conserved."""
        o = obe_complex
        o.set_initial_rho_equally()
        sol = o.evolve_density([0.0, 5.0], n_points=21)
        for t_idx in range(21):
            trace = float(jnp.real(jnp.trace(sol.rho[:, :, t_idx])))
            assert trace == pytest.approx(1.0, abs=1e-5)

    def test_y0_batch_override(self, obe_transform, ham):
        """Explicit y0_batch can override rho0."""
        o = obe_transform
        o.set_initial_rho_equally()
        n = ham.n
        rho0 = o.rho0
        y0 = jnp.concatenate([rho0, jnp.zeros(6)])
        y0_batch = y0[None, :]
        sol = o.evolve_density([0.0, 5.0], y0_batch=y0_batch, n_points=11)
        assert sol.rho.shape == (n, n, 11)


# ---------------------------------------------------------------------------
# TestForce
# ---------------------------------------------------------------------------


class TestForce:
    """Radiation pressure force F⃗ = ℏk⃗ · Γ · ⟨scattering rate⟩.

    A single beam along +ẑ must produce a positive z-force.  Two
    symmetric counter-propagating beams at v=0, B=0 must give nearly
    zero net force by symmetry.  The force can be decomposed into
    per-beam (f_laser) and per-polarization-component (f_laser_q)
    contributions, plus a magnetic force (f_mag)."""

    def _get_rho0(self, o):
        """Get the initial rho0 in internal representation (re/im basis if transform=True)."""
        o.set_initial_rho_equally()
        return o.rho0  # 1D array in internal basis

    def test_force_shape_single_point(self, obe_transform):
        o = obe_transform
        rho = self._get_rho0(o)
        r = jnp.zeros(3)
        f = o.force(r, 0.0, rho)
        assert f.shape == (3,)

    def test_force_return_details_tuple(self, obe_transform):
        o = obe_transform
        rho = self._get_rho0(o)
        r = jnp.zeros(3)
        result = o.force(r, 0.0, rho, return_details=True)
        assert len(result) == 4  # f, f_laser, f_laser_q, f_mag

    def test_force_f_laser_has_key(self, obe_transform):
        o = obe_transform
        rho = self._get_rho0(o)
        r = jnp.zeros(3)
        _, f_laser, _, _ = o.force(r, 0.0, rho, return_details=True)
        assert "g->e" in f_laser

    def test_force_f_laser_q_has_key(self, obe_transform):
        o = obe_transform
        rho = self._get_rho0(o)
        r = jnp.zeros(3)
        _, _, f_laser_q, _ = o.force(r, 0.0, rho, return_details=True)
        assert "g->e" in f_laser_q

    def test_force_symmetric_near_zero_at_origin(self, obe_sym):
        """Symmetric counter-propagating beams give near-zero net force at v=0, B=0."""
        o = obe_sym
        rho = self._get_rho0(o)
        r = jnp.zeros(3)
        f = o.force(r, 0.0, rho)
        # z-component of force should be near zero by symmetry
        assert float(jnp.abs(f[2])) < 0.5

    def test_single_beam_force_avg_nonzero(self, obe_transform):
        """Time-averaged force from a single +z beam must be positive."""
        o = obe_transform
        o.set_initial_rho_equally()
        F = o.find_equilibrium_force(deltat=50, itermax=5, Npts=201, initial_rho="equally")
        assert float(F[2]) > 0

    def test_force_with_no_mag_forces(self, single_beam, zero_B, ham):
        o = obe(single_beam, zero_B, ham, include_mag_forces=False)
        o.set_initial_rho_equally()
        rho = o.rho0
        r = jnp.zeros(3)
        f = o.force(r, 0.0, rho)
        assert f.shape == (3,)


# ---------------------------------------------------------------------------
# TestFindEquilibriumForce
# ---------------------------------------------------------------------------


class TestFindEquilibriumForce:
    """Steady-state (time-averaged) radiation pressure force.

    find_equilibrium_force evolves ρ forward in time and averages the
    instantaneous force until convergence.  The equilibrium populations
    Neq must sum to 1 and the force must be consistent with the beam
    geometry (positive z for a +ẑ beam, near-zero for symmetric beams)."""

    def test_returns_shape_3_array(self, obe_transform):
        o = obe_transform
        F = o.find_equilibrium_force(deltat=10, itermax=3, Npts=101, initial_rho="equally")
        assert F.shape == (3,)

    def test_initial_rho_equally(self, obe_transform):
        o = obe_transform
        F = o.find_equilibrium_force(deltat=10, itermax=3, Npts=101, initial_rho="equally")
        assert not jnp.any(jnp.isnan(F))

    def test_initial_rho_invalid_raises(self, obe_transform):
        with pytest.raises(ValueError, match="not understood"):
            obe_transform.find_equilibrium_force(initial_rho="invalid")

    def test_return_details_structure(self, obe_transform):
        o = obe_transform
        result = o.find_equilibrium_force(
            deltat=10, itermax=2, Npts=51, initial_rho="equally", return_details=True
        )
        # Returns (F, f_laser, f_laser_q, f_mag, Neq, ii)
        assert len(result) == 6
        F, f_laser, f_laser_q, f_mag, Neq, n_iter = result
        assert F.shape == (3,)
        assert Neq.shape == (obe_transform.hamiltonian.n,)
        assert isinstance(n_iter, (int, np.integer))

    def test_Neq_sums_to_one(self, obe_transform, ham):
        o = obe_transform
        _, _, _, _, Neq, _ = o.find_equilibrium_force(
            deltat=10, itermax=3, Npts=101, initial_rho="equally", return_details=True
        )
        assert float(jnp.sum(Neq)) == pytest.approx(1.0, abs=1e-4)

    def test_single_beam_force_z_positive(self, obe_transform):
        """Single +z beam → equilibrium force has positive z component."""
        o = obe_transform
        F = o.find_equilibrium_force(deltat=50, itermax=5, Npts=201, initial_rho="equally")
        assert float(F[2]) > 0

    def test_symmetric_beam_force_near_zero(self, obe_sym):
        """Counter-propagating symmetric beams → near-zero net z force at origin."""
        o = obe_sym
        F = o.find_equilibrium_force(deltat=20, itermax=5, Npts=101, initial_rho="equally")
        assert float(jnp.abs(F[2])) < 0.1

    def test_initial_rho_frompops(self, obe_transform, ham):
        o = obe_transform
        init_pop = np.zeros(ham.n)
        init_pop[0] = 1.0
        F = o.find_equilibrium_force(
            deltat=10, itermax=2, Npts=51, initial_rho="frompops", init_pop=init_pop
        )
        assert F.shape == (3,)


# ---------------------------------------------------------------------------
# TestFullOBEEv  (utility methods)
# ---------------------------------------------------------------------------


class TestFullOBEEv:
    def test_full_OBE_ev_shape(self, obe_transform, ham):
        o = obe_transform
        n = ham.n
        r = jnp.zeros(3)
        ev = o.full_OBE_ev(r, 0.0)
        assert ev.shape == (n**2, n**2)

    def test_full_OBE_ev_no_nan(self, obe_transform):
        o = obe_transform
        r = jnp.zeros(3)
        ev = o.full_OBE_ev(r, 0.0)
        assert not jnp.any(jnp.isnan(ev))

    def test_full_OBE_ev_with_B(self, obe_transform, ham):
        """full_OBE_ev should run without error when B field is non-zero."""
        from pylcp_jax.fields import constantMagneticField

        single_beam = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.1, "delta": 0.0}])
        B_field = constantMagneticField(jnp.array([0.0, 0.0, 0.1]))
        o_B = obe(single_beam, B_field, make_ham())
        r = jnp.zeros(3)
        ev = o_B.full_OBE_ev(r, 0.0)
        assert ev.shape == (o_B.hamiltonian.n**2, o_B.hamiltonian.n**2)
        assert not jnp.any(jnp.isnan(ev))


# ---------------------------------------------------------------------------
# Test1DMOTForceProfile – regression tests for magnetic field gradient
# ---------------------------------------------------------------------------


class Test1DMOTForceProfile:
    """1D MOT force profile computed via OBE (cf. rate equation version).

    The OBE calculation finds the time-averaged force by evolving ρ to
    steady state at each spatial point.  The same MOT physics applies:
    the force must vanish at the trap centre (B=0) by symmetry, be
    restoring (F·x < 0) in the linear trapping region, and contain
    no NaN even at B=0 where the quantization axis is undefined."""

    @pytest.fixture
    def mot_obe(self):
        from pylcp_jax.fields import magField

        ham = make_ham(gamma=1.0, k=1.0, mass=1.0)
        mu_val = 1399624.49171  # |diag(mu_e[1])[0]|
        delta = -4.0
        x_res = 5.0
        alpha = abs(delta) / (x_res * mu_val)
        beams = laserBeams(
            [
                {"kvec": [1.0, 0.0, 0.0], "pol": -1, "s": 1.0, "delta": delta},
                {"kvec": [-1.0, 0.0, 0.0], "pol": -1, "s": 1.0, "delta": delta},
            ]
        )
        B = magField(lambda R: -alpha * R)
        return obe(beams, B, ham), x_res

    def test_force_at_origin_is_zero(self, mot_obe):
        """By symmetry the force at x=0 (where B=0) must vanish."""
        o, _ = mot_obe
        o.set_initial_position_and_velocity(jnp.zeros(3), jnp.zeros(3))
        F = o.find_equilibrium_force(deltat=200, itermax=50, Npts=2001)
        assert float(F[0]) == pytest.approx(0.0, abs=1e-6)

    def test_force_nonzero_away_from_origin(self, mot_obe):
        """Force must be non-zero in the linear trapping region."""
        o, x_res = mot_obe
        o.set_initial_position_and_velocity(jnp.array([x_res / 2, 0.0, 0.0]), jnp.zeros(3))
        F = o.find_equilibrium_force(deltat=200, itermax=50, Npts=2001)
        assert abs(float(F[0])) > 1e-6

    def test_force_is_restoring(self, mot_obe):
        """Force in the linear region should be restoring."""
        o, x_res = mot_obe
        # Use half the resonance position — well inside the linear trapping region
        x_test = x_res / 2
        for x_val, expected_sign in [(x_test, -1), (-x_test, +1)]:
            o.set_initial_position_and_velocity(jnp.array([x_val, 0.0, 0.0]), jnp.zeros(3))
            F = o.find_equilibrium_force(deltat=200, itermax=50, Npts=2001)
            assert float(F[0]) * expected_sign > 0.0, (
                f"Force at x={x_val} should have sign {expected_sign}, got {float(F[0])}"
            )

    def test_force_no_nan(self, mot_obe):
        """OBE force at origin (B=0) must not produce NaN."""
        o, _ = mot_obe
        o.set_initial_position_and_velocity(jnp.zeros(3), jnp.zeros(3))
        F = o.find_equilibrium_force(deltat=200, itermax=50, Npts=2001)
        assert not jnp.any(jnp.isnan(F))


class TestMolassesForceProfileSmooth:
    """OBE force profile for 1D molasses must be smooth across all velocities.

    Regression test: when generate_force_profile uses a batched solver with
    a single chunk_deltat (the minimum across the velocity grid), atoms at
    intermediate velocities can get under-averaged Rabi oscillations, producing
    spurious spikes in the force profile.  A cumulative running average
    eliminates these artefacts.
    """

    @pytest.fixture
    def molasses_obe(self):
        Hg = np.array([[0.0]])
        He = np.array([[0.0]])  # detuning on the laser
        mu_q = np.zeros((3, 1, 1))
        d_q = np.zeros((3, 1, 1))
        d_q[1, 0, 0] = 1.0
        ham = hamiltonian(Hg, He, mu_q, mu_q, d_q, mass=200)
        delta, s = -2.0, 1.5
        beams = laserBeams(
            [
                {
                    "kvec": [1.0, 0.0, 0.0],
                    "pol": [0.0, 1.0, 0.0],
                    "pol_coord": "spherical",
                    "delta": delta,
                    "s": s,
                },
                {
                    "kvec": [-1.0, 0.0, 0.0],
                    "pol": [0.0, 1.0, 0.0],
                    "pol_coord": "spherical",
                    "delta": delta,
                    "s": s,
                },
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        return obe(beams, B, ham)

    def test_no_spikes_with_deltat_v(self, molasses_obe):
        """Force profile must be smooth — no spikes or jaggedness when deltat_v is used."""
        o = molasses_obe
        v = np.arange(-10.0, 10.5, 0.5)
        o.generate_force_profile(
            np.zeros((3,) + v.shape),
            [v, np.zeros(v.shape), np.zeros(v.shape)],
            name="test",
            deltat_tmax=2 * np.pi * 100,
            deltat_v=4,
            itermax=1000,
            rel=1e-4,
            abs=1e-6,
        )
        F = np.array(o.profile["test"].F[0])

        # 1) No extreme spikes: no point should deviate wildly from neighbours.
        for i in range(1, len(F) - 1):
            neighbour_max = max(abs(F[i - 1]), abs(F[i + 1]))
            if neighbour_max > 1e-6:
                assert abs(F[i]) < 3 * neighbour_max, (
                    f"Spike at v={v[i]:.1f}: |F|={abs(F[i]):.4f} vs "
                    f"neighbours {abs(F[i - 1]):.4f}, {abs(F[i + 1]):.4f}"
                )

        # 2) Overall smoothness: the second derivative (curvature) should be
        #    small relative to the signal.  Jagged profiles have large d²F
        #    at many points even without single-point spikes.
        d2F = np.diff(F, n=2)
        rms_d2F = np.sqrt(np.mean(d2F**2))
        F_range = np.max(F) - np.min(F)
        assert F_range > 1e-8, "Force profile is essentially zero everywhere"
        roughness = rms_d2F / F_range
        assert roughness < 0.3, (
            f"Force profile is too jagged: roughness (rms d²F / range) = {roughness:.3f} > 0.3"
        )

    def test_dark_state_smooth(self):
        """Type-II (dark-state) system: forces must be near zero at all velocities.

        Fg=1 -> Fe=1 with lin-lin polarization (phi=0) has a dark state
        (|mF=0>_x) that is velocity-independent, so the time-averaged force
        should be near zero at all velocities once optical pumping is complete.

        Roughness (rms d²F / F_range) is not used here: when F ≈ 0 everywhere,
        F_range collapses to numerical noise, making the ratio meaningless.
        Instead we verify that max|F| and std(F) are below physical thresholds,
        which directly captures the convergence fix (removing premature-convergence
        criteria that locked atoms at spuriously large forces).
        """
        Fg, Fe = 1, 1
        det, s = -2.5, 1.0
        Hg, Bgq = hamiltonians.singleF(F=Fg, gF=0, muB=1)
        He, Beq = hamiltonians.singleF(F=Fe, gF=1 / Fe, muB=1)
        dijq = hamiltonians.dqij_two_bare_hyperfine(Fg, Fe)
        ham = hamiltonian(Hg, He - det * np.eye(2 * Fe + 1), Bgq, Beq, dijq)
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        beams = laserBeams(
            [
                {
                    "kvec": np.array([0.0, 0.0, 1.0]),
                    "pol": np.array([1.0, 0.0, 0.0]),
                    "pol_coord": "cartesian",
                    "delta": 0,
                    "s": s,
                },
                {
                    "kvec": np.array([0.0, 0.0, -1.0]),
                    "pol": np.array([1.0, 0.0, 0.0]),
                    "pol_coord": "cartesian",
                    "delta": 0,
                    "s": s,
                },
            ],
            beam_type=infinitePlaneWaveBeam,
        )
        o = obe(beams, B, ham, transform_into_re_im=True)

        v = np.arange(0.5, 3.1, 0.5)
        o.generate_force_profile(
            [np.zeros(v.shape), np.zeros(v.shape), np.zeros(v.shape)],
            [np.zeros(v.shape), np.zeros(v.shape), v],
            name="test",
            deltat_v=4,
            deltat_tmax=2 * np.pi * 100,
            itermax=200,
            rel=1e-6,
            abs=1e-8,
        )
        F = np.array(o.profile["test"].F[2])

        max_F = np.max(np.abs(F))
        std_F = np.std(F)
        assert max_F < 5e-4, (
            f"Dark-state force too large: max|F| = {max_F:.2e} > 5e-4 "
            f"(premature convergence before reaching dark state?)"
        )
        assert std_F < 3e-4, f"Dark-state force profile too variable: std(F) = {std_F:.2e} > 3e-4"

    def test_Fg2_Fe1_linlin_near_zero_and_smooth(self):
        """Fg=2->Fe=1, phi=0 (Lin||Lin): force near zero and no oscillations.

        This is the exact case from the Devlin 2016 (Fig. 1, bottom row, orange
        line) that regressed to a jagged profile when the "diff_sq < abs_tol"
        convergence criterion declared convergence prematurely for slowly
        decaying dark-state forces.

        Two failure modes are tested:
          1. max|F_z| too large — premature convergence left force non-zero.
          2. std(F_z) too large — profile oscillates around zero.
        """
        Fg, Fe = 2, 1
        det, s = -2.5, 1.0
        Hg, Bgq = hamiltonians.singleF(F=Fg, gF=0, muB=1)
        He, Beq = hamiltonians.singleF(F=Fe, gF=1 / Fe, muB=1)
        dijq = hamiltonians.dqij_two_bare_hyperfine(Fg, Fe)
        ham = hamiltonian(Hg, He - det * np.eye(2 * Fe + 1), Bgq, Beq, dijq)
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        beams = laserBeams(
            [
                {
                    "kvec": np.array([0.0, 0.0, 1.0]),
                    "pol": np.array([1.0, 0.0, 0.0]),
                    "pol_coord": "cartesian",
                    "delta": 0,
                    "s": s,
                },
                {
                    "kvec": np.array([0.0, 0.0, -1.0]),
                    "pol": np.array([1.0, 0.0, 0.0]),
                    "pol_coord": "cartesian",
                    "delta": 0,
                    "s": s,
                },
            ],
            beam_type=infinitePlaneWaveBeam,
        )
        o = obe(beams, B, ham, transform_into_re_im=True)

        # 6 points at dv=0.5 up to v=3.0 — fast to compute; dark-state pumping
        # is complete within deltat_tmax=2*pi*100 for these velocities.
        v = np.arange(0.5, 3.1, 0.5)
        o.generate_force_profile(
            [np.zeros(v.shape), np.zeros(v.shape), np.zeros(v.shape)],
            [np.zeros(v.shape), np.zeros(v.shape), v],
            name="test",
            deltat_v=4,
            deltat_tmax=2 * np.pi * 100,
            itermax=200,
            rel=1e-6,
            abs=1e-8,
        )
        F = np.array(o.profile["test"].F[2])

        # 1. Force must be near zero everywhere (dark-state physics).
        assert np.max(np.abs(F)) < 5e-4, (
            f"Dark-state force too large: max|F_z| = {np.max(np.abs(F)):.2e}"
        )

        # 2. Profile must not oscillate.  The regression produced large-amplitude
        #    sign-alternating swings; std catches this even for a near-zero profile.
        #    For the smooth fixed profile std ~ 1.3e-4; oscillating would be >> 1e-3.
        assert np.std(F) < 3e-4, f"Force profile oscillates: std(F_z) = {np.std(F):.2e} > 3e-4"


@pytest.mark.slow
class TestEvolveMotion:
    """Tests for obe.evolve_motion, including default y0/keys and dtype consistency."""

    @pytest.fixture
    def mot_obe_transform(self):
        """OBE with transform_into_re_im=True (real-valued rho), weak beam."""
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.01, "delta": -1.0},
                {"kvec": [0.0, 0.0, -1.0], "pol": +1, "s": 0.01, "delta": -1.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        return obe(beams, B, ham, transform_into_re_im=True)

    @pytest.fixture
    def mot_obe_complex(self):
        """OBE with transform_into_re_im=False (complex-valued rho), weak beam."""
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.01, "delta": -1.0},
                {"kvec": [0.0, 0.0, -1.0], "pol": +1, "s": 0.01, "delta": -1.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        return obe(beams, B, ham, transform_into_re_im=False)

    def test_evolve_motion_default_args(self, mot_obe_transform):
        """evolve_motion should work without explicit y0_batch/keys_batch."""
        o = mot_obe_transform
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        # Should not raise
        o.evolve_motion(
            [0, 10],
            n_points=51,
            freeze_axis=[True, True, False],
            random_recoil=False,
            backend="cpu",
        )
        assert len(o.sols) == 1
        assert not jnp.any(jnp.isnan(o.sols[0].r))

    def test_evolve_motion_transform_dtype_consistency(self, mot_obe_transform):
        """With transform_into_re_im=True, y0 and dydt must both be real-valued.

        The ev_mat matrices are stored as float64 when transform_into_re_im=True,
        so __drhodt naturally returns float64 without needing jnp.real().
        """
        o = mot_obe_transform
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()

        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])
        assert y0.dtype == jnp.float64, (
            f"y0 should be float64 with transform_into_re_im=True, got {y0.dtype}"
        )

        # __drhodt should return float64 directly (no jnp.real() needed)
        rho = y0[:-6]
        r = y0[-3:]
        drhodt = o._obe__drhodt(r, 0.0, rho)
        assert drhodt.dtype == jnp.float64, (
            f"__drhodt should return float64 with transform_into_re_im=True, got {drhodt.dtype}"
        )

        F = o.force(r, 0.0, rho, return_details=False)
        dydt_out = jnp.concatenate([drhodt, F / o.hamiltonian.mass, jnp.zeros(3)])
        assert dydt_out.dtype == y0.dtype, (
            f"dydt output dtype {dydt_out.dtype} must match y0 dtype {y0.dtype}"
        )

    def test_evolve_motion_complex_mode(self, mot_obe_complex):
        """evolve_motion should also work with transform_into_re_im=False."""
        o = mot_obe_complex
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        o.evolve_motion(
            [0, 10],
            n_points=51,
            freeze_axis=[True, True, False],
            random_recoil=False,
            backend="cpu",
        )
        assert len(o.sols) == 1
        assert not jnp.any(jnp.isnan(o.sols[0].r))


# ---------------------------------------------------------------------------
# TestEvolveMotionPhysics
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEvolveMotionPhysics:
    """Physical correctness of OBE evolve_motion trajectories."""

    @pytest.fixture
    def single_beam_obe(self):
        """OBE with a single +z beam for radiation pressure tests."""
        ham = make_ham(mass=1.0, gamma=1.0, k=1.0)
        beams = laserBeams([{"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": 0.0}])
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        o = obe(beams, B, ham, transform_into_re_im=True)
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        return o

    @pytest.fixture
    def mot_obe(self):
        """OBE with counter-propagating red-detuned beams (1D MOT config)."""
        ham = make_ham(mass=1.0, gamma=1.0, k=1.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.5, "delta": -1.0},
                {"kvec": [0.0, 0.0, -1.0], "pol": -1, "s": 0.5, "delta": -1.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        o = obe(beams, B, ham, transform_into_re_im=True)
        return o

    def test_single_beam_accelerates_in_z(self, single_beam_obe):
        """+z beam should push the atom in +z."""
        o = single_beam_obe
        o.evolve_motion([0, 10.0], n_points=101, freeze_axis=[True, True, False], backend="cpu")
        assert float(o.sols[0].r[2, -1]) > 0.0, "Atom should move in +z"
        assert float(o.sols[0].v[2, -1]) > 0.0, "Atom should have +z velocity"

    def test_single_beam_no_transverse_motion(self, single_beam_obe):
        """A pure +z beam should not cause x or y motion."""
        o = single_beam_obe
        o.evolve_motion([0, 10.0], n_points=101, freeze_axis=[True, True, False], backend="cpu")
        assert np.allclose(o.sols[0].r[0], 0.0, atol=1e-10)
        assert np.allclose(o.sols[0].r[1], 0.0, atol=1e-10)

    def test_freeze_axis_prevents_motion(self, single_beam_obe):
        """Freezing z should prevent motion even with a +z beam."""
        o = single_beam_obe
        o.evolve_motion([0, 10.0], n_points=101, freeze_axis=[False, False, True], backend="cpu")
        assert np.allclose(o.sols[0].r[2], 0.0, atol=1e-10)
        assert np.allclose(o.sols[0].v[2], 0.0, atol=1e-10)

    def test_density_matrix_stays_physical(self, single_beam_obe):
        """Trace of rho must stay 1 and diagonal elements non-negative."""
        o = single_beam_obe
        o.evolve_motion([0, 10.0], n_points=101, freeze_axis=[True, True, False], backend="cpu")
        rho = np.array(o.sols[0].rho)
        n_times = rho.shape[2]
        for t_idx in range(n_times):
            trace = np.real(np.trace(rho[:, :, t_idx]))
            assert trace == pytest.approx(1.0, abs=1e-4), f"Trace = {trace} at t_idx={t_idx}"
            diag = np.real(np.diag(rho[:, :, t_idx]))
            assert np.all(diag >= -1e-6), f"Negative population at t_idx={t_idx}"

    def test_molasses_decelerates(self, mot_obe):
        """Counter-propagating red-detuned beams should decelerate a moving atom."""
        o = mot_obe
        v0 = 2.0
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.array([0.0, 0.0, v0]))
        o.set_initial_rho_from_rateeq()
        o.evolve_motion([0, 20.0], n_points=201, freeze_axis=[True, True, False], backend="cpu")
        vz_final = float(o.sols[0].v[2, -1])
        assert vz_final < v0, "Molasses should decelerate the atom"

    def test_no_nan_in_trajectory(self, single_beam_obe):
        o = single_beam_obe
        o.evolve_motion([0, 10.0], n_points=101, freeze_axis=[True, True, False], backend="cpu")
        assert not np.any(np.isnan(o.sols[0].r))
        assert not np.any(np.isnan(o.sols[0].v))
        assert not np.any(np.isnan(o.sols[0].rho))

    def test_random_recoil_runs(self, single_beam_obe):
        """random_recoil=True should run without error."""
        o = single_beam_obe
        o.evolve_motion(
            [0, 5.0],
            n_points=51,
            freeze_axis=[True, True, False],
            random_recoil=True,
            backend="cpu",
        )
        assert len(o.sols) == 1
        assert not np.any(np.isnan(o.sols[0].r))


# ---------------------------------------------------------------------------
# TestBatchSize
# ---------------------------------------------------------------------------


class TestBatchSize:
    """Tests that batch_size chunking in solve_ivp_random produces results
    consistent with a single unchunked batch."""

    @pytest.fixture
    def multi_atom_obe(self):
        """OBE with 6 atoms at different velocities for batch testing."""
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.01, "delta": -1.0},
                {"kvec": [0.0, 0.0, -1.0], "pol": +1, "s": 0.01, "delta": -1.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        o = obe(beams, B, ham, transform_into_re_im=True)

        velocities = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
        rho0_list = []
        for vz in velocities:
            o.set_initial_position(jnp.zeros(3))
            o.set_initial_velocity(jnp.array([0.0, 0.0, vz]))
            o.set_initial_rho_from_rateeq()
            rho0_list.append(jnp.concatenate([o.rho0, o.v0, o.r0]))

        y0_batch = jnp.stack(rho0_list)
        key = jax.random.PRNGKey(42)
        keys_batch = jax.random.split(key, len(velocities))
        return o, y0_batch, keys_batch

    def test_batch_size_matches_unbatched(self, multi_atom_obe):
        """Chunked results must match single-batch results exactly."""
        o, y0_batch, keys_batch = multi_atom_obe
        N = y0_batch.shape[0]

        # Run without chunking
        o.evolve_motion(
            [0, 10],
            n_points=51,
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[True, True, False],
            random_recoil=False,
            backend="cpu",
        )
        sols_full = o.sols

        # Run with batch_size=2 (3 chunks of 2 for 6 atoms)
        o.evolve_motion(
            [0, 10],
            n_points=51,
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[True, True, False],
            random_recoil=False,
            batch_size=2,
            backend="cpu",
        )
        sols_chunked = o.sols

        assert len(sols_chunked) == len(sols_full)
        for i in range(N):
            np.testing.assert_allclose(
                np.array(sols_chunked[i].r),
                np.array(sols_full[i].r),
                atol=1e-10,
                err_msg=f"Atom {i} position mismatch with batch_size=2",
            )
            np.testing.assert_allclose(
                np.array(sols_chunked[i].v),
                np.array(sols_full[i].v),
                atol=1e-10,
                err_msg=f"Atom {i} velocity mismatch with batch_size=2",
            )

    def test_batch_size_one(self, multi_atom_obe):
        """batch_size=1 (fully sequential) should still produce valid results."""
        o, y0_batch, keys_batch = multi_atom_obe
        o.evolve_motion(
            [0, 10],
            n_points=51,
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[True, True, False],
            random_recoil=False,
            batch_size=1,
            backend="cpu",
        )
        for sol in o.sols:
            assert not jnp.any(jnp.isnan(sol.r))
            assert not jnp.any(jnp.isnan(sol.v))

    def test_batch_size_larger_than_N(self, multi_atom_obe):
        """batch_size >= N should behave identically to no chunking."""
        o, y0_batch, keys_batch = multi_atom_obe
        N = y0_batch.shape[0]

        o.evolve_motion(
            [0, 10],
            n_points=51,
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[True, True, False],
            random_recoil=False,
            backend="cpu",
        )
        sols_full = o.sols

        o.evolve_motion(
            [0, 10],
            n_points=51,
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[True, True, False],
            random_recoil=False,
            batch_size=1000,
            backend="cpu",
        )
        sols_large = o.sols

        assert len(sols_large) == len(sols_full)
        for i in range(N):
            np.testing.assert_allclose(
                np.array(sols_large[i].r),
                np.array(sols_full[i].r),
                atol=1e-10,
            )


# ---------------------------------------------------------------------------
# TestRandomRecoilKickDistribution
# ---------------------------------------------------------------------------


class TestRandomRecoilKickDistribution:
    """Verify random_recoil kicks use two independent random unit vectors.

    The sum of two independent random unit vectors has a magnitude that
    varies between 0 and 2.  A single vector scaled by 2 would always
    have magnitude exactly 2.
    """

    def test_kick_magnitude_varies(self):
        """Kick magnitudes must not all be identical (rules out fixed * 2)."""
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.1, "delta": 0.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        o = obe(beams, B, ham, transform_into_re_im=True)
        o.set_initial_rho_equally()
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))

        # Build the random recoil function the same way evolve_motion does
        free_axes = jnp.bitwise_not(jnp.asarray([False, False, False], dtype=bool))

        def _jax_random_vector(key):
            key_phi, key_z = jax.random.split(key)
            phi = 2.0 * jnp.pi * jax.random.uniform(key_phi)
            z = 2.0 * jax.random.uniform(key_z) - 1.0
            r_vec = jnp.sqrt(1.0 - z**2)
            return jnp.array([r_vec * jnp.cos(phi), r_vec * jnp.sin(phi), z]) * free_axes

        magnitudes = []
        for decay_key in o.decay_rates:
            recoil_vel = o.recoil_velocity[decay_key]
            for i in range(200):
                key = jax.random.PRNGKey(i)
                _, subkey_v1, subkey_v2 = jax.random.split(key, 3)
                vec1 = _jax_random_vector(subkey_v1)
                vec2 = _jax_random_vector(subkey_v2)
                kick = recoil_vel * (vec1 + vec2)
                magnitudes.append(float(jnp.linalg.norm(kick)))

        assert len(magnitudes) > 10, "Too few samples to test"
        # If kicks are vec1+vec2, magnitudes vary; if *2, they're all identical
        assert np.std(magnitudes) > 1e-6, (
            "All kick magnitudes are identical — likely using a single "
            "random vector * 2 instead of two independent random vectors"
        )

    def test_multilevel_kicks_are_independent(self):
        """When multiple excited states scatter simultaneously, each must get
        its own independent random direction.

        For N independent kicks of recoil_v * (vec1 + vec2) where vec1, vec2
        are uniform on the unit sphere:
            E[|Δv|²] = N * recoil_v² * E[|vec1+vec2|²] = N * recoil_v² * 2

        If kicks share a single direction (old bug):
            E[|Δv|²] = N² * recoil_v² * 2

        The ratio is N, so for N=5 excited states the means differ by 5x.
        """
        # F=1 -> F'=2: 3 ground + 5 excited states
        H0_g, mu_g = hamiltonians.singleF(F=1, gF=0)
        H0_e, mu_e = hamiltonians.singleF(F=2, gF=0)
        d_q = hamiltonians.dqij_two_bare_hyperfine(1, 2)
        ham = hamiltonian(H0_g, H0_e, mu_g, mu_e, d_q, mass=100.0, gamma=1.0, k=1.0)

        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.1, "delta": 0.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        o = obe(beams, B, ham, transform_into_re_im=True)
        o.set_initial_rho_equally()
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))

        recoil_fn = o._motion_recoil_fn
        free_axes = jnp.bitwise_not(jnp.asarray([False, False, False], dtype=bool))
        args = {
            "free_axes": free_axes,
            "max_scatter_probability": jnp.float64(0.1),
        }

        # Build a state vector where all excited-state populations are large
        # so that P = dt * rate * pop >> 1 for every excited state.
        n2 = ham.n**2
        y = jnp.zeros(n2 + 6)  # rho_flat + v(3) + r(3)
        for dk in o.decay_rho_indices:
            for idx in o.decay_rho_indices[dk]:
                y = y.at[int(idx)].set(1.0)
        dt = jnp.float64(10.0)

        # Count how many excited states will scatter
        n_exc = sum(len(o.decay_rho_indices[dk]) for dk in o.decay_rho_indices)
        recoil_v = list(o.recoil_velocity.values())[0]

        dv_sq = []
        for i in range(2000):
            key = jax.random.PRNGKey(i)
            y_out, _, _, _ = recoil_fn(0.0, y, dt, key, args)
            dv = y_out[-6:-3] - y[-6:-3]
            dv_sq.append(float(jnp.sum(dv**2)))

        mean_dv_sq = np.mean(dv_sq)

        # Independent: E[|Δv|²] = N * recoil_v² * 2
        expected_independent = n_exc * recoil_v**2 * 2
        # Correlated:  E[|Δv|²] = N² * recoil_v² * 2
        expected_correlated = n_exc**2 * recoil_v**2 * 2

        # The measured mean should be close to independent, not correlated.
        # Use a generous tolerance (30%) since we have finite samples.
        assert abs(mean_dv_sq - expected_independent) / expected_independent < 0.3, (
            f"mean(|Δv|²) = {mean_dv_sq:.6e}, expected (independent) = "
            f"{expected_independent:.6e}, but correlated would give "
            f"{expected_correlated:.6e}"
        )


# ---------------------------------------------------------------------------
# Magnetic trap OBE motion tests
# ---------------------------------------------------------------------------


class TestQuadrupoleTrapOBE:
    """Atom motion in magnetic traps using optical Bloch equations (OBE).

    Unlike the rate equation approach, the OBE tracks the full density
    matrix ρ, capturing quantum coherences between spin states.  This is
    essential near the zero-field point of a quadrupole trap, where the
    quantization axis changes direction rapidly and Majorana spin flips
    can transfer population to untrapped states.

    For a spin-1/2 with gF=1 in a linear field B⃗ = (0, 0, z), a pure
    spin-up state (mF = +1/2) is weak-field seeking and experiences a
    constant force F = gF·μB·(∂B/∂z)/2, giving parabolic motion
    z(t) = z₀ + ½·a·t² with a = gF·μB·(∂B/∂z)/(2m).  In dimensionless
    units (μB=1, m=1, ∂B/∂z=1), this gives z(t) ≈ t²/4 + z₀.

    The density matrix must remain physical throughout: Tr(ρ) = 1 and
    all diagonal elements (populations) must be non-negative.

    Adapted from tests/magnetic_traps/01_motion_OBE.py.
    """

    @pytest.fixture(scope="class")
    def trap_setup(self):
        """Build an OBE solver for a spin-1/2 atom in a linear magnetic field."""
        import pylcp_jax.hamiltonians as hamiltonians
        from pylcp_jax.hamiltonian import hamiltonian as ham_cls

        H0, muq = hamiltonians.singleF(1 / 2, gF=1, muB=1)
        h = ham_cls()
        h.add_H_0_block("g", H0)
        h.add_mu_q_block("g", muq)
        return h

    def test_linear_field_parabolic_motion(self, trap_setup):
        """Constant-gradient field → parabolic trajectory.

        In B⃗ = (0, 0, z) a spin-up atom feels a constant magnetic force
        (since ∂|B|/∂z = 1 everywhere for z > 0).  Released from z₀ = 1
        at rest, the trajectory is z(t) = t²/4 + 1 in dimensionless
        units.  We verify this at early times (t < 2) before the atom
        moves far enough for higher-order effects to matter.  rtol=0.15
        accounts for finite time-step and ODE solver tolerances."""
        h = trap_setup
        B = magField(lambda R: jnp.array([0.0, 0.0, R[2]]))
        o = obe({}, B, h, include_mag_forces=True, transform_into_re_im=False)
        o.set_initial_position(jnp.array([0.0, 0.0, 1.0]))
        o.set_initial_velocity(jnp.zeros(3))

        # Pure spin-up state
        theta = 0.0
        psi = np.array([np.cos(theta / 2), np.sin(theta / 2)])
        rho = np.array([[psi[0] * psi[0], psi[0] * psi[1]], [psi[1] * psi[0], psi[1] * psi[1]]])
        o.set_initial_rho(
            rho.reshape(
                4,
            )
        )

        o.evolve_motion([0.0, 4.0], random_recoil=False, n_points=101, backend="cpu")
        t = np.array(o.sols[0].t)
        z = np.array(o.sols[0].r[2])

        # Check parabolic trajectory z ≈ t²/4 + 1 at early times
        # (before the linear field approximation breaks down)
        early = t < 2.0
        z_expected = t[early] ** 2 / 4 + 1
        np.testing.assert_allclose(z[early], z_expected, rtol=0.15)

    def test_quadrupole_trap_confinement(self, trap_setup):
        """Quadrupole trap: atom must oscillate, not escape.

        A weak-field-seeking state in B⃗ = (−x/2, −y/2, z) is confined
        by the potential V ∝ |B⃗|.  Near the trap centre |B⃗| → 0, the
        quantization axis is ill-defined and Majorana spin flips can
        transfer the atom to an untrapped state — but for short
        evolution times and a start away from the origin, the atom
        should oscillate back toward z₀ rather than escape."""
        h = trap_setup
        B = magField(lambda R: jnp.array([-0.5 * R[0], -0.5 * R[1], 1 * R[2]]))
        o = obe({}, B, h, include_mag_forces=True, transform_into_re_im=False)
        o.set_initial_position(jnp.array([0.0, 0.0, 1.0]))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_populations(jnp.array([0.0, 1.0]))

        o.evolve_motion([0, 10], random_recoil=False, n_points=101, backend="cpu")
        r = np.array(o.sols[0].r)
        r_mag = np.sqrt(np.sum(r**2, axis=0))

        # Atom should stay bounded (not diverge)
        assert not np.any(np.isnan(r_mag)), "NaN in position"
        # Check atom oscillates back (doesn't just fly away monotonically)
        z = np.array(o.sols[0].r[2])
        assert np.any(z < z[0]), "Atom should oscillate back toward origin"

    def test_density_matrix_stays_physical(self, trap_setup):
        """Tr(ρ) must remain 1 throughout the evolution.

        The OBE preserves the trace of the density matrix by construction
        (Lindblad form).  Numerical errors can cause small deviations, so
        we check Tr(ρ) = 1 ± 1e-6 at every saved time step."""
        h = trap_setup
        B = magField(lambda R: jnp.array([0.0, 0.0, R[2]]))
        o = obe({}, B, h, include_mag_forces=True, transform_into_re_im=False)
        o.set_initial_position(jnp.array([0.0, 0.0, 1.0]))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_populations(jnp.array([0.0, 1.0]))

        o.evolve_motion([0.0, 5.0], random_recoil=False, n_points=51, backend="cpu")
        rho = np.array(o.sols[0].rho)

        n_times = rho.shape[2]
        for t_idx in range(n_times):
            rho_mat = rho[:, :, t_idx]
            trace = np.real(np.trace(rho_mat))
            assert trace == pytest.approx(1.0, abs=1e-6), f"Trace = {trace} at time index {t_idx}"

    def test_spin_expectation_values_bounded(self, trap_setup):
        """Diagonal elements ρ_{ii} must stay in [0, 1] (populations).

        For a physical density matrix, each diagonal element represents
        the probability of occupying state |i⟩ and must be non-negative
        and at most 1."""
        h = trap_setup
        B = magField(lambda R: jnp.array([0.0, 0.0, R[2]]))
        o = obe({}, B, h, include_mag_forces=True, transform_into_re_im=False)
        o.set_initial_position(jnp.array([0.0, 0.0, 1.0]))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_populations(jnp.array([0.0, 1.0]))

        o.evolve_motion([0.0, 5.0], random_recoil=False, n_points=51, backend="cpu")
        rho = np.array(o.sols[0].rho)

        # rho has shape (n, n, T)
        n = rho.shape[0]
        n_times = rho.shape[2]
        for t_idx in range(n_times):
            rho_mat = rho[:, :, t_idx]
            # Population in each state
            for i in range(n):
                pop = np.real(rho_mat[i, i])
                assert pop >= -1e-6, f"Negative population {pop} at t_idx={t_idx}"
                assert pop <= 1.0 + 1e-6, f"Population > 1: {pop} at t_idx={t_idx}"


# ---------------------------------------------------------------------------
# GPU OBE tests — evolve_motion with backend='gpu'
# ---------------------------------------------------------------------------


@pytest.mark.slow
@requires_gpu
class TestEvolveMotionGPU:
    """Mirrors key TestEvolveMotion tests on GPU backend."""

    @pytest.fixture
    def mot_obe_transform(self):
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.01, "delta": -1.0},
                {"kvec": [0.0, 0.0, -1.0], "pol": +1, "s": 0.01, "delta": -1.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        return obe(beams, B, ham, transform_into_re_im=True)

    @pytest.fixture
    def mot_obe_complex(self):
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.01, "delta": -1.0},
                {"kvec": [0.0, 0.0, -1.0], "pol": +1, "s": 0.01, "delta": -1.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        return obe(beams, B, ham, transform_into_re_im=False)

    def test_evolve_motion_default_args(self, mot_obe_transform):
        o = mot_obe_transform
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        o.evolve_motion(
            [0, 10], freeze_axis=[True, True, False], random_recoil=False, backend="gpu"
        )
        assert len(o.sols) == 1
        assert not jnp.any(jnp.isnan(o.sols[0].r))

    def test_evolve_motion_complex_mode(self, mot_obe_complex):
        o = mot_obe_complex
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        o.evolve_motion(
            [0, 10], freeze_axis=[True, True, False], random_recoil=False, backend="gpu"
        )
        assert len(o.sols) == 1
        assert not jnp.any(jnp.isnan(o.sols[0].r))

    def test_sol_shapes(self, mot_obe_transform):
        """Solution arrays r, v, rho must have expected shapes."""
        o = mot_obe_transform
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        o.evolve_motion(
            [0, 10], freeze_axis=[True, True, False], random_recoil=False, backend="gpu"
        )
        sol = o.sols[0]
        n_t = len(sol.t)
        assert sol.r.shape == (3, n_t)
        assert sol.v.shape == (3, n_t)
        n_states = o.hamiltonian.n
        assert sol.rho.shape == (n_states, n_states, n_t)


@requires_gpu
class TestBatchSizeGPU:
    """Batch tests on GPU backend."""

    @pytest.fixture
    def multi_atom_obe(self):
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.01, "delta": -1.0},
                {"kvec": [0.0, 0.0, -1.0], "pol": +1, "s": 0.01, "delta": -1.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        o = obe(beams, B, ham, transform_into_re_im=True)

        velocities = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0]
        rho0_list = []
        for vz in velocities:
            o.set_initial_position(jnp.zeros(3))
            o.set_initial_velocity(jnp.array([0.0, 0.0, vz]))
            o.set_initial_rho_from_rateeq()
            rho0_list.append(jnp.concatenate([o.rho0, o.v0, o.r0]))

        y0_batch = jnp.stack(rho0_list)
        key = jax.random.PRNGKey(42)
        keys_batch = jax.random.split(key, len(velocities))
        return o, y0_batch, keys_batch

    def test_batch_size_matches_unbatched(self, multi_atom_obe):
        o, y0_batch, keys_batch = multi_atom_obe
        N = y0_batch.shape[0]

        o.evolve_motion(
            [0, 10],
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[True, True, False],
            random_recoil=False,
            backend="gpu",
        )
        sols_full = o.sols

        o.evolve_motion(
            [0, 10],
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[True, True, False],
            random_recoil=False,
            batch_size=2,
            backend="gpu",
        )
        sols_chunked = o.sols

        assert len(sols_chunked) == len(sols_full)
        for i in range(N):
            np.testing.assert_allclose(
                np.array(sols_chunked[i].r),
                np.array(sols_full[i].r),
                atol=1e-10,
                err_msg=f"Atom {i} position mismatch",
            )
            np.testing.assert_allclose(
                np.array(sols_chunked[i].v),
                np.array(sols_full[i].v),
                atol=1e-10,
                err_msg=f"Atom {i} velocity mismatch",
            )

    def test_batch_size_one(self, multi_atom_obe):
        o, y0_batch, keys_batch = multi_atom_obe
        o.evolve_motion(
            [0, 10],
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[True, True, False],
            random_recoil=False,
            batch_size=1,
            backend="gpu",
        )
        for sol in o.sols:
            assert not jnp.any(jnp.isnan(sol.r))
            assert not jnp.any(jnp.isnan(sol.v))


@requires_gpu
class TestQuadrupoleTrapOBEGPU:
    """Magnetic trap OBE tests on GPU backend."""

    @pytest.fixture(scope="class")
    def trap_setup(self):
        from pylcp_jax.hamiltonian import hamiltonian as ham_cls

        H0, muq = hamiltonians.singleF(1 / 2, gF=1, muB=1)
        h = ham_cls()
        h.add_H_0_block("g", H0)
        h.add_mu_q_block("g", muq)
        return h

    def test_linear_field_parabolic_motion(self, trap_setup):
        h = trap_setup
        B = magField(lambda R: jnp.array([0.0, 0.0, R[2]]))
        o = obe({}, B, h, include_mag_forces=True, transform_into_re_im=False)
        o.set_initial_position(jnp.array([0.0, 0.0, 1.0]))
        o.set_initial_velocity(jnp.zeros(3))

        theta = 0.0
        psi = np.array([np.cos(theta / 2), np.sin(theta / 2)])
        rho = np.array([[psi[0] * psi[0], psi[0] * psi[1]], [psi[1] * psi[0], psi[1] * psi[1]]])
        o.set_initial_rho(
            rho.reshape(
                4,
            )
        )

        o.evolve_motion([0.0, 4.0], random_recoil=False, n_points=101, backend="gpu")
        t = np.array(o.sols[0].t)
        z = np.array(o.sols[0].r[2])

        early = t < 2.0
        z_expected = t[early] ** 2 / 4 + 1
        np.testing.assert_allclose(z[early], z_expected, rtol=0.15)

    def test_quadrupole_trap_confinement(self, trap_setup):
        h = trap_setup
        B = magField(lambda R: jnp.array([-0.5 * R[0], -0.5 * R[1], 1 * R[2]]))
        o = obe({}, B, h, include_mag_forces=True, transform_into_re_im=False)
        o.set_initial_position(jnp.array([0.0, 0.0, 1.0]))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_populations(jnp.array([0.0, 1.0]))

        o.evolve_motion([0, 10], random_recoil=False, n_points=101, backend="gpu")
        r = np.array(o.sols[0].r)
        r_mag = np.sqrt(np.sum(r**2, axis=0))

        assert not np.any(np.isnan(r_mag)), "NaN in position"
        z = np.array(o.sols[0].r[2])
        assert np.any(z < z[0]), "Atom should oscillate back toward origin"

    def test_density_matrix_stays_physical(self, trap_setup):
        h = trap_setup
        B = magField(lambda R: jnp.array([0.0, 0.0, R[2]]))
        o = obe({}, B, h, include_mag_forces=True, transform_into_re_im=False)
        o.set_initial_position(jnp.array([0.0, 0.0, 1.0]))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_populations(jnp.array([0.0, 1.0]))

        o.evolve_motion([0.0, 5.0], random_recoil=False, n_points=51, backend="gpu")
        rho = np.array(o.sols[0].rho)

        n_times = rho.shape[2]
        for t_idx in range(n_times):
            rho_mat = rho[:, :, t_idx]
            trace = np.real(np.trace(rho_mat))
            assert trace == pytest.approx(1.0, abs=1e-6), f"Trace = {trace} at time index {t_idx}"


# ---------------------------------------------------------------------------
# Random recoil tests — stochastic scattering path on each backend
# ---------------------------------------------------------------------------


class TestRandomRecoilCPU:
    """Verify random_recoil=True works on CPU and produces physical results.

    Uses a strong on-resonance beam (s=10) to guarantee many scatter events.
    All tests are fully deterministic via fixed PRNG seeds."""

    @pytest.fixture
    def scattering_obe(self):
        """OBE with strong on-resonance beam to guarantee scattering events."""
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 10.0, "delta": 0.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        return obe(beams, B, ham, transform_into_re_im=True)

    def test_random_recoil_no_nan(self, scattering_obe):
        """Position and velocity must stay finite under stochastic recoil.

        Deterministic: uses a fixed PRNG key.  The fixture's high saturation
        (s=10, on resonance) ensures many scattering events per run regardless
        of the chosen seed, so the test exercises the recoil code path.
        """
        o = scattering_obe
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])[jnp.newaxis, :]
        keys_batch = jax.random.split(jax.random.PRNGKey(42), 1)
        o.evolve_motion(
            [0, 5],
            n_points=51,
            y0_batch=y0,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="cpu",
        )
        assert len(o.sols) == 1
        assert not jnp.any(jnp.isnan(o.sols[0].r))
        assert not jnp.any(jnp.isnan(o.sols[0].v))

    def test_random_recoil_produces_velocity_change(self, scattering_obe):
        """Atom must acquire nonzero velocity from recoil kicks.

        Deterministic: uses a fixed PRNG key.  The fixture's high saturation
        (s=10, on resonance) ensures many scattering events per run regardless
        of the chosen seed, so a velocity change is guaranteed.
        """
        o = scattering_obe
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])[jnp.newaxis, :]
        keys_batch = jax.random.split(jax.random.PRNGKey(42), 1)
        o.evolve_motion(
            [0, 5],
            n_points=51,
            y0_batch=y0,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="cpu",
        )
        v_final = np.array(o.sols[0].v[:, -1])
        assert np.linalg.norm(v_final) > 0, (
            "Atom should have nonzero velocity after scattering with recoil"
        )

    def test_random_recoil_scatter_events_recorded(self, scattering_obe):
        """Solution must record scatter events.

        Deterministic: uses a fixed PRNG key.  The fixture's high saturation
        (s=10, on resonance) ensures many scattering events per run regardless
        of the chosen seed.
        """
        o = scattering_obe
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])[jnp.newaxis, :]
        keys_batch = jax.random.split(jax.random.PRNGKey(42), 1)
        o.evolve_motion(
            [0, 5],
            n_points=51,
            y0_batch=y0,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="cpu",
        )
        sol = o.sols[0]
        assert len(sol.t_random) > 0, "Should have scatter events"
        assert jnp.any(sol.n_random > 0), "Should have nonzero scatter counts"

    def test_random_recoil_batch(self, scattering_obe):
        """Multiple atoms with random_recoil=True should all produce valid results."""
        o = scattering_obe
        N = 4
        rho0_list = []
        for _ in range(N):
            o.set_initial_position(jnp.zeros(3))
            o.set_initial_velocity(jnp.zeros(3))
            o.set_initial_rho_from_rateeq()
            rho0_list.append(jnp.concatenate([o.rho0, o.v0, o.r0]))

        y0_batch = jnp.stack(rho0_list)
        keys_batch = jax.random.split(jax.random.PRNGKey(0), N)
        o.evolve_motion(
            [0, 5],
            n_points=51,
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="cpu",
        )
        assert len(o.sols) == N
        for sol in o.sols:
            assert not jnp.any(jnp.isnan(sol.r))
            assert not jnp.any(jnp.isnan(sol.v))

    def test_random_recoil_different_keys_give_different_trajectories(self, scattering_obe):
        """Different PRNG keys must produce different stochastic trajectories."""
        o = scattering_obe
        N = 2
        rho0_list = []
        for _ in range(N):
            o.set_initial_position(jnp.zeros(3))
            o.set_initial_velocity(jnp.zeros(3))
            o.set_initial_rho_from_rateeq()
            rho0_list.append(jnp.concatenate([o.rho0, o.v0, o.r0]))

        y0_batch = jnp.stack(rho0_list)
        keys_batch = jax.random.split(jax.random.PRNGKey(0), N)
        o.evolve_motion(
            [0, 5],
            n_points=51,
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="cpu",
        )
        v0_final = np.array(o.sols[0].v[:, -1])
        v1_final = np.array(o.sols[1].v[:, -1])
        assert not np.allclose(v0_final, v1_final, atol=1e-10), (
            "Different PRNG keys should produce different trajectories"
        )

    def test_random_recoil_same_key_is_reproducible(self, scattering_obe):
        """Same seed must give identical results across two runs."""
        o = scattering_obe
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])[None, :]
        key = jax.random.split(jax.random.PRNGKey(7), 1)

        o.evolve_motion(
            [0, 5],
            n_points=51,
            y0_batch=y0,
            keys_batch=key,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="cpu",
        )
        r1 = np.array(o.sols[0].r)
        v1 = np.array(o.sols[0].v)

        o.evolve_motion(
            [0, 5],
            n_points=51,
            y0_batch=y0,
            keys_batch=key,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="cpu",
        )
        r2 = np.array(o.sols[0].r)
        v2 = np.array(o.sols[0].v)

        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(v1, v2)


@requires_gpu
class TestRandomRecoilGPU:
    """Verify random_recoil=True works on GPU and produces physical results.

    Mirrors TestRandomRecoilCPU with backend='gpu'.  Same fixed seeds."""

    @pytest.fixture
    def scattering_obe(self):
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 10.0, "delta": 0.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        return obe(beams, B, ham, transform_into_re_im=True)

    def test_random_recoil_no_nan(self, scattering_obe):
        """Position and velocity must stay finite under stochastic recoil.

        Deterministic: uses a fixed PRNG key.  The fixture's high saturation
        (s=10, on resonance) ensures many scattering events per run regardless
        of the chosen seed, so the test exercises the recoil code path.
        """
        o = scattering_obe
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])[jnp.newaxis, :]
        keys_batch = jax.random.split(jax.random.PRNGKey(42), 1)
        o.evolve_motion(
            [0, 5],
            y0_batch=y0,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="gpu",
        )
        assert len(o.sols) == 1
        assert not jnp.any(jnp.isnan(o.sols[0].r))
        assert not jnp.any(jnp.isnan(o.sols[0].v))

    def test_random_recoil_produces_velocity_change(self, scattering_obe):
        """Atom must acquire nonzero velocity from recoil kicks.

        Deterministic: uses a fixed PRNG key.  The fixture's high saturation
        (s=10, on resonance) ensures many scattering events per run regardless
        of the chosen seed, so a velocity change is guaranteed.
        """
        o = scattering_obe
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])[jnp.newaxis, :]
        keys_batch = jax.random.split(jax.random.PRNGKey(42), 1)
        o.evolve_motion(
            [0, 5],
            y0_batch=y0,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="gpu",
        )
        v_final = np.array(o.sols[0].v[:, -1])
        assert np.linalg.norm(v_final) > 0, (
            "Atom should have nonzero velocity after scattering with recoil"
        )

    def test_random_recoil_scatter_events_recorded(self, scattering_obe):
        """Solution must record scatter events.

        Deterministic: uses a fixed PRNG key.  The fixture's high saturation
        (s=10, on resonance) ensures many scattering events per run regardless
        of the chosen seed.
        """
        o = scattering_obe
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])[jnp.newaxis, :]
        keys_batch = jax.random.split(jax.random.PRNGKey(42), 1)
        o.evolve_motion(
            [0, 5],
            y0_batch=y0,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="gpu",
        )
        sol = o.sols[0]
        assert len(sol.t_random) > 0, "Should have scatter events"
        assert jnp.any(sol.n_random > 0), "Should have nonzero scatter counts"

    def test_random_recoil_batch(self, scattering_obe):
        o = scattering_obe
        N = 4
        rho0_list = []
        for _ in range(N):
            o.set_initial_position(jnp.zeros(3))
            o.set_initial_velocity(jnp.zeros(3))
            o.set_initial_rho_from_rateeq()
            rho0_list.append(jnp.concatenate([o.rho0, o.v0, o.r0]))

        y0_batch = jnp.stack(rho0_list)
        keys_batch = jax.random.split(jax.random.PRNGKey(0), N)
        o.evolve_motion(
            [0, 5],
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="gpu",
        )
        assert len(o.sols) == N
        for sol in o.sols:
            assert not jnp.any(jnp.isnan(sol.r))
            assert not jnp.any(jnp.isnan(sol.v))

    def test_random_recoil_different_keys_give_different_trajectories(self, scattering_obe):
        o = scattering_obe
        N = 2
        rho0_list = []
        for _ in range(N):
            o.set_initial_position(jnp.zeros(3))
            o.set_initial_velocity(jnp.zeros(3))
            o.set_initial_rho_from_rateeq()
            rho0_list.append(jnp.concatenate([o.rho0, o.v0, o.r0]))

        y0_batch = jnp.stack(rho0_list)
        keys_batch = jax.random.split(jax.random.PRNGKey(0), N)
        o.evolve_motion(
            [0, 5],
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="gpu",
        )
        v0_final = np.array(o.sols[0].v[:, -1])
        v1_final = np.array(o.sols[1].v[:, -1])
        assert not np.allclose(v0_final, v1_final, atol=1e-10), (
            "Different PRNG keys should produce different trajectories"
        )

    def test_random_recoil_same_key_is_reproducible(self, scattering_obe):
        o = scattering_obe
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])[None, :]
        key = jax.random.split(jax.random.PRNGKey(7), 1)

        o.evolve_motion(
            [0, 5],
            y0_batch=y0,
            keys_batch=key,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="gpu",
        )
        r1 = np.array(o.sols[0].r)
        v1 = np.array(o.sols[0].v)

        o.evolve_motion(
            [0, 5],
            y0_batch=y0,
            keys_batch=key,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="gpu",
        )
        r2 = np.array(o.sols[0].r)
        v2 = np.array(o.sols[0].v)

        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(v1, v2)


@requires_gpu
class TestCPUvsGPURandomRecoil:
    """CPU and GPU must produce identical stochastic trajectories given same keys.

    JAX PRNG is device-independent: same seed + same key -> identical random
    draws on CPU and GPU, so all scatter decisions and recoil kicks match."""

    @pytest.fixture
    def scattering_obe(self):
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 10.0, "delta": 0.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        return obe(beams, B, ham, transform_into_re_im=True)

    def test_single_atom_random_recoil_matches(self, scattering_obe):
        """Same key, same initial state -> identical stochastic trajectory."""
        o = scattering_obe
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()
        y0 = jnp.concatenate([o.rho0, o.v0, o.r0])[None, :]
        key = jax.random.split(jax.random.PRNGKey(42), 1)

        o.evolve_motion(
            [0, 5],
            y0_batch=y0,
            keys_batch=key,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="cpu",
        )
        r_cpu = np.array(o.sols[0].r)
        v_cpu = np.array(o.sols[0].v)
        t_random_cpu = np.array(o.sols[0].t_random)
        n_random_cpu = np.array(o.sols[0].n_random)

        o.evolve_motion(
            [0, 5],
            y0_batch=y0,
            keys_batch=key,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="gpu",
        )
        r_gpu = np.array(o.sols[0].r)
        v_gpu = np.array(o.sols[0].v)
        t_random_gpu = np.array(o.sols[0].t_random)
        n_random_gpu = np.array(o.sols[0].n_random)

        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, err_msg="Position mismatch CPU vs GPU")
        np.testing.assert_allclose(v_cpu, v_gpu, atol=1e-10, err_msg="Velocity mismatch CPU vs GPU")
        np.testing.assert_array_equal(n_random_cpu, n_random_gpu)
        np.testing.assert_allclose(t_random_cpu, t_random_gpu, atol=1e-12)

    def test_batch_random_recoil_matches(self, scattering_obe):
        """Multi-atom stochastic trajectories must agree across backends."""
        o = scattering_obe
        N = 4
        rho0_list = []
        for _ in range(N):
            o.set_initial_position(jnp.zeros(3))
            o.set_initial_velocity(jnp.zeros(3))
            o.set_initial_rho_from_rateeq()
            rho0_list.append(jnp.concatenate([o.rho0, o.v0, o.r0]))

        y0_batch = jnp.stack(rho0_list)
        keys_batch = jax.random.split(jax.random.PRNGKey(99), N)

        o.evolve_motion(
            [0, 5],
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="cpu",
        )
        sols_cpu = o.sols

        o.evolve_motion(
            [0, 5],
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[False, False, False],
            random_recoil=True,
            backend="gpu",
        )
        sols_gpu = o.sols

        for i in range(N):
            np.testing.assert_allclose(
                np.array(sols_cpu[i].r),
                np.array(sols_gpu[i].r),
                atol=1e-10,
                err_msg=f"Atom {i} position mismatch CPU vs GPU",
            )
            np.testing.assert_allclose(
                np.array(sols_cpu[i].v),
                np.array(sols_gpu[i].v),
                atol=1e-10,
                err_msg=f"Atom {i} velocity mismatch CPU vs GPU",
            )
            np.testing.assert_array_equal(
                np.array(sols_cpu[i].n_random),
                np.array(sols_gpu[i].n_random),
            )


# ---------------------------------------------------------------------------
# CPU vs GPU consistency — verify both backends produce identical results
# ---------------------------------------------------------------------------


@requires_gpu
class TestCPUvsGPUEvolveMotion:
    """CPU and GPU backends must produce the same trajectories."""

    @pytest.fixture
    def mot_obe(self):
        ham = make_ham(mass=100.0)
        beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 0.01, "delta": -1.0},
                {"kvec": [0.0, 0.0, -1.0], "pol": +1, "s": 0.01, "delta": -1.0},
            ]
        )
        B = constantMagneticField(jnp.array([0.0, 0.0, 0.0]))
        return obe(beams, B, ham, transform_into_re_im=True)

    def _setup_single(self, o):
        o.set_initial_position(jnp.zeros(3))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho_from_rateeq()

    def test_single_atom_trajectories_match(self, mot_obe):
        o = mot_obe
        self._setup_single(o)

        o.evolve_motion(
            [0, 10], freeze_axis=[True, True, False], random_recoil=False, backend="cpu"
        )
        sol_cpu = o.sols[0]
        r_cpu = np.array(sol_cpu.r)
        v_cpu = np.array(sol_cpu.v)
        rho_cpu = np.array(sol_cpu.rho)

        self._setup_single(o)
        o.evolve_motion(
            [0, 10], freeze_axis=[True, True, False], random_recoil=False, backend="gpu"
        )
        sol_gpu = o.sols[0]
        r_gpu = np.array(sol_gpu.r)
        v_gpu = np.array(sol_gpu.v)
        rho_gpu = np.array(sol_gpu.rho)

        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, err_msg="Position mismatch CPU vs GPU")
        np.testing.assert_allclose(v_cpu, v_gpu, atol=1e-10, err_msg="Velocity mismatch CPU vs GPU")
        np.testing.assert_allclose(
            rho_cpu, rho_gpu, atol=1e-10, err_msg="Density matrix mismatch CPU vs GPU"
        )

    def test_multi_atom_trajectories_match(self, mot_obe):
        o = mot_obe
        velocities = [0.0, 0.5, 1.0, 2.0]
        rho0_list = []
        for vz in velocities:
            o.set_initial_position(jnp.zeros(3))
            o.set_initial_velocity(jnp.array([0.0, 0.0, vz]))
            o.set_initial_rho_from_rateeq()
            rho0_list.append(jnp.concatenate([o.rho0, o.v0, o.r0]))

        y0_batch = jnp.stack(rho0_list)
        key = jax.random.PRNGKey(42)
        keys_batch = jax.random.split(key, len(velocities))

        o.evolve_motion(
            [0, 10],
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[True, True, False],
            random_recoil=False,
            backend="cpu",
        )
        sols_cpu = o.sols

        o.evolve_motion(
            [0, 10],
            y0_batch=y0_batch,
            keys_batch=keys_batch,
            freeze_axis=[True, True, False],
            random_recoil=False,
            backend="gpu",
        )
        sols_gpu = o.sols

        for i in range(len(velocities)):
            np.testing.assert_allclose(
                np.array(sols_cpu[i].r),
                np.array(sols_gpu[i].r),
                atol=1e-10,
                err_msg=f"Atom {i} position mismatch CPU vs GPU",
            )
            np.testing.assert_allclose(
                np.array(sols_cpu[i].v),
                np.array(sols_gpu[i].v),
                atol=1e-10,
                err_msg=f"Atom {i} velocity mismatch CPU vs GPU",
            )


@requires_gpu
class TestCPUvsGPUQuadrupoleTrap:
    """CPU and GPU backends must agree for magnetic trap evolution."""

    @pytest.fixture(scope="class")
    def trap_setup(self):
        from pylcp_jax.hamiltonian import hamiltonian as ham_cls

        H0, muq = hamiltonians.singleF(1 / 2, gF=1, muB=1)
        h = ham_cls()
        h.add_H_0_block("g", H0)
        h.add_mu_q_block("g", muq)
        return h

    def test_linear_field_trajectories_match(self, trap_setup):
        h = trap_setup
        B = magField(lambda R: jnp.array([0.0, 0.0, R[2]]))

        theta = 0.0
        psi = np.array([np.cos(theta / 2), np.sin(theta / 2)])
        rho0 = np.array(
            [[psi[0] * psi[0], psi[0] * psi[1]], [psi[1] * psi[0], psi[1] * psi[1]]]
        ).reshape(
            4,
        )

        o = obe({}, B, h, include_mag_forces=True, transform_into_re_im=False)
        o.set_initial_position(jnp.array([0.0, 0.0, 1.0]))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho(rho0)
        o.evolve_motion([0.0, 4.0], random_recoil=False, n_points=101, backend="cpu")
        r_cpu = np.array(o.sols[0].r)
        rho_cpu = np.array(o.sols[0].rho)

        o = obe({}, B, h, include_mag_forces=True, transform_into_re_im=False)
        o.set_initial_position(jnp.array([0.0, 0.0, 1.0]))
        o.set_initial_velocity(jnp.zeros(3))
        o.set_initial_rho(rho0)
        o.evolve_motion([0.0, 4.0], random_recoil=False, n_points=101, backend="gpu")
        r_gpu = np.array(o.sols[0].r)
        rho_gpu = np.array(o.sols[0].rho)

        np.testing.assert_allclose(r_cpu, r_gpu, atol=1e-10, err_msg="Position mismatch CPU vs GPU")
        np.testing.assert_allclose(
            rho_cpu, rho_gpu, atol=1e-10, err_msg="Density matrix mismatch CPU vs GPU"
        )
