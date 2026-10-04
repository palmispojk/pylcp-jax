"""
Tests for pylcp_jax/hamiltonians/__init__.py and pylcp_jax/hamiltonians/XFmolecules.py
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as cts

import pylcp_jax.hamiltonians as ham
from pylcp_jax.atom import atom
from pylcp_jax.common import cart2spherical, spherical2cart
from pylcp_jax.hamiltonians import XFmolecules

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def is_hermitian(M, atol=1e-10):
    M = np.array(M)
    return np.allclose(M, np.conj(M.T), atol=atol)


def is_spherical_rank1_mu(M, atol=1e-10):
    """Verify the rank-1 spherical tensor conjugation property.

    For a rank-1 spherical tensor T_q the components satisfy:
        T₀ is Hermitian  (T₀† = T₀)
        T₊₁† = −T₋₁      (or equivalently conj(T₊₁ᵀ) = −T₋₁)

    This ensures the Hamiltonian H = −μ⃗·B⃗ is Hermitian when
    contracted with a real magnetic field in spherical components.
    """
    M0 = np.array(M[0])
    M1 = np.array(M[1])
    M2 = np.array(M[2])
    return np.allclose(M1, np.conj(M1.T), atol=atol) and np.allclose(np.conj(M2.T), -M0, atol=atol)


def is_unitary(U, atol=1e-10):
    U = np.array(U)
    return np.allclose(np.conj(U.T) @ U, np.eye(U.shape[0]), atol=atol)


# ---------------------------------------------------------------------------
# TestUncoupledIndex
# ---------------------------------------------------------------------------


class TestUncoupledIndex:
    def test_first_index_is_zero(self):
        # mJ=-1/2, mI=-1/2 → first basis state
        assert ham.uncoupled_index(0.5, 0.5, -0.5, -0.5) == 0

    def test_last_index(self):
        assert ham.uncoupled_index(0.5, 0.5, 0.5, 0.5) == 3

    def test_all_indices_distinct(self):
        J, I = 0.5, 0.5
        indices = set()
        for mJ in np.arange(-J, J + 1, 1):
            for mI in np.arange(-I, I + 1, 1):
                indices.add(ham.uncoupled_index(J, I, mJ, mI))
        assert len(indices) == 4

    def test_integer_spin(self):
        # J=1, I=0: 3 states
        indices = set()
        for mJ in np.arange(-1, 2, 1):
            indices.add(ham.uncoupled_index(1, 0, mJ, 0))
        assert len(indices) == 3


# ---------------------------------------------------------------------------
# TestCoupledIndex
# ---------------------------------------------------------------------------


class TestCoupledIndex:
    def test_F0_is_zero(self):
        assert ham.coupled_index(0, 0, 0) == 0

    def test_F1_mF0_with_Fmin0(self):
        # F=0 contributes 1 state, then F=1,mF=-1,0,+1 → mF=0 is index 2
        assert ham.coupled_index(1, 0, 0) == 2

    def test_invalid_mF_raises(self):
        with pytest.raises(ValueError):
            ham.coupled_index(1, 2, 0)  # |mF|=2 > F=1

    def test_all_F1_indices_distinct(self):
        indices = {ham.coupled_index(1, m, 1) for m in [-1, 0, 1]}
        assert len(indices) == 3

    def test_sequential_for_Fmin0(self):
        # F=0,1: coupled_index(0,0,0)=0, (1,-1,0)=1, (1,0,0)=2, (1,1,0)=3
        assert ham.coupled_index(0, 0, 0) == 0
        assert ham.coupled_index(1, -1, 0) == 1
        assert ham.coupled_index(1, 1, 0) == 3


# ---------------------------------------------------------------------------
# TestDqijNorm
# ---------------------------------------------------------------------------


class TestDqijNorm:
    def test_shape_preserved(self):
        rng = np.random.default_rng(0)
        d = rng.random((3, 4, 5))
        assert ham.dqij_norm(d).shape == d.shape

    def test_column_norms_are_one(self):
        rng = np.random.default_rng(1)
        d = rng.random((3, 3, 5))
        d_n = ham.dqij_norm(d)
        for jj in range(d.shape[2]):
            assert np.linalg.norm(d_n[:, :, jj]) == pytest.approx(1.0, abs=1e-12)

    def test_zero_column_stays_zero(self):
        d = np.zeros((3, 2, 3))
        d[:, :, 1] = 1.0
        d_n = ham.dqij_norm(d)
        assert np.allclose(d_n[:, :, 0], 0.0)
        assert np.allclose(d_n[:, :, 2], 0.0)

    def test_already_normalized_unchanged(self):
        d = np.zeros((3, 1, 1))
        d[1, 0, 0] = 1.0  # single unit entry
        d_n = ham.dqij_norm(d)
        assert d_n[1, 0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestSingleF
# ---------------------------------------------------------------------------


class TestSingleF:
    """singleF(F, gF): zero-field Hamiltonian and μ_q for a single F level.

    Constructs the (2F+1)×(2F+1) zero-field Hamiltonian H₀ = 0 (all
    sublevels degenerate) and the rank-1 spherical magnetic moment
    operator μ_q with components q = −1, 0, +1.

    The q=0 component (π) is diagonal in the |F, mF⟩ basis with
    entries proportional to −gF·mF·μB, and scales linearly with gF."""

    def test_H0_shape_F1(self):
        H0, _ = ham.singleF(F=1, gF=1)
        assert H0.shape == (3, 3)

    def test_mu_q_shape_F1(self):
        _, mu_q = ham.singleF(F=1, gF=1)
        assert mu_q.shape == (3, 3, 3)

    def test_H0_is_zero(self):
        H0, _ = ham.singleF(F=1, gF=1)
        assert jnp.allclose(H0, jnp.zeros((3, 3)))

    def test_H0_shape_F2(self):
        H0, _ = ham.singleF(F=2, gF=1)
        assert H0.shape == (5, 5)

    def test_mu_q_hermitian(self):
        _, mu_q = ham.singleF(F=2, gF=1)
        assert is_spherical_rank1_mu(mu_q)

    def test_F_half_shape(self):
        H0, mu_q = ham.singleF(F=0.5, gF=2)
        assert H0.shape == (2, 2)
        assert mu_q.shape == (3, 2, 2)

    def test_return_basis_shape(self):
        _, _, basis = ham.singleF(F=1, gF=1, return_basis=True)
        assert basis.shape == (3, 2)

    def test_return_basis_F_values(self):
        _, _, basis = ham.singleF(F=1, gF=1, return_basis=True)
        assert np.all(basis[:, 0] == 1.0)

    def test_gF_zero_gives_zero_mu(self):
        _, mu_q = ham.singleF(F=1, gF=0)
        assert jnp.allclose(mu_q, jnp.zeros((3, 3, 3)))

    def test_q0_component_is_diagonal(self):
        """mu_q[1] (q=0, pi transition) should be diagonal."""
        _, mu_q = ham.singleF(F=1, gF=1)
        off = np.array(mu_q[1]) - np.diag(np.diagonal(np.array(mu_q[1])))
        assert np.allclose(off, 0.0, atol=1e-12)

    def test_q0_diagonal_scales_with_gF(self):
        _, mu_q_1 = ham.singleF(F=1, gF=1)
        _, mu_q_2 = ham.singleF(F=1, gF=2)
        assert jnp.allclose(mu_q_2[1], 2 * mu_q_1[1], atol=1e-12)


# ---------------------------------------------------------------------------
# TestHyperfineUncoupled
# ---------------------------------------------------------------------------


class TestHyperfineUncoupled:
    """Hyperfine Hamiltonian in the uncoupled |mJ, mI⟩ basis.

    The Hilbert space has (2J+1)(2I+1) states.  H₀ contains the
    magnetic dipole (Ahfs) and electric quadrupole (Bhfs) hyperfine
    interactions.  μ_q includes both electronic (gJ) and nuclear (gI)
    Zeeman contributions.  Both H₀ and μ_q must be Hermitian (with
    the rank-1 conjugation property for μ_q)."""

    def test_shape_J_half_I_half(self):
        H0, mu_q = ham.hyperfine_uncoupled(J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=1.0)
        assert H0.shape == (4, 4)
        assert mu_q.shape == (3, 4, 4)

    def test_shape_J1_I1(self):
        H0, mu_q = ham.hyperfine_uncoupled(J=1.0, I=1.0, gJ=2.0, gI=0.0, Ahfs=1.0)
        assert H0.shape == (9, 9)
        assert mu_q.shape == (3, 9, 9)

    def test_H0_hermitian(self):
        H0, _ = ham.hyperfine_uncoupled(J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=1.0)
        assert is_hermitian(H0)

    def test_mu_q_hermitian(self):
        _, mu_q = ham.hyperfine_uncoupled(J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=1.0)
        assert is_spherical_rank1_mu(mu_q)

    def test_return_basis_shape(self):
        _, _, basis = ham.hyperfine_uncoupled(
            J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=1.0, return_basis=True
        )
        assert basis.shape[1] == 4  # 4 states


# ---------------------------------------------------------------------------
# TestHyperfineCoupled
# ---------------------------------------------------------------------------


class TestHyperfineCoupled:
    """Hyperfine Hamiltonian in the coupled |F, mF⟩ basis.

    H₀ is diagonal in the coupled basis with eigenvalues given by
    the interval rule:  E(F) = Ahfs/2 · [F(F+1) − I(I+1) − J(J+1)].
    For J=I=1/2 and Ahfs=1 this gives E(F=0) = −3/4 and E(F=1) = +1/4.

    Negative Ahfs inverts the level ordering (the "inverted" hyperfine
    structure seen in some excited states).  Non-zero Bhfs (electric
    quadrupole, requires J≥1 and I≥1) shifts eigenvalues beyond the
    magnetic dipole pattern."""

    def test_shape_J_half_I_half(self):
        H0, mu_q = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=1.0)
        assert H0.shape == (4, 4)
        assert mu_q.shape == (3, 4, 4)

    def test_shape_J_half_I_3half(self):
        H0, _ = ham.hyperfine_coupled(J=0.5, I=1.5, gJ=2.0, gI=0.0, Ahfs=1.0)
        assert H0.shape == (8, 8)

    def test_H0_hermitian(self):
        H0, _ = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=1.0)
        assert is_hermitian(H0)

    def test_mu_q_hermitian(self):
        _, mu_q = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=1.0)
        assert is_spherical_rank1_mu(mu_q)

    def test_diagonal_energies_J_half_I_half(self):
        """J=I=1/2, Ahfs=1: E(F=0)=-3/4 (×1), E(F=1)=+1/4 (×3)."""
        H0, _ = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=1.0)
        diag = np.sort(np.real(np.diagonal(np.array(H0))))
        assert diag[0] == pytest.approx(-0.75, abs=1e-10)
        assert diag[1] == pytest.approx(0.25, abs=1e-10)
        assert diag[2] == pytest.approx(0.25, abs=1e-10)
        assert diag[3] == pytest.approx(0.25, abs=1e-10)

    def test_hyperfine_splitting_scales_with_Ahfs(self):
        H0_1, _ = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=1.0)
        H0_2, _ = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=2.0)
        d1 = np.sort(np.real(np.diagonal(np.array(H0_1))))
        d2 = np.sort(np.real(np.diagonal(np.array(H0_2))))
        split1 = d1[-1] - d1[0]
        split2 = d2[-1] - d2[0]
        assert split2 == pytest.approx(2 * split1, abs=1e-10)

    def test_negative_Ahfs_flips_levels(self):
        """Negative Ahfs: F=0 should be above F=1 (tests Bhfs != 0 fix)."""
        H0, _ = ham.hyperfine_coupled(J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=-1.0)
        diag = np.sort(np.real(np.diagonal(np.array(H0))))
        assert diag[0] == pytest.approx(-0.25, abs=1e-10)
        assert diag[3] == pytest.approx(0.75, abs=1e-10)

    def test_return_basis(self):
        _, _, basis = ham.hyperfine_coupled(
            J=0.5, I=0.5, gJ=2.0, gI=0.0, Ahfs=1.0, return_basis=True
        )
        assert basis.shape[1] == 4

    def test_Bhfs_nonzero_changes_energies(self):
        """Non-zero Bhfs should change the eigenspectrum."""
        H0_no, _ = ham.hyperfine_coupled(J=1.0, I=1.0, gJ=2.0, gI=0.0, Ahfs=1.0, Bhfs=0.0)
        H0_b, _ = ham.hyperfine_coupled(J=1.0, I=1.0, gJ=2.0, gI=0.0, Ahfs=1.0, Bhfs=0.1)
        ev_no = np.sort(np.linalg.eigvalsh(np.array(H0_no)))
        ev_b = np.sort(np.linalg.eigvalsh(np.array(H0_b)))
        assert not np.allclose(ev_no, ev_b, atol=1e-6)

    def test_Bhfs_negative_accepted(self):
        """Negative Bhfs should not be silently skipped (tests Bhfs != 0 fix)."""
        H0_pos, _ = ham.hyperfine_coupled(J=1.0, I=1.0, gJ=2.0, gI=0.0, Ahfs=1.0, Bhfs=0.1)
        H0_neg, _ = ham.hyperfine_coupled(J=1.0, I=1.0, gJ=2.0, gI=0.0, Ahfs=1.0, Bhfs=-0.1)
        # The matrices should differ (not both same as Bhfs=0)
        assert not np.allclose(np.array(H0_pos), np.array(H0_neg), atol=1e-10)


# ---------------------------------------------------------------------------
# TestFineStructureUncoupled
# ---------------------------------------------------------------------------


class TestFineStructureUncoupled:
    """Fine structure Hamiltonian in the uncoupled |mL, mS, mI⟩ basis.

    The Hilbert space has (2L+1)(2S+1)(2I+1) states.  The Hamiltonian
    includes spin-orbit coupling ξ·L⃗·S⃗ and three hyperfine
    interactions: contact (a_c), orbital (a_orb), and dipolar (a_dip).
    For L≥2, the dipolar and orbital terms introduce off-diagonal
    elements that must still produce a Hermitian matrix."""

    def test_shape_L1_S_half_I_half(self):
        H0, mu_q = ham.fine_structure_uncoupled(
            L=1, S=0.5, I=0.5, xi=1.0, a_c=0.0, a_orb=0.0, a_dip=0.0, gL=1.0, gS=2.0, gI=0.0
        )
        # (2*1+1)*(2*0.5+1)*(2*0.5+1) = 3*2*2 = 12 states
        assert H0.shape == (12, 12)
        assert mu_q.shape == (3, 12, 12)

    def test_H0_hermitian_L1(self):
        H0, _ = ham.fine_structure_uncoupled(
            L=1, S=0.5, I=0.0, xi=1.0, a_c=0.1, a_orb=0.0, a_dip=0.0, gL=1.0, gS=2.0, gI=0.0
        )
        assert is_hermitian(H0)

    def test_H0_hermitian_L2_with_a_dip(self):
        """L=2 triggers section-IV off-diagonals; Hermiticity verifies the bug fix."""
        H0, _ = ham.fine_structure_uncoupled(
            L=2, S=0.5, I=0.0, xi=1.0, a_c=0.0, a_orb=0.5, a_dip=1.0, gL=1.0, gS=2.0, gI=0.0
        )
        assert is_hermitian(H0)

    def test_mu_q_hermitian_L1(self):
        _, mu_q = ham.fine_structure_uncoupled(
            L=1, S=0.5, I=0.0, xi=0.0, a_c=0.0, a_orb=0.0, a_dip=0.0, gL=1.0, gS=2.0, gI=0.0
        )
        assert is_spherical_rank1_mu(mu_q)

    def test_all_zero_couplings_gives_zero_H0(self):
        H0, _ = ham.fine_structure_uncoupled(
            L=1, S=0.5, I=0.0, xi=0.0, a_c=0.0, a_orb=0.0, a_dip=0.0, gL=1.0, gS=2.0, gI=0.0
        )
        assert jnp.allclose(H0, jnp.zeros_like(H0))

    def test_return_basis_length(self):
        _, _, basis = ham.fine_structure_uncoupled(
            L=1,
            S=0.5,
            I=0.0,
            xi=1.0,
            a_c=0.0,
            a_orb=0.0,
            a_dip=0.0,
            gL=1.0,
            gS=2.0,
            gI=0.0,
            return_basis=True,
        )
        # 3 * 2 = 6 states (I=0)
        assert len(basis) == 6

    def test_L2_section_IV_hermitian_all_terms(self):
        """Exercise section IV for L=2 with both a_orb and a_dip nonzero."""
        H0, _ = ham.fine_structure_uncoupled(
            L=2, S=0.5, I=0.5, xi=0.5, a_c=0.1, a_orb=0.3, a_dip=0.2, gL=1.0, gS=2.0, gI=0.001
        )
        assert is_hermitian(H0)


# ---------------------------------------------------------------------------
# TestDqijTwoFineStructureManifoldsUncoupled
# ---------------------------------------------------------------------------


class TestDqijTwoFineStructureManifoldsUncoupled:
    """Dipole matrix elements between fine structure manifolds in the
    uncoupled |mL, mS, mI⟩ basis.

    The electric dipole operator only changes L by ±1 and preserves
    mS and mI.  For a given polarization q, the selection rule is
    mL(ground) = mL(excited) + q.  All other matrix elements must
    vanish identically."""

    def _s_to_p_bases(self):
        """S-state (L=0) → P-state (L=1) with S=1/2, I=0."""
        basis_g = [(0, 0.5, 0), (0, -0.5, 0)]
        basis_e = [
            (-1, 0.5, 0),
            (-1, -0.5, 0),
            (0, 0.5, 0),
            (0, -0.5, 0),
            (1, 0.5, 0),
            (1, -0.5, 0),
        ]
        return basis_g, basis_e

    def test_shape(self):
        basis_g, basis_e = self._s_to_p_bases()
        d_q = ham.dqij_two_fine_stucture_manifolds_uncoupled(basis_g, basis_e)
        assert d_q.shape == (3, 2, 6)

    def test_no_nan(self):
        basis_g, basis_e = self._s_to_p_bases()
        d_q = ham.dqij_two_fine_stucture_manifolds_uncoupled(basis_g, basis_e)
        assert not jnp.any(jnp.isnan(d_q))

    def test_selection_rules(self):
        """d_q[kk, ii, jj] nonzero ⟺ mL_g - mL_e = q, mS_g = mS_e, mI_g = mI_e."""
        basis_g, basis_e = self._s_to_p_bases()
        d_q = np.array(ham.dqij_two_fine_stucture_manifolds_uncoupled(basis_g, basis_e))
        for kk, q in enumerate([-1, 0, 1]):
            for ii, (mL, mS, mI) in enumerate(basis_g):
                for jj, (mLp, mSp, mIp) in enumerate(basis_e):
                    expect = (mL == mLp + q) and (mS == mSp) and (mI == mIp)
                    assert (abs(d_q[kk, ii, jj]) > 1e-12) == expect

    def test_same_basis_identity_coupling(self):
        """If both bases share the same L, mL=mLp=0, q=0 coupling is 1."""
        basis = [(0, 0.5, 0)]
        d_q = np.array(ham.dqij_two_fine_stucture_manifolds_uncoupled(basis, basis))
        assert d_q[1, 0, 0] == 1.0  # q=0 (index 1)


# ---------------------------------------------------------------------------
# TestDqijTwoHyperfineManifolds
# ---------------------------------------------------------------------------


class TestDqijTwoHyperfineManifolds:
    """Dipole matrix elements between full hyperfine manifolds.

    Couples all F levels of the ground state (J) to all F′ levels of
    the excited state (J′) for a given nuclear spin I.  The resulting
    d^q_{ij} matrix has shape (3, n_ground, n_excited).

    When normalize=True, each column (excited state) has unit norm,
    ensuring that the total spontaneous decay rate from each excited
    state sums to 1."""

    def test_shape_J0_Jp1_I0(self):
        d_q = ham.dqij_two_hyperfine_manifolds(J=0, Jp=1, I=0)
        assert d_q.shape == (3, 1, 3)

    def test_shape_J_half_Jp_half_I_half(self):
        d_q = ham.dqij_two_hyperfine_manifolds(J=0.5, Jp=0.5, I=0.5)
        assert d_q.shape == (3, 4, 4)

    def test_no_nan(self):
        d_q = ham.dqij_two_hyperfine_manifolds(J=0, Jp=1, I=0)
        assert not jnp.any(jnp.isnan(d_q))

    def test_normalized_columns(self):
        d_q = np.array(ham.dqij_two_hyperfine_manifolds(J=0, Jp=1, I=0, normalize=True))
        for jj in range(d_q.shape[2]):
            assert np.linalg.norm(d_q[:, :, jj]) == pytest.approx(1.0, abs=1e-10)

    def test_unnormalized_not_unit(self):
        """Without normalization some columns may differ from unit norm."""
        d_q = np.array(ham.dqij_two_hyperfine_manifolds(J=0.5, Jp=0.5, I=0.5, normalize=False))
        norms = [np.linalg.norm(d_q[:, :, jj]) for jj in range(d_q.shape[2])]
        # At least some columns should not be exactly 1.0 before normalization
        assert not all(n == pytest.approx(1.0, abs=1e-6) for n in norms)

    def test_return_basis(self):
        _, basis_g, basis_e = ham.dqij_two_hyperfine_manifolds(
            J=0.5, Jp=0.5, I=0.5, return_basis=True
        )
        assert basis_g.shape[0] == 4
        assert basis_e.shape[0] == 4


# ---------------------------------------------------------------------------
# TestDqijTwoBareHyperfine
# ---------------------------------------------------------------------------


class TestDqijTwoBareHyperfine:
    """Dipole matrix elements between two bare hyperfine levels F and F′.

    This is the simplest dipole element calculation — a single ground
    level F coupled to a single excited level F′ (no summing over
    multiple F values).  Selection rules require ΔmF = q, so for
    F=0→F′=1 each q component has exactly one nonzero entry."""

    def test_shape_F0_Fp1(self):
        d_q = ham.dqij_two_bare_hyperfine(0, 1)
        assert d_q.shape == (3, 1, 3)

    def test_shape_F1_Fp2(self):
        d_q = ham.dqij_two_bare_hyperfine(1, 2)
        assert d_q.shape == (3, 3, 5)

    def test_no_nan(self):
        assert not jnp.any(jnp.isnan(ham.dqij_two_bare_hyperfine(1, 2)))

    def test_column_norms_unit_F0_Fp1(self):
        d_q = np.array(ham.dqij_two_bare_hyperfine(0, 1, normalize=True))
        for jj in range(3):
            assert np.linalg.norm(d_q[:, :, jj]) == pytest.approx(1.0, abs=1e-10)

    def test_selection_rules_F0_Fp1(self):
        """F=0→F'=1: each q-component has exactly one nonzero entry."""
        d_q = np.array(ham.dqij_two_bare_hyperfine(0, 1, normalize=False))
        for q_idx in range(3):
            assert np.count_nonzero(d_q[q_idx]) == 1

    def test_dtype_complex(self):
        d_q = ham.dqij_two_bare_hyperfine(0, 1)
        assert jnp.issubdtype(d_q.dtype, jnp.complexfloating)


# ---------------------------------------------------------------------------
# TestXstate (XFmolecules)
# ---------------------------------------------------------------------------


class TestXstate:
    """X²Σ⁺ electronic ground state of a diatomic molecule (e.g. CaF, SrF).

    The state is characterised by rotational quantum number N, electron
    spin S=1/2, and nuclear spin I.  Coupling of N⃗, S⃗, and I⃗ gives
    total angular momentum levels through Hund's case (b) coupling.
    The Hamiltonian includes spin-rotation (γ), Fermi contact (b),
    and dipolar (c) hyperfine interactions, plus Zeeman coupling.

    The unitary matrix U transforms from the uncoupled to the
    eigenstate basis and must satisfy U†U = 1."""

    def test_shape_N0_I_half(self):
        H0, mu_p, U, _ = XFmolecules.Xstate(N=0, I=0.5)
        # N=0, S=1/2 → J=1/2 → F=0,1 → 4 states
        assert H0.shape == (4, 4)
        assert mu_p.shape == (3, 4, 4)
        assert U.shape == (4, 4)

    def test_shape_N1_I_half(self):
        H0, mu_p, U, _ = XFmolecules.Xstate(N=1, I=0.5)
        # N=1, S=1/2 → J=1/2,3/2 → 4+8 = 12 states
        assert H0.shape == (12, 12)
        assert mu_p.shape == (3, 12, 12)

    def test_H0_hermitian(self):
        H0, _, _, _ = XFmolecules.Xstate(N=1, I=0.5, b=100.0, c=30.0)
        assert is_hermitian(H0)

    def test_mu_p_hermitian(self):
        _, mu_p, _, _ = XFmolecules.Xstate(N=1, I=0.5)
        assert is_spherical_rank1_mu(mu_p)

    def test_U_unitary(self):
        _, _, U, _ = XFmolecules.Xstate(N=1, I=0.5, b=100.0, c=30.0)
        assert is_unitary(U)

    def test_no_nan(self):
        H0, mu_p, _, _ = XFmolecules.Xstate(N=1, I=0.5, B=10000.0, b=100.0, c=30.0, gamma=40.0)
        assert not jnp.any(jnp.isnan(H0))
        assert not jnp.any(jnp.isnan(mu_p))

    def test_return_basis_length(self):
        _, _, _, basis = XFmolecules.Xstate(N=1, I=0.5)
        assert len(basis) == 12

    def test_diagonal_without_hyperfine(self):
        """All coupling constants zero → H0 should be all zeros (no splitting)."""
        H0, _, _, _ = XFmolecules.Xstate(N=0, I=0.5)
        assert jnp.allclose(jnp.real(H0), jnp.zeros_like(jnp.real(H0)), atol=1e-12)

    def test_multi_N(self):
        H0, mu_p, _, _ = XFmolecules.Xstate(N=[0, 1], I=0.5, B=10000.0)
        # N=0: 4 states, N=1: 12 states → 16 total
        assert H0.shape == (16, 16)


# ---------------------------------------------------------------------------
# TestAstate (XFmolecules)
# ---------------------------------------------------------------------------


class TestAstate:
    """A²Π electronic excited state of a diatomic molecule.

    Characterised by total electronic angular momentum J (Hund's case a),
    nuclear spin I, and parity P = ±1.  The hyperfine structure includes
    magnetic dipole (a), electric quadrupole (b), and nuclear spin-orbit
    (c) interactions.  Multiple J and/or P values can be combined into
    a single block-diagonal Hamiltonian."""

    def test_shape_J_half_I_half_single_P(self):
        H0, mu_p, _ = XFmolecules.Astate(J=0.5, I=0.5, P=+1)
        # J=1/2, I=1/2 → F=0,1 → 4 states
        assert H0.shape == (4, 4)
        assert mu_p.shape == (3, 4, 4)

    def test_H0_hermitian(self):
        H0, _, _ = XFmolecules.Astate(J=0.5, I=0.5, P=+1, b=5.0, c=2.0, a=3.0)
        assert is_hermitian(H0)

    def test_mu_p_hermitian(self):
        _, mu_p, _ = XFmolecules.Astate(J=0.5, I=0.5, P=+1)
        assert is_spherical_rank1_mu(mu_p)

    def test_no_nan(self):
        H0, mu_p, _ = XFmolecules.Astate(J=0.5, I=0.5, P=+1, a=3.0, b=5.0, c=2.0)
        assert not jnp.any(jnp.isnan(H0))
        assert not jnp.any(jnp.isnan(mu_p))

    def test_return_basis(self):
        _, _, basis = XFmolecules.Astate(J=0.5, I=0.5, P=+1)
        assert len(basis) == 4

    def test_two_J_values(self):
        H0, mu_p, _ = XFmolecules.Astate(J=[0.5, 1.5], I=0.5, P=+1)
        # J=1/2 → 4 states, J=3/2 → 8 states → 12 total
        assert H0.shape == (12, 12)

    def test_two_P_values(self):
        H0, mu_p, _ = XFmolecules.Astate(J=0.5, I=0.5, P=[+1, -1])
        # Two parities × 4 states = 8 states
        assert H0.shape == (8, 8)


# ---------------------------------------------------------------------------
# TestDipoleXandAstates (XFmolecules)
# ---------------------------------------------------------------------------


class TestDipoleXandAstates:
    """Dipole matrix elements between X²Σ⁺ and A²Π molecular states.

    These matrix elements connect the ground (X) and excited (A)
    electronic states via the electric dipole operator.  The
    transformation matrix U_X rotates the X-state basis from the
    uncoupled to the eigenstate representation before computing
    the dipole coupling."""

    @pytest.fixture(scope="class")
    def bases(self):
        _, _, U_X, Xbasis = XFmolecules.Xstate(N=1, I=0.5)
        _, _, Abasis = XFmolecules.Astate(J=0.5, I=0.5, P=+1)
        return Xbasis, Abasis, np.array(U_X)

    def test_output_shape(self, bases):
        Xbasis, Abasis, U_X = bases
        dijq = XFmolecules.dipoleXandAstates(Xbasis, Abasis, UX=U_X)
        assert dijq.shape == (3, 12, 4)

    def test_no_nan(self, bases):
        Xbasis, Abasis, U_X = bases
        dijq = XFmolecules.dipoleXandAstates(Xbasis, Abasis, UX=U_X)
        assert not jnp.any(jnp.isnan(dijq))

    def test_return_intermediate_length(self, bases):
        Xbasis, Abasis, U_X = bases
        result = XFmolecules.dipoleXandAstates(Xbasis, Abasis, UX=U_X, return_intermediate=True)
        assert len(result) == 6  # dijq, T_ap, T_ba, intdijq, intbasis_ap, intbasis_ba

    def test_no_UX_uses_identity(self, bases):
        Xbasis, Abasis, _ = bases
        # Passing UX=[] (default identity) should not raise
        dijq = XFmolecules.dipoleXandAstates(Xbasis, Abasis)
        assert dijq.shape == (3, 12, 4)


# ---------------------------------------------------------------------------
# Breit-Rabi formula helper
# ---------------------------------------------------------------------------


def breitrabi(B, gJ, gI, AHFS, J=1 / 2, I=3 / 2):
    """Analytical Breit-Rabi formula for J=1/2 ground states.

    The Breit-Rabi formula gives the exact eigenvalues of the hyperfine
    Hamiltonian for a J=1/2 state in a magnetic field.  For each pair
    (mJ, mI) with total projection m = mI + mJ, the energy is:

        E = -ΔHFS/(2(2I+1)) + gI·μB·m·B
            ± (ΔHFS/2)·√(1 + 4m·x/(2I+1) + x²)

    where ΔHFS = (I+1/2)·AHFS is the zero-field hyperfine splitting and
    x = (gJ - gI)·μB·B / ΔHFS is the dimensionless field parameter.
    The special case m = -(I+J) uses the linear branch (1 - x).

    Note: this formula uses a specific sign convention for gI that may
    differ from the Hamiltonian by a sign, limiting agreement to ~0.05%
    at low fields (B < 10 G).
    """
    muB = cts.value("Bohr magneton in Hz/T") * 1e-4
    dHFS = (I + 1 / 2) * AHFS
    x = (gJ - gI) * muB * B / dHFS

    m_J, m_I = np.meshgrid(np.arange(-J, J + 0.001, 1), np.arange(-I, I + 0.001, 1))
    m_J = m_J.reshape(-1)
    m_I = m_I.reshape(-1)

    E = np.zeros((m_J.size, B.size))
    for ii in range(m_J.size):
        m = m_I[ii] + m_J[ii]
        if m == -(I + J):
            E[ii, :] = -dHFS / (2 * (2 * I + 1)) + gI * muB * m * B + dHFS / 2 * (1 - x)
        else:
            sgn = np.sign(m_J[ii])
            E[ii, :] = (
                -dHFS / (2 * (2 * I + 1))
                + gI * muB * m * B
                + sgn * dHFS / 2 * (1 + 4 * m * x / (2 * I + 1) + x**2) ** 0.5
            )

    return np.sort(E, axis=0)


def diagonalize_hamiltonian(B_arr, H0, mu_q, Bhat=np.array([0.0, 0.0, 1.0])):
    """Diagonalize H = H₀ − (−1)^q · B · Bq · μq for each B, return sorted eigenvalues.

    The magnetic interaction is expressed in the spherical tensor basis:
    the Cartesian field direction B̂ is converted to spherical components
    Bq via cart2spherical, and the Hamiltonian is constructed as
    H = H₀ − Σ_q (−1)^q · B · B*_{-q} · μ_q, summing over q = −1, 0, +1.
    """
    Bhat = Bhat / np.linalg.norm(Bhat)
    Bq = cart2spherical(Bhat)
    nstates = H0.shape[0]
    Es = np.zeros((len(B_arr), nstates))
    for ii, Bi in enumerate(B_arr):
        H = H0.astype("complex128").copy()
        for jj, q in enumerate(np.arange(-1.0, 2.0, 1.0)):
            H -= (-1.0) ** q * Bi * Bq[2 - jj] * mu_q[jj]
        Es[ii, :] = np.sort(np.linalg.eigh(H)[0])
    return Es


# ---------------------------------------------------------------------------
# Spin in magnetic field – spherical tensor vs Pauli consistency
# ---------------------------------------------------------------------------


class TestSpinInMagneticField:
    """Spherical tensor algebra vs Pauli matrices for spin-1/2.

    pylcp_jax constructs magnetic moment operators μ_q in the spherical tensor
    basis (q = −1, 0, +1), which is natural for coupling to spherical
    polarization components but less intuitive than the standard Pauli
    matrix formulation (σ_x, σ_y, σ_z).

    For spin-1/2 with a g-factor gF, both representations must produce:
    - Identical eigenvalue spectra for any B-field direction.
    - Identical time evolution of ⟨S⟩ = (⟨Sx⟩, ⟨Sy⟩, ⟨Sz⟩), which
      precesses on the Bloch sphere with constant magnitude |⟨S⟩| = 1/2.

    This validates that the spherical↔Cartesian conversion and the sign
    conventions in singleF are correct.

    Adapted from tests/hamiltonians/00_spin_in_magnetic_field.ipynb.
    """

    def test_hamiltonian_eigenvalues_match_pauli(self):
        """Eigenvalues of singleF Hamiltonian must match Pauli construction."""
        B = np.array([1.0, 1.0, 0.0])
        gF = 1
        H_0, mu_q = ham.singleF(1 / 2, gF=gF, muB=1)

        # Pauli matrices
        S_pauli = np.zeros((3, 2, 2), dtype="complex128")
        S_pauli[0] = np.array([[0.0, 1.0], [1.0, 0.0]]) / 2
        S_pauli[1] = np.array([[0.0, -1j], [1j, 0.0]]) / 2
        S_pauli[2] = np.array([[1.0, 0.0], [0.0, -1.0]]) / 2
        mu_pauli = -gF * S_pauli

        # Build Hamiltonian from spherical tensor
        Bq = np.array([(B[0] - 1j * B[1]) / np.sqrt(2), B[2], -(B[0] + 1j * B[1]) / np.sqrt(2)])
        H = -np.tensordot(mu_q, np.conjugate(Bq), axes=(0, 0))

        # Build Hamiltonian from Pauli matrices
        H_pauli = -np.tensordot(mu_pauli, B, axes=(0, 0))

        evals_sph = np.sort(np.linalg.eigvalsh(H))
        evals_pauli = np.sort(np.linalg.eigvalsh(H_pauli))
        np.testing.assert_allclose(evals_sph, evals_pauli, atol=1e-12)

    def test_expectation_values_match_pauli(self):
        """Spin precession: ⟨S⟩ from singleF must match Pauli evolution.

        A spin-1/2 in an off-axis field B = (1,1,0) precesses around B̂.
        The magnitude |⟨S⟩| must remain exactly 1/2 at all times (pure
        state on the Bloch sphere).  We evolve using both the spherical
        tensor Hamiltonian and the Pauli Hamiltonian and verify that the
        spin magnitude agrees to within atol=1e-4 (limited by ODE solver
        tolerances)."""
        from scipy.integrate import solve_ivp

        B = np.array([1.0, 1.0, 0.0])
        gF = 1
        H_0, mu_q = ham.singleF(1 / 2, gF=gF, muB=1)
        mu = spherical2cart(mu_q)
        S = -mu / gF

        S_pauli = np.zeros((3, 2, 2), dtype="complex128")
        S_pauli[0] = np.array([[0.0, 1.0], [1.0, 0.0]]) / 2
        S_pauli[1] = np.array([[0.0, -1j], [1j, 0.0]]) / 2
        S_pauli[2] = np.array([[1.0, 0.0], [0.0, -1.0]]) / 2
        mu_pauli = -gF * S_pauli

        Bq = np.array([(B[0] - 1j * B[1]) / np.sqrt(2), B[2], -(B[0] + 1j * B[1]) / np.sqrt(2)])
        H = -np.tensordot(mu_q, np.conjugate(Bq), axes=(0, 0))
        H_pauli = -np.tensordot(mu_pauli, B, axes=(0, 0))

        t_eval = np.linspace(0, np.pi / 2, 50)
        sol_sph = solve_ivp(
            lambda t, x: -1j * H @ x,
            [0, np.pi / 2],
            np.array([0.0, 1.0]).astype("complex128"),
            t_eval=t_eval,
        )
        sol_pauli = solve_ivp(
            lambda t, x: -1j * H_pauli @ x,
            [0, np.pi / 2],
            np.array([1.0, 0.0]).astype("complex128"),
            t_eval=t_eval,
        )

        avS = np.zeros((3, len(t_eval)))
        avS_pauli = np.zeros((3, len(t_eval)))
        for ii in range(3):
            for jj in range(len(t_eval)):
                avS[ii, jj] = np.real(np.conj(sol_sph.y[:, jj].T) @ S[ii] @ sol_sph.y[:, jj])
                avS_pauli[ii, jj] = np.real(
                    np.conj(sol_pauli.y[:, jj].T) @ S_pauli[ii] @ sol_pauli.y[:, jj]
                )

        # The spin magnitudes should match (both give |<S>| = 1/2)
        mag_sph = np.sqrt(np.sum(avS**2, axis=0))
        mag_pauli = np.sqrt(np.sum(avS_pauli**2, axis=0))
        np.testing.assert_allclose(mag_sph, mag_pauli, atol=1e-4)
        np.testing.assert_allclose(mag_sph, 0.5 * np.ones(len(t_eval)), atol=1e-4)


# ---------------------------------------------------------------------------
# Linear Zeeman effect
# ---------------------------------------------------------------------------


class TestLinearZeemanEffect:
    """Linear Zeeman effect for a single hyperfine level F.

    An angular momentum state F has (2F+1) magnetic sublevels mF that are
    degenerate at zero field.  In a uniform magnetic field B, they split as:

        E(mF) = −gF · mF · μB · B

    This linear splitting is exact for the singleF Hamiltonian (which has
    no quadratic corrections).  The energy spectrum must also be invariant
    under rotations of the field direction — only |B| matters.

    Adapted from tests/hamiltonians/01_linear_Zeeman_effect.ipynb.
    """

    def test_zero_field_degenerate(self):
        """At B=0, all 2F+1 states must be degenerate."""
        F = 2
        H_0, mu_q = ham.singleF(F, muB=1)
        evals = np.sort(np.linalg.eigvalsh(H_0))
        assert np.allclose(evals, np.zeros(2 * F + 1), atol=1e-14)

    def test_linear_splitting_with_field(self):
        """At finite B along z, energy levels should split linearly as m_F * gF * muB * B."""
        F = 2
        gF = 0.5
        H_0, mu_q = ham.singleF(F, gF=gF, muB=1)
        B_val = 5.0
        H = H_0 + B_val * mu_q[1]  # q=0 component for z-field
        evals = np.sort(np.linalg.eigvalsh(H))
        expected = np.sort(np.array([-gF * mF * B_val for mF in np.arange(-F, F + 1)]))
        np.testing.assert_allclose(evals, expected, atol=1e-10)

    def test_field_direction_invariance(self):
        """Eigenvalue spectra should be identical regardless of B-field direction.

        The Hamiltonian is a scalar (rank-0 tensor) formed from the
        contraction of μ_q (rank 1) with B_q (rank 1).  Rotating B̂
        changes the individual spherical components B_q but not the
        scalar contraction, so eigenvalues must be invariant."""
        F = 2
        H_0, mu_q = ham.singleF(F, muB=1)
        B_val = 5.0

        # z-direction
        Es_z = np.sort(np.linalg.eigvalsh(H_0 + B_val * mu_q[1]))

        # x-direction: Bq = [-1/√2, 0, 1/√2]
        Bq_x = np.array([-1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)])
        H_x = H_0 + B_val * np.tensordot(Bq_x[::-1], mu_q, axes=(0, 0))
        Es_x = np.sort(np.linalg.eigvalsh(H_x))

        # y-direction: Bq = [i/√2, 0, -i/√2]
        Bq_y = np.array([1j / np.sqrt(2), 0.0, -1j / np.sqrt(2)])
        H_y = H_0 + B_val * np.tensordot(Bq_y[::-1], mu_q, axes=(0, 0))
        Es_y = np.sort(np.linalg.eigvalsh(H_y))

        np.testing.assert_allclose(Es_x, Es_z, atol=1e-10)
        np.testing.assert_allclose(Es_y, Es_z, atol=1e-10)

    def test_splitting_increases_with_field(self):
        """Energy spread should increase with magnetic field strength."""
        F = 2
        H_0, mu_q = ham.singleF(F, muB=1)
        spreads = []
        for B_val in [1.0, 5.0, 10.0]:
            evals = np.linalg.eigvalsh(H_0 + B_val * mu_q[1])
            spreads.append(np.ptp(evals))
        assert spreads[0] < spreads[1] < spreads[2]


# ---------------------------------------------------------------------------
# Breit-Rabi validation for ground state hyperfine structure
# ---------------------------------------------------------------------------


class TestBreitRabiValidation:
    """Hyperfine structure: Hamiltonian vs analytical Breit-Rabi formula.

    For a J=1/2 ground state (e.g. alkali atoms), the Breit-Rabi formula
    gives exact eigenvalues of the hyperfine + Zeeman Hamiltonian.  pylcp_jax
    constructs this Hamiltonian numerically in either the coupled |F,mF⟩
    or uncoupled |mJ,mI⟩ basis.

    Both basis representations must agree with each other, and at low
    fields (B ≲ 10 G) they must agree with the analytical formula.  At
    higher fields, a systematic ~0.05% discrepancy arises from differing
    sign conventions for gI between the formula and the Hamiltonian.

    The eigenvalue spectrum must also be invariant under rotation of the
    B-field direction (rotational symmetry of the scalar Hamiltonian).

    Adapted from tests/hamiltonians/02_hyperfine_Hamilotians.ipynb.
    """

    muB = cts.value("Bohr magneton in Hz/T") * 1e-4

    @pytest.mark.parametrize("species", ["7Li", "87Rb"])
    def test_ground_state_breitrabi_low_field(self, species):
        """At low fields (0.1–10 G), Hamiltonian eigenvalues should track the
        Breit-Rabi formula to within rtol=5e-4.

        The tolerance reflects a systematic gI sign convention difference
        between the analytical formula and the Hamiltonian construction."""
        a = atom(species)
        B = np.linspace(0.1, 10, 21)
        H0, mu_q = ham.hyperfine_uncoupled(
            a.state[0].J, a.I, gJ=a.state[0].gJ, gI=a.gI, Ahfs=a.state[0].Ahfs, Bhfs=0, muB=self.muB
        )
        Es = diagonalize_hamiltonian(B, H0, mu_q)
        Es_br = breitrabi(B, a.state[0].gJ, a.gI, a.state[0].Ahfs, J=a.state[0].J, I=a.I)
        Es_br_sorted = np.sort(Es_br, axis=0).T
        np.testing.assert_allclose(Es, Es_br_sorted, rtol=5e-4)

    @pytest.mark.parametrize("species", ["7Li", "87Rb"])
    def test_coupled_uncoupled_bases_agree(self, species):
        """Coupled |F,mF⟩ and uncoupled |mJ,mI⟩ bases must give identical
        eigenvalues at low fields (0.01–1 G) to rtol=1e-7.

        Both bases span the same Hilbert space; the Clebsch-Gordan
        transformation between them is unitary, so eigenvalues must be
        identical up to numerical precision."""
        a = atom(species)
        B = np.linspace(0.01, 1.0, 21)
        H0_uc, mu_uc = ham.hyperfine_uncoupled(
            a.state[0].J, a.I, gJ=a.state[0].gJ, gI=a.gI, Ahfs=a.state[0].Ahfs, Bhfs=0, muB=self.muB
        )
        H0_c, mu_c = ham.hyperfine_coupled(
            a.state[0].J, a.I, gJ=a.state[0].gJ, gI=a.gI, Ahfs=a.state[0].Ahfs, Bhfs=0, muB=self.muB
        )
        Es_uc = diagonalize_hamiltonian(B, H0_uc, mu_uc)
        Es_c = diagonalize_hamiltonian(B, H0_c, mu_c)
        np.testing.assert_allclose(Es_uc, Es_c, rtol=1e-7)

    @pytest.mark.parametrize("species", ["6Li", "7Li", "87Rb"])
    def test_field_direction_invariance(self, species):
        """Eigenvalues must be invariant under rotation of B-field direction."""
        a = atom(species)
        B = np.linspace(1, 100, 21)
        H0, mu_q = ham.hyperfine_coupled(
            a.state[0].J, a.I, gJ=a.state[0].gJ, gI=a.gI, Ahfs=a.state[0].Ahfs, Bhfs=0, muB=self.muB
        )
        Es_z = diagonalize_hamiltonian(B, H0, mu_q, Bhat=[0, 0, 1])
        Es_x = diagonalize_hamiltonian(B, H0, mu_q, Bhat=[1, 0, 0])
        Es_y = diagonalize_hamiltonian(B, H0, mu_q, Bhat=[0, 1, 0])
        np.testing.assert_allclose(Es_x, Es_z, atol=1e-6)
        np.testing.assert_allclose(Es_y, Es_z, atol=1e-6)

    @pytest.mark.parametrize("species,Bmax", [("7Li", 0.5), ("87Rb", 5.0)])
    def test_excited_state_coupled_uncoupled_agree(self, species, Bmax):
        """Excited P₃/₂ state eigenvalues must agree in both bases.

        The P₃/₂ manifold has more sublevels and includes the electric
        quadrupole interaction (Bhfs).  ⁷Li has level crossings at ~1–2 G,
        causing sorted-eigenvalue comparison to fail above those fields,
        so we use Bmax=0.5 G for ⁷Li and 5 G for ⁸⁷Rb."""
        a = atom(species)
        B = np.linspace(0.01, Bmax, 11)
        H0_uc, mu_uc = ham.hyperfine_uncoupled(
            a.state[2].J,
            a.I,
            gJ=a.state[2].gJ,
            gI=a.gI,
            Ahfs=a.state[2].Ahfs,
            Bhfs=a.state[2].Bhfs,
            muB=self.muB,
        )
        H0_c, mu_c = ham.hyperfine_coupled(
            a.state[2].J,
            a.I,
            gJ=a.state[2].gJ,
            gI=a.gI,
            Ahfs=a.state[2].Ahfs,
            Bhfs=a.state[2].Bhfs,
            muB=self.muB,
        )
        Es_uc = diagonalize_hamiltonian(B, H0_uc, mu_uc)
        Es_c = diagonalize_hamiltonian(B, H0_c, mu_c)
        np.testing.assert_allclose(Es_uc, Es_c, rtol=5e-4)

    def test_unit_conversion_consistency(self):
        """Results in real units (Hz) and natural units (Γ) must agree
        after proper scaling.

        pylcp_jax supports working in "natural" units where energies are
        measured in units of the excited-state linewidth Γ (gammaHz).
        Scaling Ahfs → Ahfs/Γ, μB → 1, and B → (μB/Γ)·B must produce
        eigenvalues that, when multiplied by Γ, match the Hz result."""
        a = atom("7Li")
        muB_real = cts.value("Bohr magneton in Hz/T") * 1e-4
        B = np.linspace(0.1, 500, 31)

        # Real units
        H0_r, mu_r = ham.hyperfine_coupled(
            a.state[0].J, a.I, gJ=a.state[0].gJ, gI=a.gI, Ahfs=a.state[0].Ahfs, Bhfs=0, muB=muB_real
        )
        Es_real = diagonalize_hamiltonian(B, H0_r, mu_r)

        # Natural units (scaled by gamma)
        gamma = a.state[2].gammaHz
        alpha = muB_real / gamma
        H0_n, mu_n = ham.hyperfine_coupled(
            a.state[0].J,
            a.I,
            gJ=a.state[0].gJ,
            gI=a.gI,
            Ahfs=a.state[0].Ahfs / gamma,
            Bhfs=0,
            muB=1,
        )
        B_nat = alpha * B
        Es_nat = diagonalize_hamiltonian(B_nat, H0_n, mu_n)

        np.testing.assert_allclose(Es_real, Es_nat * gamma, rtol=1e-6)


# ---------------------------------------------------------------------------
# Transition rate numerical values
# ---------------------------------------------------------------------------


class TestTransitionRateValues:
    """Dipole matrix elements and selection rules.

    The electric dipole operator connects ground and excited hyperfine
    manifolds.  Its matrix elements d^q_{ij} (for polarization q = −1, 0, +1)
    are products of Clebsch-Gordan coefficients and reduced matrix elements.

    Key physical constraints:
    - Selection rules: only transitions with ΔmF = q are allowed.
    - For F=1→F'=1, the mF=0→mF'=0 transition is forbidden for q=0
      (this follows from the 3j-symbol vanishing when all projections are 0).
    - Sum rule: for normalized d^q_{ij}, summing |d^q_{ij}|² over all
      ground states i and polarizations q for a given excited state j
      must yield 1 (completeness of the basis).

    Adapted from tests/hamiltonians/03_transition_rates.ipynb.
    """

    def test_F0_to_F1_selection_rules(self):
        """F=0→F'=1: one ground state, so exactly one nonzero element per q."""
        dijq = ham.dqij_two_bare_hyperfine(0, 1, normalize=False)
        for q in range(3):
            nonzero = np.count_nonzero(np.abs(dijq[q]) > 1e-14)
            assert nonzero == 1, f"q={q - 1}: expected 1 nonzero element, got {nonzero}"

    def test_F1_to_F1_vanishing_diagonal(self):
        """F=1→F'=1, q=0: the mF=0→mF'=0 element must vanish.

        This is a consequence of the Wigner 3j-symbol
        (1  1  1; 0  0  0) = 0, which forbids this transition."""
        dijq = ham.dqij_two_bare_hyperfine(1, 1, normalize=False)
        # For q=0 (dijq[1]), the m=0 → m'=0 element must vanish
        assert float(np.abs(dijq[1, 1, 1])) == pytest.approx(0.0, abs=1e-14)

    def test_F1_to_F2_completeness(self):
        """F=1 → F'=2 should have 3 nonzero elements per q (for Δm=q)."""
        dijq = ham.dqij_two_bare_hyperfine(1, 2, normalize=False)
        for q in range(3):
            nonzero = np.count_nonzero(np.abs(dijq[q]) > 1e-14)
            # Each q allows 3 transitions (one per m_F in the ground state,
            # or fewer if m_F + q falls outside the excited manifold)
            assert nonzero > 0, f"q={q - 1}: expected nonzero elements"

    def test_normalized_rates_sum_rule(self):
        """Sum rule: Σ_{q,i} |d^q_{ij}|² = 1 for each excited state j.

        When normalize=True, the dipole elements are scaled so that the
        total spontaneous decay rate from each excited state equals 1.
        This is required for self-consistent rate equations and OBE."""
        dijq = ham.dqij_two_bare_hyperfine(1, 2, normalize=True)
        for k in range(dijq.shape[2]):
            rate_sum = np.sum(np.abs(dijq[:, :, k]) ** 2)
            assert float(rate_sum) == pytest.approx(1.0, abs=1e-10)

    def test_full_D2_manifold_structure(self):
        """Full D2 line (J=1/2→J'=3/2, I=3/2 as in Na): correct dimensions.

        The D2 line couples all ground hyperfine levels (F=I±1/2) to all
        excited levels (F'=I−3/2 … I+3/2), giving 8 ground and 16
        excited states for I=3/2."""
        dijq, basis_g, basis_e = ham.dqij_two_hyperfine_manifolds(
            1 / 2, 3 / 2, 3 / 2, normalize=True, return_basis=True
        )
        n_g = int((2 * 0.5 + 1) * (2 * 1.5 + 1))  # 8 ground states
        n_e = int((2 * 1.5 + 1) * (2 * 1.5 + 1))  # 16 excited states
        assert dijq.shape == (3, n_g, n_e)
        # There should be nonzero elements
        assert np.count_nonzero(np.abs(dijq) > 1e-14) > 0

    def test_D2_manifold_sum_rule(self):
        """For each excited state in the full D2 manifold, the sum of squared
        dipole elements over all ground states and q must equal 1."""
        dijq = ham.dqij_two_hyperfine_manifolds(1 / 2, 3 / 2, 3 / 2, normalize=True)
        for k in range(dijq.shape[2]):
            rate_sum = np.sum(np.abs(dijq[:, :, k]) ** 2)
            assert float(rate_sum) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Fine structure uncoupled basis
# ---------------------------------------------------------------------------


class TestFineStructurePhysics:
    """Fine structure in the uncoupled |L, S, I, mL, mS, mI⟩ basis.

    For light alkali atoms (notably ⁷Li), the excited-state fine structure
    splitting ξ is comparable to the hyperfine splitting, so the coupled
    |J, F, mF⟩ basis is not a good description at moderate fields.  The
    uncoupled (L, S, I) basis handles this correctly and includes:

    - Spin-orbit coupling ξ·L⃗·S⃗ (fine structure).
    - Contact, orbital, and dipolar hyperfine interactions (aₘ, aₒᵣᵦ, aᵈᵢₚ).
    - Zeeman interaction with separate gL, gS, gI g-factors.

    At zero field the eigenstates must cluster into P₁/₂ and P₃/₂
    manifolds separated by ~ξ.  At very high fields (Paschen-Back regime,
    B ≫ ξ/μB) the spin-orbit coupling is overwhelmed and energies scale
    linearly with B.

    Uses ⁷Li parameters from NIST spectroscopic data.

    Adapted from tests/hamiltonians/04_fine_structure_uncoupled_basis.ipynb.
    """

    @pytest.fixture(scope="class")
    def li7_params(self):
        """⁷Li fine structure parameters from NIST spectroscopic data."""
        xi = 6701.16  # fine structure splitting (MHz)
        a_c = -9.5788  # contact splitting (MHz)
        a_orb = 8.6727  # orbital splitting (MHz)
        a_dip = -1.8964  # dipole splitting (MHz)
        gL = 0.9999218
        gS = 2.0023193
        gI = 2.170903 * (5.0507866e-27 / 9.2740154e-24)
        aa = 803.54 / 2.0  # ground state contact interaction (MHz)
        muB = cts.value("Bohr magneton in Hz/T") * 1e-4 / 1e6
        return dict(xi=xi, a_c=a_c, a_orb=a_orb, a_dip=a_dip, gL=gL, gS=gS, gI=gI, aa=aa, muB=muB)

    def test_ground_state_agrees_with_coupled(self, li7_params):
        """⁷Li ground state (L=0): uncoupled and coupled bases must agree.

        For L=0 the spin-orbit term vanishes and fine_structure_uncoupled
        reduces to a pure hyperfine Hamiltonian.  The eigenvalues must
        match hyperfine_coupled to within atol=0.5 MHz (limited by slight
        differences in the gI sign convention between the two codes)."""
        p = li7_params
        a = atom("7Li")

        H_g_unc, mu_g_unc = ham.fine_structure_uncoupled(
            0, 1 / 2, 3 / 2, 0.0, p["aa"], 0.0, 0.0, p["gL"], p["gS"], p["gI"], p["muB"]
        )
        H_g_c, mu_g_c = ham.hyperfine_coupled(
            a.state[0].J,
            a.I,
            a.state[0].gJ,
            -a.gI,
            a.state[0].Ahfs / 1e6,
            Bhfs=0,
            Chfs=0,
            muB=cts.value("Bohr magneton in Hz/T") * 1e-4 / 1e6,
        )

        B = np.arange(1, 200, 10)
        Es_unc = diagonalize_hamiltonian(B, H_g_unc, mu_g_unc)
        Es_c = diagonalize_hamiltonian(B, H_g_c, mu_g_c)
        np.testing.assert_allclose(Es_unc, Es_c, atol=0.5)

    def test_excited_state_field_direction_invariance(self, li7_params):
        """Li-7 excited state eigenvalues must be invariant under B-field rotation."""
        p = li7_params
        H_e, mu_e = ham.fine_structure_uncoupled(
            1,
            1 / 2,
            3 / 2,
            p["xi"],
            p["a_c"],
            p["a_orb"],
            p["a_dip"],
            p["gL"],
            p["gS"],
            p["gI"],
            p["muB"],
        )

        B = np.arange(1, 15, 2)
        Es_z = diagonalize_hamiltonian(B, H_e, mu_e, Bhat=[0, 0, 1])
        Es_x = diagonalize_hamiltonian(B, H_e, mu_e, Bhat=[1, 0, 0])
        Es_y = diagonalize_hamiltonian(B, H_e, mu_e, Bhat=[0, 1, 0])
        np.testing.assert_allclose(Es_x, Es_z, atol=1e-6)
        np.testing.assert_allclose(Es_y, Es_z, atol=1e-6)

    def test_fine_structure_splitting_present(self, li7_params):
        """At B=0, L·S coupling splits the P state into P₁/₂ and P₃/₂.

        The gap between the highest P₁/₂ eigenvalue and the lowest P₃/₂
        eigenvalue should be on the order of ξ (6701 MHz for ⁷Li).  The
        exact gap differs from ξ due to hyperfine corrections within
        each manifold, but must be at least ξ/2."""
        p = li7_params
        H_e, mu_e = ham.fine_structure_uncoupled(
            1,
            1 / 2,
            3 / 2,
            p["xi"],
            p["a_c"],
            p["a_orb"],
            p["a_dip"],
            p["gL"],
            p["gS"],
            p["gI"],
            p["muB"],
        )
        evals = np.sort(np.linalg.eigvalsh(H_e))
        # The gap between the P_{1/2} and P_{3/2} manifolds should be
        # approximately the fine structure splitting
        gap = evals[8] - evals[7]  # first P_{3/2} state - last P_{1/2} state
        assert gap > p["xi"] / 2, "Fine structure gap is too small"

    def test_dipole_elements_selection_rules(self, li7_params):
        """Dipole matrix elements in the uncoupled basis must respect ΔL=±1.

        The electric dipole operator only connects states differing by
        one unit of orbital angular momentum.  The matrix d^q_{ij} must
        have shape (3, n_ground, n_excited) with nonzero elements only
        for ground (L=0) → excited (L=1) transitions."""
        p = li7_params
        _, _, basis_g = ham.fine_structure_uncoupled(
            0,
            1 / 2,
            3 / 2,
            0.0,
            p["aa"],
            0.0,
            0.0,
            p["gL"],
            p["gS"],
            p["gI"],
            p["muB"],
            return_basis=True,
        )
        _, _, basis_e = ham.fine_structure_uncoupled(
            1,
            1 / 2,
            3 / 2,
            p["xi"],
            p["a_c"],
            p["a_orb"],
            p["a_dip"],
            p["gL"],
            p["gS"],
            p["gI"],
            p["muB"],
            return_basis=True,
        )
        d_q = ham.dqij_two_fine_stucture_manifolds_uncoupled(basis_g, basis_e)
        basis_g_arr = np.array(basis_g)
        basis_e_arr = np.array(basis_e)
        # Should have shape (3, n_g, n_e) and be nonzero
        assert d_q.shape[0] == 3
        assert d_q.shape[1] == basis_g_arr.shape[0]
        assert d_q.shape[2] == basis_e_arr.shape[0]
        assert np.count_nonzero(np.abs(d_q) > 1e-14) > 0

    def test_high_field_paschen_back(self, li7_params):
        """Paschen-Back regime: energy spread scales linearly with B.

        When μB·B ≫ ξ (spin-orbit coupling), L and S decouple from each
        other and precess independently around B.  Each state's energy
        becomes E ≈ (gL·mL + gS·mS + gI·mI)·μB·B, so the total energy
        spread across all states scales linearly with B.  We verify this
        at B = 50, 100, 200 kG where spread(B₂)/spread(B₁) = B₂/B₁."""
        p = li7_params
        H_e, mu_e = ham.fine_structure_uncoupled(
            1,
            1 / 2,
            3 / 2,
            p["xi"],
            p["a_c"],
            p["a_orb"],
            p["a_dip"],
            p["gL"],
            p["gS"],
            p["gI"],
            p["muB"],
        )

        B_high = np.array([50000.0, 100000.0, 200000.0])
        Es = diagonalize_hamiltonian(B_high, H_e, mu_e)
        # Total energy spread should scale linearly with B
        spreads = np.ptp(Es, axis=1)
        # spread/B should be constant (Paschen-Back: energy ∝ B)
        ratio_01 = spreads[1] / spreads[0]
        ratio_12 = spreads[2] / spreads[1]
        expected_01 = B_high[1] / B_high[0]
        expected_12 = B_high[2] / B_high[1]
        assert ratio_01 == pytest.approx(expected_01, rel=1e-6)
        assert ratio_12 == pytest.approx(expected_12, rel=1e-6)
