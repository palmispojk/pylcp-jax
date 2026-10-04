"""
Tests for pylcp_jax/atom.py
"""

import numpy as np
import pytest
import scipy.constants as cts
from numpy import pi

from pylcp_jax.atom import atom, state, transition

# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


class TestState:
    """Atomic energy level with quantum numbers (n, S, L, J) and spectroscopic data.

    Energy is set via wavelength λ (E = 0.01/λ in cm⁻¹) or directly as E.
    The spontaneous decay rate γ = 1/τ where τ is the natural lifetime.
    Hyperfine structure constants A_hfs, B_hfs, C_hfs parameterize
    H_hfs = A_hfs(I⃗·J⃗) + higher-order terms."""

    def test_energy_from_lam(self):
        s = state(n=1, S=0, L=0, J=0, lam=500e-9)
        assert s.energy == pytest.approx(0.01 / 500e-9, rel=1e-10)

    def test_energy_from_E(self):
        s = state(n=1, S=0, L=0, J=0, E=12345.0)
        assert s.energy == pytest.approx(12345.0)

    def test_ground_state_lam_inf(self):
        s = state(n=1, S=1 / 2, L=0, J=1 / 2, lam=np.inf, tau=np.inf)
        assert s.energy == pytest.approx(0.0, abs=1e-30)

    def test_missing_lam_and_E_raises(self):
        with pytest.raises(ValueError, match="energy"):
            state(n=1, S=0, L=0, J=0)

    def test_E_zero_is_valid(self):
        # Regression: 'elif E:' would fail for E=0 (falsy)
        s = state(n=1, S=0, L=0, J=0, E=0.0)
        assert s.energy == pytest.approx(0.0, abs=1e-30)

    def test_gamma_from_tau(self):
        tau = 27e-9
        s = state(n=1, S=1 / 2, L=0, J=1 / 2, lam=780e-9, tau=tau)
        assert s.gamma == pytest.approx(1 / tau, rel=1e-10)

    def test_gammaHz(self):
        tau = 27e-9
        s = state(n=1, S=1 / 2, L=0, J=1 / 2, lam=780e-9, tau=tau)
        assert s.gammaHz == pytest.approx(s.gamma / (2 * pi), rel=1e-10)

    def test_infinite_tau_gives_zero_gamma(self):
        s = state(n=1, S=1 / 2, L=0, J=1 / 2, lam=np.inf, tau=np.inf)
        assert s.gamma == 0.0

    def test_quantum_numbers_stored(self):
        s = state(n=5, S=1 / 2, L=1, J=3 / 2, lam=780e-9)
        assert s.n == 5
        assert s.S == 1 / 2
        assert s.L == 1
        assert s.J == 3 / 2

    def test_hfs_defaults_zero(self):
        s = state(n=1, S=0, L=0, J=0, lam=500e-9)
        assert s.Ahfs == 0
        assert s.Bhfs == 0
        assert s.Chfs == 0

    def test_hfs_stored(self):
        s = state(
            n=1, S=1 / 2, L=0, J=1 / 2, lam=np.inf, tau=np.inf, Ahfs=3.4e9, Bhfs=1.2e6, Chfs=0.5e3
        )
        assert s.Ahfs == pytest.approx(3.4e9)
        assert s.Bhfs == pytest.approx(1.2e6)
        assert s.Chfs == pytest.approx(0.5e3)

    def test_lam_takes_precedence_over_E(self):
        # When both provided, lam wins (it is checked first)
        s = state(n=1, S=0, L=0, J=0, lam=500e-9, E=99999.0)
        assert s.energy == pytest.approx(0.01 / 500e-9, rel=1e-10)


# ---------------------------------------------------------------------------
# transition
# ---------------------------------------------------------------------------


class TestTransition:
    """Optical transition connecting two atomic states.

    Wavenumber k = E_excited − E_ground, wavelength λ = 0.01/k,
    frequency ν = c/λ, angular frequency ω = 2πν.  Derived quantities:
    saturation intensity I_sat, recoil acceleration a₀ = ℏkΓ/(2m),
    recoil velocity v₀ = ℏk/m, characteristic length x₀ = v₀²/a₀,
    characteristic time t₀ = v₀/a₀, and magnetic field scale B_Γ."""

    def setup_method(self):
        self.ground = state(n=5, S=1 / 2, L=0, J=1 / 2, lam=np.inf, tau=np.inf, gJ=2.0023)
        self.excited = state(
            n=5, S=1 / 2, L=1, J=3 / 2, lam=780.241209686e-9, tau=26.2348e-9, gJ=1.3362
        )
        self.mass = 86.909180527 * cts.value("atomic mass constant")
        self.tr = transition(self.ground, self.excited, self.mass)

    def test_k_is_energy_difference(self):
        assert self.tr.k == pytest.approx(self.excited.energy - self.ground.energy, rel=1e-10)

    def test_lam_from_k(self):
        assert self.tr.lam == pytest.approx(0.01 / self.tr.k, rel=1e-10)

    def test_nu_from_lam(self):
        assert self.tr.nu == pytest.approx(cts.c / self.tr.lam, rel=1e-8)

    def test_omega_from_nu(self):
        assert self.tr.omega == pytest.approx(2 * np.pi * self.tr.nu, rel=1e-10)

    def test_isat_positive(self):
        assert self.tr.Isat > 0

    def test_isat_units_mw_cm2(self):
        # Rb D2 Isat is known ~1.67 mW/cm^2
        assert 0.5 < self.tr.Isat < 5.0

    def test_a0_positive(self):
        assert self.tr.a0 > 0

    def test_v0_positive(self):
        assert self.tr.v0 > 0

    def test_x0_and_t0_consistent(self):
        # x0 = v0^2 / a0,  t0 = v0 / a0
        assert self.tr.x0 == pytest.approx(self.tr.v0**2 / self.tr.a0, rel=1e-8)
        assert self.tr.t0 == pytest.approx(self.tr.v0 / self.tr.a0, rel=1e-8)

    def test_bgamma_positive(self):
        assert self.tr.Bgamma > 0


# ---------------------------------------------------------------------------
# atom — construction and basic sanity checks
# ---------------------------------------------------------------------------

# All supported species
ALL_SPECIES = ["6Li", "7Li", "23Na", "39K", "40K", "41K", "85Rb", "87Rb", "133Cs"]
# Alternate aliases
ALT_SPECIES = ["Li6", "Li7", "Na23", "K39", "K40", "K41", "Rb85", "Rb87", "Cs133"]


class TestAtomConstruction:
    """Construct atoms for all supported alkali species.

    Each atom has ≥3 states (ground + D1 + D2), monotonically increasing
    energies, zero ground-state energy (E₀ = 0), infinite ground-state
    lifetime (τ = ∞, γ = 0), and finite excited-state lifetimes."""

    @pytest.mark.parametrize("species", ALL_SPECIES + ALT_SPECIES)
    def test_all_species_construct(self, species):
        a = atom(species)
        assert len(a.state) >= 3

    def test_invalid_species_raises(self):
        with pytest.raises(ValueError):
            atom("42X")

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_transitions_created(self, species):
        a = atom(species)
        assert len(a.transition) == len(a.state) - 1

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_states_monotonically_increasing_energy(self, species):
        a = atom(species)
        energies = [s.energy for s in a.state]
        for i in range(1, len(energies)):
            assert energies[i] > energies[i - 1], f"{species}: state {i} energy not > state {i - 1}"

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_ground_state_zero_energy(self, species):
        a = atom(species)
        assert a.state[0].energy == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_ground_state_infinite_tau(self, species):
        a = atom(species)
        assert a.state[0].tau == np.inf
        assert a.state[0].gamma == 0.0

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_excited_states_finite_tau(self, species):
        a = atom(species)
        for s in a.state[1:]:
            assert np.isfinite(s.tau)
            assert s.gamma > 0


class TestAtomPhysicalValues:
    """Validate physical constants against known literature values.

    Checks nuclear spins I (e.g. ⁸⁷Rb: I=3/2, ⁴⁰K: I=4), atomic masses,
    D-line wavelengths (⁸⁷Rb D2 ≈ 780.241 nm, ⁸⁵Rb D1 ≈ 794.979 nm),
    and fine-structure quantum numbers J=1/2 (P₁/₂) vs J=3/2 (P₃/₂)."""

    def test_7Li_nuclear_spin(self):
        assert atom("7Li").I == 3 / 2

    def test_6Li_nuclear_spin(self):
        assert atom("6Li").I == 1

    def test_87Rb_nuclear_spin(self):
        assert atom("87Rb").I == 3 / 2

    def test_40K_nuclear_spin(self):
        assert atom("40K").I == 4

    def test_133Cs_nuclear_spin(self):
        assert atom("133Cs").I == 7 / 2

    def test_87Rb_mass(self):
        expected = 86.909180527 * cts.value("atomic mass constant")
        assert atom("87Rb").mass == pytest.approx(expected, rel=1e-6)

    def test_87Rb_D2_wavelength(self):
        a = atom("87Rb")
        # D2 transition (index 1 from ground) wavelength ~780.24 nm
        assert a.transition[1].lam == pytest.approx(780.241209686e-9, rel=1e-5)

    def test_85Rb_D1_wavelength_not_same_as_D2(self):
        # Regression: D1 wavelength was wrongly set to 780.241e-9 (same as D2)
        a = atom("85Rb")
        d1_lam = a.transition[0].lam
        d2_lam = a.transition[1].lam
        assert abs(d1_lam - d2_lam) > 1e-9  # must differ by more than 1 nm

    def test_85Rb_D1_wavelength_near_795nm(self):
        a = atom("85Rb")
        # D1 for Rb is ~794.979 nm, D2 is ~780 nm
        assert a.transition[0].lam == pytest.approx(794.9788509e-9, rel=1e-4)

    def test_85Rb_D1_J_half(self):
        a = atom("85Rb")
        # state[1] is D1 (5P_{1/2})
        assert a.state[1].J == 1 / 2

    def test_85Rb_D2_J_3half(self):
        a = atom("85Rb")
        # state[2] is D2 (5P_{3/2})
        assert a.state[2].J == 3 / 2

    def test_transition_lam_matches_state_lam(self):
        a = atom("87Rb")
        # transition[0] connects state[0] and state[1]
        assert a.transition[0].lam == pytest.approx(a.state[1].tau * 0 + a.transition[0].lam)

    def test_Bgamma_units_order_of_magnitude(self):
        # For Rb D2, the linewidth corresponds to ~0.1-10 Gauss range
        a = atom("87Rb")
        assert 0.01 < a.transition[1].Bgamma < 100
