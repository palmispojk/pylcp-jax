"""
Rate equation solver for multi-level laser cooling.

Constructs and solves the optical pumping rate equations from the given laser
beams, magnetic field, and block-diagonal Hamiltonian.  Supports equilibrium
population finding, force-profile generation (JAX-vectorised for diagonal
Hamiltonians), and time-dependent trajectory integration with optional
stochastic photon recoil via GPU-batched diffrax solvers.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from types import SimpleNamespace

import jax.numpy as jnp
from scipy.integrate import solve_ivp as scipy_solve_ivp
from scipy.interpolate import interp1d

from .common import (
    base_force_profile,
    progressBar,
)
from .governingeq import governingeq
from .integration_tools import solve_ivp_random as solve_ivp_random_cpu
from .integration_tools_gpu import solve_ivp_dense, solve_ivp_random


def abs2(x):
    """Return the squared magnitude of a complex number or array.

    Parameters
    ----------
    x : complex or array_like
        Input value(s).

    Returns
    -------
    abs2 : float or array_like
        ``x.real**2 + x.imag**2``.
    """
    return x.real**2 + x.imag**2


class force_profile(base_force_profile):
    """
    Rate equation force profile.

    The force profile object stores all of the calculated quantities created by
    the rateeq.generate_force_profile() method.  It has the following attributes:

    Attributes
    ----------
    R : array_like, shape (3, ...)
        Positions at which the force profile was calculated.
    V : array_like, shape (3, ...)
        Velocities at which the force profile was calculated.
    F : array_like, shape (3, ...)
        Total equilibrium force at position R and velocity V.
    f_mag : array_like, shape (3, ...)
        Magnetic force at position R and velocity V.
    f : dictionary of array_like
        The forces due to each laser, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp_jax.laserBeams object used to create the governing equation.
    Neq : array_like
        Equilibrium population found.
    Rijl : dictionary of array_like
        The pumping rates of each laser, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp_jax.laserBeams object used to create the governing equation.
    """

    def __init__(self, R, V, laserBeams, hamiltonian):
        super().__init__(R, V, laserBeams, hamiltonian)

        self.Rijl = {}
        for key in laserBeams:
            self.Rijl[key] = np.zeros(
                self.R[0].shape
                + (
                    len(laserBeams[key].beam_vector),
                    hamiltonian.blocks[hamiltonian.laser_keys[key]].n,
                    hamiltonian.blocks[hamiltonian.laser_keys[key]].m,
                )
            )

    def store_data(self, ind, Neq, F, F_laser, F_mag, Rijl):
        """Store force-profile results at a single grid index.

        Extends the base-class method by also storing the per-laser pumping
        rates ``Rijl``.

        Parameters
        ----------
        ind : tuple of int
            Multi-dimensional index into the profile arrays.
        Neq : array_like or None
            Equilibrium population vector.
        F : array_like, shape (3,)
            Total force.
        F_laser : dict of array_like
            Per-laser force contributions.
        F_mag : array_like, shape (3,)
            Magnetic force contribution.
        Rijl : dict of array_like
            Per-laser pumping rates, keyed by transition label.
        """
        super().store_data(ind, Neq, F, F_laser, F_mag)

        for key in Rijl:
            self.Rijl[key][ind] = Rijl[key]


class rateeq(governingeq):
    """
    The rate equations.

    This class constructs the rate equations from the given laser
    beams, magnetic field, and hamiltonian.

    Parameters
    ----------
    laserBeams : dictionary of pylcp_jax.laserBeams, pylcp_jax.laserBeams, or list of pylcp_jax.laserBeam
        The laserBeams that will be used in constructing the optical Bloch
        equations, addressing transitions in the block diagonal hamiltonian.  It can
        be any of the following:

            * A dictionary of pylcp_jax.laserBeams: if this is the case, the keys of
              the dictionary should match available :math:`d^{nm}` matrices
              in the pylcp_jax.hamiltonian object.  The key structure should be
              `n->m`.
            * pylcp_jax.laserBeams: a single set of laser beams is assumed to
              address the transition `g->e`.
            * a list of pylcp_jax.laserBeam: automatically promoted to a
              pylcp_jax.laserBeams object assumed to address the transition `g->e`.
    magField : pylcp_jax.magField or callable
        The function or object that defines the magnetic field.
    hamiltonian : pylcp_jax.hamiltonian
        The internal hamiltonian of the particle.
    a : array_like, shape (3,), optional
        A default acceleration to apply to the particle's motion, usually
        gravity. Default: [0., 0., 0.]
    include_mag_forces : boolean
        Optional flag to include magnetic forces in the force calculation.
        Default: True
    r0 : array_like, shape (3,), optional
        Initial position.  Default: [0., 0., 0.]
    v0 : array_like, shape (3,), optional
        Initial velocity.  Default: [0., 0., 0.]
    """

    def __init__(
        self,
        laserBeams,
        magField,
        hamiltonian,
        a=np.array([0.0, 0.0, 0.0]),
        include_mag_forces=True,
        svd_eps=1e-10,
        r0=np.array([0.0, 0.0, 0.0]),
        v0=np.array([0.0, 0.0, 0.0]),
    ):
        super().__init__(laserBeams, magField, hamiltonian, a=a, r0=r0, v0=v0)

        self.include_mag_forces = include_mag_forces
        self.svd_eps = svd_eps

        self.tdepend = {}
        self.tdepend["B"] = False

        self.decay_rates = {}
        self.decay_N_indices = {}

        if np.all(self.hamiltonian.diagonal):
            self._calc_decay_comp_of_Rev(self.hamiltonian)

        self.recoil_velocity = {}
        for key in self.hamiltonian.laser_keys:
            self.recoil_velocity[key] = (
                self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters["k"]
                / self.hamiltonian.mass
            )

        self.Rijl = {}
        self.profile = {}

        # Lazy-initialized JAX components (for diagonal hamiltonians)
        self._jax_components = None

    def _calc_decay_comp_of_Rev(self, rotated_ham):
        """
        Construct the decay portion of the evolution matrix.

        Parameters
        ----------
        rotated_ham: pylcp_jax.hamiltonian object
            The diagonalized hamiltonian with rotated d_q matrices
        """
        self.Rev_decay = jnp.zeros((self.hamiltonian.n, self.hamiltonian.n), dtype=jnp.float64)

        for key in self.hamiltonian.laser_keys:
            ind = rotated_ham.laser_keys[key]
            d_q_block = rotated_ham.blocks[ind]

            noff = int(np.sum(rotated_ham.ns[: ind[0]]))
            moff = int(np.sum(rotated_ham.ns[: ind[1]]))

            n = rotated_ham.ns[ind[0]]
            m = rotated_ham.ns[ind[1]]

            gamma = d_q_block.parameters["gamma"]

            self.decay_rates[key] = gamma * np.sum(abs2(d_q_block.matrix[:, :, :]), axis=(0, 1))

            self.decay_N_indices[key] = np.arange(moff, moff + m)

            m_idx = jnp.arange(moff, moff + m)
            self.Rev_decay = self.Rev_decay.at[m_idx, m_idx].add(
                -jnp.array(self.decay_rates[key], dtype=jnp.float64)
            )

            self.Rev_decay = self.Rev_decay.at[noff : noff + n, moff : moff + m].add(
                jnp.array(
                    gamma * np.sum(abs2(d_q_block.matrix), axis=0),
                    dtype=jnp.float64,
                )
            )

        return self.Rev_decay

    def _calc_pumping_rates(self, r, v, t, Bhat):
        """
        Compute optical pumping rates for each laser beam.

        Calculates R_{ij,l} at the given position, velocity, and time.
        Stores results in ``self.Rijl``.
        """
        for key in self.laserBeams:
            ind = self.hamiltonian.rotated_hamiltonian.laser_keys[key]
            d_q = self.hamiltonian.rotated_hamiltonian.blocks[ind].matrix
            gamma = self.hamiltonian.blocks[ind].parameters["gamma"]

            E1 = jnp.real(
                jnp.array(
                    np.diag(self.hamiltonian.rotated_hamiltonian.blocks[ind[0], ind[0]].matrix),
                    dtype=jnp.complex128,
                )
            )
            E2 = jnp.real(
                jnp.array(
                    np.diag(self.hamiltonian.rotated_hamiltonian.blocks[ind[1], ind[1]].matrix),
                    dtype=jnp.complex128,
                )
            )

            E2, E1 = jnp.meshgrid(E2, E1)

            self.Rijl[key] = jnp.zeros((len(self.laserBeams[key].beam_vector),) + d_q.shape[1:])

            kvecs = self.laserBeams[key].kvec(r, t)
            intensities = self.laserBeams[key].intensity(r, t)
            deltas = self.laserBeams[key].delta(t)
            projs = self.laserBeams[key].project_pol(Bhat, R=r, t=t)

            Rijl_list = []
            for ll, (kvec, intensity, proj, delta) in enumerate(
                zip(kvecs, intensities, projs, deltas)
            ):
                fijq = jnp.abs(d_q[0] * proj[2] + d_q[1] * proj[1] + d_q[2] * proj[0]) ** 2
                Rijl_list.append(
                    jnp.real(
                        gamma
                        * intensity
                        / 2
                        * fijq
                        / (1 + 4 * (-(E2 - E1) + delta - jnp.dot(kvec, v)) ** 2 / gamma**2)
                    )
                )

            self.Rijl[key] = jnp.stack(Rijl_list, axis=0)

    def _add_pumping_rates_to_Rev(self):
        for key in self.laserBeams:
            ind = self.hamiltonian.rotated_hamiltonian.laser_keys[key]

            n_off = sum(self.hamiltonian.rotated_hamiltonian.ns[: ind[0]])
            n = self.hamiltonian.rotated_hamiltonian.ns[ind[0]]
            m_off = sum(self.hamiltonian.rotated_hamiltonian.ns[: ind[1]])
            m = self.hamiltonian.rotated_hamiltonian.ns[ind[1]]

            Rij = jnp.sum(self.Rijl[key], axis=0)

            n_idx = jnp.arange(n_off, n_off + n)
            m_idx = jnp.arange(m_off, m_off + m)

            self.Rev = self.Rev.at[n_off : n_off + n, m_off : m_off + m].add(Rij)
            self.Rev = self.Rev.at[m_off : m_off + m, n_off : n_off + n].add(Rij.T)
            self.Rev = self.Rev.at[n_idx, n_idx].add(-jnp.sum(Rij, axis=1))
            self.Rev = self.Rev.at[m_idx, m_idx].add(-jnp.sum(Rij, axis=0))

    def construct_evolution_matrix(self, r, v, t=0.0, default_axis=jnp.array([0.0, 0.0, 1.0])):
        """
        Construct the evolution matrix at a given position and time.

        Parameters
        ----------
        r : array_like, shape (3,)
            Position at which to calculate the equilibrium population
        v : array_like, shape (3,)
            Velocity at which to calculate the equilibrium population
        t : float
            Time at which to calculate the equilibrium population
        """
        if self.tdepend["B"]:
            B = self.magField.Field(r, t)
        else:
            B = self.magField.Field(r)

        Bmag = jnp.linalg.norm(B)

        if float(Bmag) > 1e-10:
            Bhat = B / Bmag
        else:
            Bhat = default_axis

        # diag_static_field requires a Python float.
        # It caches internally when B hasn't changed, so this is cheap on
        # repeated calls with the same field magnitude.
        _Bmag_float = float(Bmag)
        self.hamiltonian.diag_static_field(_Bmag_float)

        self.Rev = jnp.zeros((self.hamiltonian.n, self.hamiltonian.n), dtype=jnp.float64)

        # _calc_decay_comp_of_Rev only depends on the rotated Hamiltonian
        # (i.e. on B), so skip it when B hasn't changed.
        if not np.all(self.hamiltonian.diagonal):
            if not hasattr(self, "_last_Rev_decay_B") or self._last_Rev_decay_B != _Bmag_float:
                self._calc_decay_comp_of_Rev(self.hamiltonian.rotated_hamiltonian)
                self._last_Rev_decay_B = _Bmag_float

        self.Rev = self.Rev + self.Rev_decay

        self._calc_pumping_rates(r, v, t, Bhat)

        self._add_pumping_rates_to_Rev()

        return self.Rev, self.Rijl

    def equilibrium_populations(self, r, v, t, **kwargs):
        """
        Return the equilibrium population as determined by the rate equations.

        Parameters
        ----------
        r : array_like, shape (3,)
            Position
        v : array_like, shape (3,)
            Velocity
        t : float
            Time
        return_details : boolean, optional
            If True, also return Rev and Rijl.

        Returns
        -------
        Neq : array_like
            Equilibrium population vector
        Rev : array_like (if return_details)
        Rijl : dict (if return_details)
        """
        return_details = kwargs.pop("return_details", False)

        Rev, Rijl = self.construct_evolution_matrix(r, v, t, **kwargs)

        # SVD null-space via JAX
        U, S, VH = jnp.linalg.svd(Rev)

        # Find rows where singular value is near zero
        null_mask = S <= self.svd_eps
        n_null = int(jnp.sum(null_mask))

        if n_null == 0:
            # Fall back: take the row corresponding to the smallest S
            Neq = jnp.abs(VH[-1])
        elif n_null == 1:
            Neq = jnp.abs(VH[jnp.argmin(S)])
        else:
            # More than one null vector → degenerate, return NaN
            Neq = jnp.full(self.hamiltonian.n, jnp.nan)
            if return_details:
                return Neq, Rev, Rijl
            return Neq

        Neq = Neq / jnp.sum(Neq)

        if return_details:
            return Neq, Rev, Rijl
        return Neq

    def force(self, r, t, N, return_details=True):
        """
        Calculate the instantaneous force.

        Parameters
        ----------
        r : array_like
            Position
        t : float
            Time
        N : array_like
            Relative state populations
        return_details : boolean, optional
            If True, also return per-laser and magnetic forces.
        """
        F = jnp.zeros(3, dtype=jnp.float64)
        f = {}

        for key in self.laserBeams:
            f[key] = jnp.zeros((3, len(self.laserBeams[key].beam_vector)), dtype=jnp.float64)

            ind = self.hamiltonian.laser_keys[key]
            n_off = sum(self.hamiltonian.ns[: ind[0]])
            n = self.hamiltonian.ns[ind[0]]
            m_off = sum(self.hamiltonian.ns[: ind[1]])
            m = self.hamiltonian.ns[ind[1]]

            Ne = N[m_off : (m_off + m)]
            Ng = N[n_off : (n_off + n)]
            diff_NN = Ng[:, None] - Ne[None, :]  # (n, m)

            for ll, beam in enumerate(self.laserBeams[key].beam_vector):
                kvec = jnp.array(beam.kvec(r, t))
                scatter = jnp.sum(self.Rijl[key][ll] * diff_NN)
                f[key] = f[key].at[:, ll].add(kvec * scatter)

            F = F + jnp.sum(f[key], axis=1)

        fmag = jnp.zeros(3, dtype=jnp.float64)
        if self.include_mag_forces:
            gradBmag = self.magField.gradFieldMag(r)

            for ii, block in enumerate(np.diag(self.hamiltonian.blocks)):
                ind1 = int(np.sum(self.hamiltonian.ns[:ii]))
                ind2 = int(np.sum(self.hamiltonian.ns[: ii + 1]))
                if self.hamiltonian.diagonal[ii]:
                    if isinstance(block, tuple):
                        fmag = (
                            fmag
                            + jnp.sum(jnp.real(jnp.array(block[1].matrix[1]) @ N[ind1:ind2]))
                            * gradBmag
                        )
                    elif isinstance(block, self.hamiltonian.vector_block):
                        fmag = (
                            fmag
                            + jnp.sum(jnp.real(jnp.array(block.matrix[1]) @ N[ind1:ind2]))
                            * gradBmag
                        )
                else:
                    if isinstance(block, tuple):
                        fmag = (
                            fmag
                            + jnp.sum(
                                jnp.real(
                                    jnp.array(
                                        self.hamiltonian.U[ii].T
                                        @ block[1].matrix[1]
                                        @ self.hamiltonian.U[ii]
                                    )
                                    @ N[ind1:ind2]
                                )
                            )
                            * gradBmag
                        )
                    elif isinstance(block, self.hamiltonian.vector_block):
                        fmag = (
                            fmag
                            + jnp.sum(
                                jnp.real(
                                    jnp.array(
                                        self.hamiltonian.U[ii].T
                                        @ block.matrix[1]
                                        @ self.hamiltonian.U[ii]
                                    )
                                    @ N[ind1:ind2]
                                )
                            )
                            * gradBmag
                        )

            F = F + fmag

        if return_details:
            return F, f, fmag
        return F

    def set_initial_pop(self, N0):
        """
        Set the initial populations.

        Parameters
        ----------
        N0 : array_like
            The initial state population vector.
        """
        if len(N0) != self.hamiltonian.n:
            raise ValueError(
                "Npop has only %d entries for %d states." % (len(N0), self.hamiltonian.n)
            )
        N0 = jnp.asarray(N0, dtype=jnp.float64)
        if jnp.any(jnp.isnan(N0)) or jnp.any(jnp.isinf(N0)):
            raise ValueError("Npop has NaNs or Infs!")
        self.N0 = N0

    def set_initial_pop_from_equilibrium(self):
        """Set the initial populations based on the equilibrium at r0, v0, t=0."""
        self.N0 = self.equilibrium_populations(self.r0, self.v0, t=0.0)

    # ------------------------------------------------------------------
    # JAX component setup (for diagonal hamiltonians)
    # ------------------------------------------------------------------

    def _get_jax_components(self):
        """Precompute JAX arrays for Rev and force (diagonal Hamiltonians only)."""
        if self._jax_components is not None:
            return self._jax_components

        if not np.all(self.hamiltonian.diagonal):
            raise ValueError("_get_jax_components requires np.all(self.hamiltonian.diagonal)")

        self.hamiltonian.diag_static_field(0.0)

        Rev_decay_jax = jnp.array(self.Rev_decay, dtype=jnp.float64)

        # Per-key data for building the pumping-rate contribution to Rev.
        pump_data = {}
        for key in self.laserBeams:
            ind = self.hamiltonian.rotated_hamiltonian.laser_keys[key]
            d_q = jnp.array(self.hamiltonian.rotated_hamiltonian.blocks[ind].matrix)
            gamma = float(self.hamiltonian.rotated_hamiltonian.blocks[ind].parameters["gamma"])
            E1_d = jnp.real(
                jnp.array(
                    np.diag(self.hamiltonian.rotated_hamiltonian.blocks[ind[0], ind[0]].matrix),
                    dtype=jnp.complex128,
                )
            )
            E2_d = jnp.real(
                jnp.array(
                    np.diag(self.hamiltonian.rotated_hamiltonian.blocks[ind[1], ind[1]].matrix),
                    dtype=jnp.complex128,
                )
            )
            E2g, E1g = jnp.meshgrid(E2_d, E1_d)  # (n, m) each, float64
            n_off = int(sum(self.hamiltonian.rotated_hamiltonian.ns[: ind[0]]))
            n = int(self.hamiltonian.rotated_hamiltonian.ns[ind[0]])
            m_off = int(sum(self.hamiltonian.rotated_hamiltonian.ns[: ind[1]]))
            m = int(self.hamiltonian.rotated_hamiltonian.ns[ind[1]])

            # Precompute magnetic moment diagonals for Zeeman shift.
            def _mu_z_diag(block_idx, size):
                blk = np.diag(self.hamiltonian.blocks)[block_idx]
                if isinstance(blk, tuple):
                    return jnp.array(np.real(np.diag(blk[1].matrix[1])), dtype=jnp.float64)
                elif isinstance(blk, self.hamiltonian.vector_block):
                    return jnp.array(
                        np.real(np.diag(blk.matrix[1])),  # pyright: ignore[reportAttributeAccessIssue]
                        dtype=jnp.float64,
                    )
                return jnp.zeros(size, dtype=jnp.float64)

            mu_z_1 = _mu_z_diag(ind[0], n)
            mu_z_2 = _mu_z_diag(ind[1], m)
            mu_z_2g, mu_z_1g = jnp.meshgrid(mu_z_2, mu_z_1)
            mu_diff = mu_z_2g - mu_z_1g  # (n, m)

            pump_data[key] = dict(
                d_q=d_q,
                gamma=gamma,
                E2g=E2g,
                E1g=E1g,
                mu_diff=mu_diff,
                n_off=n_off,
                n=n,
                m_off=m_off,
                m=m,
            )

        # Per-key offsets for the force calculation (original hamiltonian).
        force_data = {}
        for key in self.laserBeams:
            ind = self.hamiltonian.laser_keys[key]
            n_off = int(sum(self.hamiltonian.ns[: ind[0]]))
            n = int(self.hamiltonian.ns[ind[0]])
            m_off = int(sum(self.hamiltonian.ns[: ind[1]]))
            m = int(self.hamiltonian.ns[ind[1]])
            force_data[key] = dict(n_off=n_off, n=n, m_off=m_off, m=m)

        # Precomputed magnetic-moment matrices for each diagonal block.
        mag_mats = []
        if self.include_mag_forces:
            for ii, block in enumerate(np.diag(self.hamiltonian.blocks)):
                ind1 = int(np.sum(self.hamiltonian.ns[:ii]))
                ind2 = int(np.sum(self.hamiltonian.ns[: ii + 1]))
                mat = None
                if self.hamiltonian.diagonal[ii]:
                    if isinstance(block, tuple):
                        mat = jnp.array(np.real(block[1].matrix[1]))
                    elif isinstance(block, self.hamiltonian.vector_block):
                        mat = jnp.array(np.real(block.matrix[1]))
                else:
                    if isinstance(block, tuple):
                        mat = jnp.array(
                            np.real(
                                self.hamiltonian.U[ii].T
                                @ block[1].matrix[1]
                                @ self.hamiltonian.U[ii]
                            )
                        )
                    elif isinstance(block, self.hamiltonian.vector_block):
                        mat = jnp.array(
                            np.real(
                                self.hamiltonian.U[ii].T @ block.matrix[1] @ self.hamiltonian.U[ii]
                            )
                        )
                mag_mats.append((ind1, ind2, mat))

        self._jax_components = (Rev_decay_jax, pump_data, force_data, mag_mats)
        return self._jax_components

    def _make_jax_rhs(self, t_val=0.0, free_axes=None):
        """Build JAX-traceable RHS for the full motion ODE (N + v + r).

        Only valid for diagonal Hamiltonians.
        """
        Rev_decay_jax, pump_data, force_data, mag_mats = self._get_jax_components()
        if free_axes is None:
            free_axes = jnp.ones(3, dtype=jnp.float64)

        n_states = self.hamiltonian.n
        mass = float(self.hamiltonian.mass)
        accel = self.constant_accel

        def rhs(t, y, _args):
            """JAX-traceable ODE RHS for populations + velocity + position."""
            N = y[:n_states]
            v = y[n_states : n_states + 3]
            r = y[n_states + 3 :]

            B = self.magField.Field(r, t)
            Bmag = jnp.linalg.norm(B)
            Bhat = jnp.where(
                Bmag > 1e-10,
                B / Bmag,
                jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
            )

            Rev = Rev_decay_jax
            Rijl_store = {}
            kvec_store = {}

            for key, kd in pump_data.items():
                d_q, gamma = kd["d_q"], kd["gamma"]
                E2g, E1g = kd["E2g"], kd["E1g"]
                mu_diff = kd["mu_diff"]
                n_off, n_k, m_off, m_k = (
                    kd["n_off"],
                    kd["n"],
                    kd["m_off"],
                    kd["m"],
                )

                kvecs = self.laserBeams[key].kvec(r, t)
                intensities = self.laserBeams[key].intensity(r, t)
                deltas = self.laserBeams[key].delta(t)
                projs = self.laserBeams[key].project_pol(Bhat, R=r, t=t)

                kvec_store[key] = kvecs

                # Zeeman shift: E(B) = E(0) - B*mu_z, so
                # -(E2(B)-E1(B)) = -(E2g-E1g) + Bmag*mu_diff
                zeeman = Bmag * mu_diff

                def beam_Rij(kvec, intensity, proj, delta):
                    """Compute the pumping rate matrix for one laser beam."""
                    fijq = jnp.abs(d_q[0] * proj[2] + d_q[1] * proj[1] + d_q[2] * proj[0]) ** 2
                    return (
                        gamma
                        * intensity
                        / 2
                        * fijq
                        / (
                            1
                            + 4 * (-(E2g - E1g) + zeeman + delta - jnp.dot(kvec, v)) ** 2 / gamma**2
                        )
                    )

                Rijl = jax.vmap(beam_Rij)(kvecs, intensities, projs, deltas)
                Rijl_store[key] = Rijl
                Rij = jnp.sum(Rijl, axis=0)

                n_idx = jnp.arange(n_off, n_off + n_k)
                m_idx = jnp.arange(m_off, m_off + m_k)
                Rev = Rev.at[n_off : n_off + n_k, m_off : m_off + m_k].add(Rij)
                Rev = Rev.at[m_off : m_off + m_k, n_off : n_off + n_k].add(Rij.T)
                Rev = Rev.at[n_idx, n_idx].add(-jnp.sum(Rij, axis=1))
                Rev = Rev.at[m_idx, m_idx].add(-jnp.sum(Rij, axis=0))

            F = jnp.zeros(3, dtype=jnp.float64)
            for key, kd in force_data.items():
                n_off, n_k, m_off, m_k = (
                    kd["n_off"],
                    kd["n"],
                    kd["m_off"],
                    kd["m"],
                )
                Ng = N[n_off : n_off + n_k]
                Ne = N[m_off : m_off + m_k]
                diff_NN = Ng[:, None] - Ne[None, :]
                Rijl = Rijl_store[key]
                kvecs = kvec_store[key]
                scatter = jnp.sum(Rijl * diff_NN[None], axis=(1, 2))
                f_key = (kvecs * scatter[:, None]).T
                F = F + jnp.sum(f_key, axis=1)

            fmag = jnp.zeros(3, dtype=jnp.float64)
            if self.include_mag_forces:
                gradBmag = self.magField.gradFieldMag(r)
                gradBmag = jnp.nan_to_num(gradBmag, nan=0.0)
                for ind1, ind2, mat in mag_mats:
                    if mat is not None:
                        fmag = fmag + jnp.sum(mat @ N[ind1:ind2]) * gradBmag
                F = F + fmag

            dN = Rev @ N
            dv = F * free_axes / mass + accel
            dr = v
            return jnp.concatenate([dN, dv, dr])

        return rhs

    # ------------------------------------------------------------------
    # ODE integration methods
    # ------------------------------------------------------------------

    def evolve_populations(
        self,
        t_span,
        n_points=1001,
        rtol=1e-5,
        atol=1e-5,
        solver_type="Dopri5",
        **kwargs,
    ):
        """
        Evolve the state population in time via diffrax (GPU-accelerated).

        Parameters
        ----------
        t_span : list or array_like
            [t_start, t_end] integration interval.
        n_points : int, optional
            Number of equally-spaced output time points.  Default: 1001.
        rtol, atol : float, optional
            Tolerances for the adaptive step-size controller.
        solver_type : str, optional
            'Dopri5' (default) or 'Bosh3'.

        Returns
        -------
        sol : SimpleNamespace
            .t  : time array (n_points,)
            .y  : populations (n_states, n_points)
        """
        if any([self.tdepend[key] for key in self.tdepend.keys()]):
            raise NotImplementedError("Time dependence not yet implemented.")

        Rev, _ = self.construct_evolution_matrix(self.r0, self.v0)
        Rev_jax = jnp.array(Rev, dtype=jnp.float64)

        ts, ys = solve_ivp_dense(
            lambda t, N: Rev_jax @ N,
            t_span,
            jnp.array(self.N0)[None, :],
            n_points=n_points,
            rtol=rtol,
            atol=atol,
            solver_type=solver_type,
        )
        # ys: (1, n_points, n_states) → sol.y: (n_states, n_points)
        sol = SimpleNamespace()
        sol.t = np.array(ts)
        sol.y = np.array(ys[0].T)
        self.sol = sol
        return self.sol

    def evolve_motion(
        self,
        t_span,
        n_points,
        freeze_axis=[False, False, False],
        random_recoil=False,
        random_force=False,
        max_scatter_probability=0.1,
        progress_bar=False,
        record_force=False,
        key=None,
        rtol=1e-5,
        atol=1e-5,
        solver_type="Dopri5",
        max_steps=100000,
        **kwargs,
    ):
        """
        Evolve the populations and motion of the atom in time.

        For diagonal hamiltonians this uses JAX/diffrax for GPU acceleration.
        For non-diagonal hamiltonians it falls back to the CPU solver.

        Parameters
        ----------
        t_span : list or array_like
            [t_start, t_end] integration interval.
        freeze_axis : list of boolean
            Freeze atomic motion along specified axes.  Default: [False,False,False]
        random_recoil : boolean
            Allow the atom to randomly recoil from spontaneous emission.
            Default: False
        random_force : boolean
            Randomly apply absorption recoil from each laser.  Default: False
        max_scatter_probability : float
            Maximum scattering probability per time step for random methods.
            Default: 0.1
        progress_bar : boolean
            Show progress bar (CPU path only).  Default: False
        record_force : boolean
            Record instantaneous force (CPU path only).  Default: False
        key : jax.random.PRNGKey, optional
            JAX PRNG key required when random_recoil or random_force is True
            and using the JAX path.
        n_points : int
            Number of evenly-spaced output time points.
        rtol, atol : float, optional
            ODE solver tolerances.  Default: 1e-5.
        solver_type : str, optional
            'Dopri5' or 'Bosh3'.  Default: 'Dopri5'.
        max_steps : int, optional
            Max steps for random solver.  Default: 100000.

        Returns
        -------
        sol : SimpleNamespace or RandomOdeResult
            .t  : time array
            .N  : population array (n_states, n_steps)
            .v  : velocity array   (3, n_steps)
            .r  : position array   (3, n_steps)
        """
        free_axes = jnp.array([not ax for ax in freeze_axis], dtype=jnp.float64)

        if not np.all(self.hamiltonian.diagonal):
            # ----------------------------------------------------------
            # CPU fallback for non-diagonal hamiltonians
            # ----------------------------------------------------------
            return self._evolve_motion_cpu(
                t_span,
                freeze_axis=freeze_axis,
                random_recoil=random_recoil,
                random_force=random_force,
                max_scatter_probability=max_scatter_probability,
                progress_bar=progress_bar,
                record_force=record_force,
                **kwargs,
            )

        # ------------------------------------------------------------------
        # JAX path (diagonal hamiltonian)
        # ------------------------------------------------------------------
        rhs = self._make_jax_rhs(free_axes=free_axes)
        n_states = self.hamiltonian.n

        y0 = jnp.concatenate(
            [
                jnp.array(self.N0, dtype=jnp.float64),
                jnp.array(self.v0, dtype=jnp.float64),
                jnp.array(self.r0, dtype=jnp.float64),
            ]
        )

        if not random_recoil and not random_force:
            # Dense output via diffrax vmap solver
            ts, ys = solve_ivp_dense(
                rhs,
                t_span,
                y0[None, :],
                n_points=n_points,
                rtol=rtol,
                atol=atol,
                solver_type=solver_type,
            )
            sol = SimpleNamespace()
            sol.t = np.array(ts)
            _y = np.array(ys[0].T)  # (state_dim, n_points)
            sol.y = _y
            sol.N = _y[:n_states]
            sol.v = _y[n_states : n_states + 3]
            sol.r = _y[n_states + 3 :]
            self.sol = sol
            return self.sol

        # Random force / recoil path
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(0, 2**31))

        Rev_decay_jax, pump_data, force_data, mag_mats = self._get_jax_components()
        mass = float(self.hamiltonian.mass)

        if random_force:
            random_func = self._make_random_force_func(
                pump_data, n_states, mass, free_axes, max_scatter_probability
            )
        else:
            random_func = self._make_random_recoil_func(
                n_states, free_axes, max_scatter_probability
            )

        keys_batch = jnp.array(key)[None]

        results = solve_ivp_random(
            rhs,
            random_func,
            t_span,
            y0[None, :],
            keys_batch,
            n_points=n_points,
            solver_type=solver_type,
            max_steps=max_steps,
            max_step=max_scatter_probability,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

        raw = results[0]
        sol = SimpleNamespace()
        sol.t = np.array(raw.t)
        _y = np.array(raw.y)  # (state_dim, n_steps)
        sol.y = _y
        sol.N = _y[:n_states]
        sol.v = _y[n_states : n_states + 3]
        sol.r = _y[n_states + 3 :]
        sol.t_random = np.array(raw.t_random)
        sol.n_random = np.array(raw.n_random)
        sol.inds_random = np.array(raw.inds_random)
        self.sol = sol
        return self.sol

    def _make_random_force_func(self, pump_data, n_states, mass, free_axes, max_P):
        """Build JAX random-force function for solve_ivp_random."""
        # Pre-count total number of laser beams across all keys
        n_beams_per_key = {k: len(self.laserBeams[k].beam_vector) for k in pump_data}

        def random_force_func(t, y, dt, key, args=None):
            """Apply stochastic absorption and emission recoil kicks per laser beam."""
            v = y[n_states : n_states + 3]
            r = y[n_states + 3 :]

            B = self.magField.Field(r, t)
            Bmag = jnp.linalg.norm(B)
            Bhat = jnp.where(
                Bmag > 1e-10,
                B / Bmag,
                jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
            )

            total_n_beams = sum(n_beams_per_key[k] for k in pump_data)
            # 2 keys per beam (dice + direction), plus 1 for the new key
            all_keys = jax.random.split(key, 2 * total_n_beams + 1)
            key_new = all_keys[0]
            ki = 1

            total_P = jnp.zeros((), dtype=jnp.float64)
            num_scatters = jnp.zeros((), dtype=jnp.int32)
            y_out = y

            for beam_key, kd in pump_data.items():
                d_q, gamma = kd["d_q"], kd["gamma"]
                E2g, E1g = kd["E2g"], kd["E1g"]

                kvecs = self.laserBeams[beam_key].kvec(r, t)
                intensities = self.laserBeams[beam_key].intensity(r, t)
                deltas = self.laserBeams[beam_key].delta(t)
                projs = self.laserBeams[beam_key].project_pol(Bhat, R=r, t=t)

                def _beam_Rij(kvec, intensity, proj, delta):
                    """Compute pumping rate matrix for one beam (random force path)."""
                    fijq = jnp.abs(d_q[0] * proj[2] + d_q[1] * proj[1] + d_q[2] * proj[0]) ** 2
                    return (
                        gamma
                        * intensity
                        / 2
                        * fijq
                        / (1 + 4 * (-(E2g - E1g) + delta - jnp.dot(kvec, v)) ** 2 / gamma**2)
                    )

                Rijl = jax.vmap(_beam_Rij)(kvecs, intensities, projs, deltas)
                Rl = jnp.sum(Rijl, axis=(1, 2))  # (n_beams,)
                P_l = Rl * dt

                for ll in range(n_beams_per_key[beam_key]):
                    dice = jax.random.uniform(all_keys[ki])
                    raw_dir = jax.random.normal(all_keys[ki + 1], shape=(3,)) * free_axes
                    ki += 2

                    scattered = dice < P_l[ll]
                    kick_abs = kvecs[ll] / mass
                    norm_d = jnp.linalg.norm(raw_dir)
                    rand_unit = raw_dir / jnp.where(norm_d > 0, norm_d, 1.0)
                    kick_emit = self.recoil_velocity[beam_key] * rand_unit

                    delta_v = jnp.where(
                        scattered,
                        (kick_abs + kick_emit) * free_axes,
                        jnp.zeros(3, dtype=jnp.float64),
                    )
                    y_out = y_out.at[n_states : n_states + 3].add(delta_v)
                    num_scatters = num_scatters + jnp.where(scattered, jnp.int32(1), jnp.int32(0))
                    total_P = total_P + P_l[ll]

            dt_max = jnp.where(total_P > 0, max_P / total_P * dt, jnp.float64(dt))
            return y_out, num_scatters, dt_max, key_new

        return random_force_func

    def _make_random_recoil_func(self, n_states, free_axes, max_P):
        """Build JAX random-recoil function for solve_ivp_random."""
        decay_rates_jax = {
            k: jnp.array(self.decay_rates[k], dtype=jnp.float64) for k in self.decay_rates
        }
        decay_indices = {k: jnp.array(self.decay_N_indices[k]) for k in self.decay_N_indices}
        # Count excited states per key
        n_excited_per_key = {k: len(self.decay_N_indices[k]) for k in self.decay_rates}

        def random_recoil_func(t, y, dt, key, args=None):
            """Apply stochastic spontaneous-emission recoil kicks from excited-state decay."""
            N = y[:n_states]

            total_n_excited = sum(n_excited_per_key[k] for k in decay_rates_jax)
            all_keys = jax.random.split(key, 3 * total_n_excited + 1)
            key_new = all_keys[0]
            ki = 1

            total_P = jnp.zeros((), dtype=jnp.float64)
            num_scatters = jnp.zeros((), dtype=jnp.int32)
            y_out = y

            def _rand_unit(subkey):
                raw = jax.random.normal(subkey, shape=(3,)) * free_axes
                norm = jnp.linalg.norm(raw)
                return raw / jnp.where(norm > 0, norm, 1.0)

            for recoil_key in decay_rates_jax:
                rates = decay_rates_jax[recoil_key]  # (n_exc,)
                indices = decay_indices[recoil_key]  # (n_exc,)
                P = dt * rates * N[indices]  # (n_exc,)

                for ii in range(n_excited_per_key[recoil_key]):
                    dice = jax.random.uniform(all_keys[ki])
                    vec1 = _rand_unit(all_keys[ki + 1])
                    vec2 = _rand_unit(all_keys[ki + 2])
                    ki += 3

                    scattered = dice < P[ii]
                    kick = self.recoil_velocity[recoil_key] * (vec1 + vec2)

                    delta_v = jnp.where(
                        scattered,
                        kick * free_axes,
                        jnp.zeros(3, dtype=jnp.float64),
                    )
                    y_out = y_out.at[n_states : n_states + 3].add(delta_v)
                    num_scatters = num_scatters + jnp.where(scattered, jnp.int32(1), jnp.int32(0))
                    total_P = total_P + P[ii]

            dt_max = jnp.where(total_P > 0, max_P / total_P * dt, jnp.float64(dt))
            return y_out, num_scatters, dt_max, key_new

        return random_recoil_func

    def _evolve_motion_cpu(
        self,
        t_span,
        freeze_axis=[False, False, False],
        random_recoil=False,
        random_force=False,
        max_scatter_probability=0.1,
        progress_bar=False,
        record_force=False,
        rng=None,
        **kwargs,
    ):
        """CPU fallback for evolve_motion (non-diagonal hamiltonians)."""
        import numpy as _np

        if rng is None:
            rng = _np.random.default_rng()

        free_axes = _np.bitwise_not(freeze_axis)

        if progress_bar:
            progress = progressBar()

        if record_force:
            ts_rec = []
            Fs_rec = []

        def motion(t, y):
            """ODE RHS for populations + velocity + position (CPU/scipy path)."""
            N = y[:-6]
            v = y[-6:-3]
            r = y[-3:]

            Rev, Rijl = self.construct_evolution_matrix(r, v, t)
            Rev_np = _np.array(Rev)

            if not random_force:
                if record_force:
                    F_out = self.force(r, t, jnp.array(N), return_details=True)
                    ts_rec.append(t)
                    Fs_rec.append(F_out)
                    F_arr = _np.array(F_out[0])
                else:
                    F_arr = _np.array(self.force(r, t, jnp.array(N), return_details=False))
                dydt = _np.concatenate(
                    (
                        Rev_np @ N,
                        F_arr * free_axes / self.hamiltonian.mass + _np.array(self.constant_accel),
                        v,
                    )
                )
            else:
                dydt = _np.concatenate((Rev_np @ N, _np.array(self.constant_accel), v))

            if _np.any(_np.isnan(dydt)):
                raise ValueError("Encountered a NaN!")

            if progress_bar:
                progress.update(t / t_span[-1])

            return dydt

        def _rand_unit(axes):
            """Return a random unit vector restricted to the specified axes (CPU path)."""
            v = rng.standard_normal(3) * axes
            norm = _np.linalg.norm(v)
            return v / norm if norm > 0 else v

        def random_force_func_cpu(t, y, dt):
            """Apply stochastic absorption + emission recoil kicks per beam (CPU path)."""
            total_P = 0
            num_scatters = 0
            for key in self.laserBeams:
                Rl = _np.array(jnp.sum(self.Rijl[key], axis=(1, 2)))
                P = Rl * dt
                dice = rng.random(len(P))
                for ii in _np.arange(len(Rl))[dice < P]:
                    num_scatters += 1
                    y[-6:-3] += (
                        _np.array(self.laserBeams[key].beam_vector[ii].kvec(y[-3:], t))
                        / self.hamiltonian.mass
                    )
                    y[-6:-3] += self.recoil_velocity[key] * _rand_unit(free_axes)
                total_P += _np.sum(P)
            return (num_scatters, (max_scatter_probability / total_P) * dt)

        def random_recoil_func_cpu(t, y, dt):
            """Apply stochastic spontaneous-emission recoil kicks (CPU path)."""
            num_scatters = 0
            total_P = 0.0
            for key in self.decay_rates:
                P = dt * self.decay_rates[key] * y[self.decay_N_indices[key]]
                dice = rng.random(len(P))
                for _ in range(_np.sum(dice < P)):
                    num_scatters += 1
                    y[-6:-3] += self.recoil_velocity[key] * (
                        _rand_unit(free_axes) + _rand_unit(free_axes)
                    )
                total_P += _np.sum(P)
            return (num_scatters, (max_scatter_probability / total_P) * dt)

        y0 = _np.concatenate((_np.array(self.N0), _np.array(self.v0), _np.array(self.r0)))
        if random_force:
            self.sol = solve_ivp_random_cpu(
                motion,
                random_force_func_cpu,
                t_span,
                y0,
                initial_max_step=max_scatter_probability,
                **kwargs,
            )
        elif random_recoil:
            self.sol = solve_ivp_random_cpu(
                motion,
                random_recoil_func_cpu,
                t_span,
                y0,
                initial_max_step=max_scatter_probability,
                **kwargs,
            )
        else:
            self.sol = scipy_solve_ivp(motion, t_span, y0, **kwargs)

        if progress_bar:
            progress.update(1.0)

        self.sol.N = self.sol.y[:-6]
        self.sol.v = self.sol.y[-6:-3]
        self.sol.r = self.sol.y[-3:]

        if record_force and not random_force:
            f_interp = interp1d(ts_rec[:-1], _np.array([f[0] for f in Fs_rec[:-1]]).T)
            self.sol.F = f_interp(self.sol.t)

            f_interp = interp1d(ts_rec[:-1], _np.array([f[2] for f in Fs_rec[:-1]]).T)
            self.sol.fmag = f_interp(self.sol.t)

            self.sol.f = {}
            for key in Fs_rec[0][1]:
                f_interp = interp1d(ts_rec[:-1], _np.array([f[1][key] for f in Fs_rec[:-1]]).T)
                self.sol.f[key] = _np.swapaxes(f_interp(self.sol.t), 0, 1)

        del self.sol.y
        return self.sol

    # ------------------------------------------------------------------
    # Equilibrium force / profile
    # ------------------------------------------------------------------

    def find_equilibrium_force(self, return_details=False, **kwargs):
        """
        Find the equilibrium force at the initial position.

        Parameters
        ----------
        return_details : boolean, optional
            If True, also return per-laser forces, equilibrium populations,
            Rijl, and magnetic forces.  Default: False

        Returns
        -------
        F_eq : array_like
            Total equilibrium force.
        (f_eq, N_eq, Rijl, f_mag) : optional, if return_details is True.
        """
        if any([self.tdepend[key] for key in self.tdepend.keys()]):
            raise NotImplementedError("Time dependence not yet implemented.")

        N_eq, Rev, Rijl = self.equilibrium_populations(
            self.r0, self.v0, t=0.0, return_details=True, **kwargs
        )

        F_eq, f_eq, f_mag = self.force(self.r0, 0.0, N_eq)

        if return_details:
            return F_eq, f_eq, N_eq, Rijl, f_mag
        return F_eq

    def generate_force_profile(self, R, V, name=None, progress_bar=False, **kwargs):
        """
        Map out the equilibrium force vs. position and velocity.

        For diagonal hamiltonians this uses JAX/vmap to compute all grid
        points in a single batched GPU call.  For non-diagonal hamiltonians
        it falls back to a sequential CPU loop.

        Parameters
        ----------
        R : array_like, shape(3, ...)
            Position grid.
        V : array_like, shape(3, ...)
            Velocity grid.
        name : str, optional
            Key for self.profile.  Defaults to the next integer string.
        progress_bar : boolean, optional
            Show progress bar (CPU path only).  Default: False.
        t : float, optional (JAX path only)
            Time at which to evaluate the profile.  Default: 0.
        kwargs :
            Additional arguments passed to find_equilibrium_force (CPU path).

        Returns
        -------
        profile : pylcp_jax.rateeq.force_profile
        """
        if not name:
            name = "{0:d}".format(len(self.profile))

        self.profile[name] = force_profile(R, V, self.laserBeams, self.hamiltonian)

        if np.all(self.hamiltonian.diagonal):
            t = kwargs.pop("t", 0.0)
            self._generate_force_profile_jax(R, V, name, t=t)
        else:
            self._generate_force_profile_cpu(R, V, name, progress_bar, **kwargs)

        return self.profile[name]

    def _generate_force_profile_jax(self, R, V, name, t=0.0):
        """
        Compute the force profile using JAX-vectorized evaluation.

        Batches all (r, v) grid points into a single ``jax.vmap`` call so the
        whole computation runs on the GPU.
        """
        Rev_decay_jax, pump_data, force_data, mag_mats = self._get_jax_components()

        # ----------------------------------------------------------------
        # Single-point function (traced once, vmapped over all grid points)
        # ----------------------------------------------------------------
        def single_point(r, v):
            """Compute equilibrium populations and force at a single (r, v) grid point."""
            B = self.magField.Field(r, t)
            Bmag = jnp.linalg.norm(B)
            Bhat = jnp.where(
                Bmag > 1e-10,
                B / Bmag,
                jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
            )

            Rev = Rev_decay_jax
            Rijl_store = {}
            kvec_store = {}

            for key, kd in pump_data.items():
                d_q, gamma = kd["d_q"], kd["gamma"]
                E2g, E1g = kd["E2g"], kd["E1g"]
                mu_diff = kd["mu_diff"]
                n_off, n_k, m_off, m_k = (
                    kd["n_off"],
                    kd["n"],
                    kd["m_off"],
                    kd["m"],
                )

                kvecs = self.laserBeams[key].kvec(r, t)
                intensities = self.laserBeams[key].intensity(r, t)
                deltas = self.laserBeams[key].delta(t)
                projs = self.laserBeams[key].project_pol(Bhat, R=r, t=t)

                kvec_store[key] = kvecs

                # Zeeman shift: E(B) = E(0) - B*mu_z, so
                # -(E2(B)-E1(B)) = -(E2g-E1g) + Bmag*mu_diff
                zeeman = Bmag * mu_diff

                def beam_Rij(kvec, intensity, proj, delta):
                    """Compute pumping rate matrix for one beam at this grid point."""
                    fijq = jnp.abs(d_q[0] * proj[2] + d_q[1] * proj[1] + d_q[2] * proj[0]) ** 2
                    return (
                        gamma
                        * intensity
                        / 2
                        * fijq
                        / (
                            1
                            + 4 * (-(E2g - E1g) + zeeman + delta - jnp.dot(kvec, v)) ** 2 / gamma**2
                        )
                    )

                Rijl = jax.vmap(beam_Rij)(kvecs, intensities, projs, deltas)
                Rijl_store[key] = Rijl
                Rij = jnp.sum(Rijl, axis=0)

                n_idx = jnp.arange(n_off, n_off + n_k)
                m_idx = jnp.arange(m_off, m_off + m_k)
                Rev = Rev.at[n_off : n_off + n_k, m_off : m_off + m_k].add(Rij)
                Rev = Rev.at[m_off : m_off + m_k, n_off : n_off + n_k].add(Rij.T)
                Rev = Rev.at[n_idx, n_idx].add(-jnp.sum(Rij, axis=1))
                Rev = Rev.at[m_idx, m_idx].add(-jnp.sum(Rij, axis=0))

            # Equilibrium populations: last right singular vector of Rev.
            _, _, VH = jnp.linalg.svd(Rev)
            Neq = jnp.abs(VH[-1])
            Neq = Neq / jnp.sum(Neq)

            F = jnp.zeros(3, dtype=jnp.float64)
            f_lasers = {}

            for key, kd in force_data.items():
                n_off, n_k, m_off, m_k = (
                    kd["n_off"],
                    kd["n"],
                    kd["m_off"],
                    kd["m"],
                )
                Ng = Neq[n_off : n_off + n_k]
                Ne = Neq[m_off : m_off + m_k]
                diff_NN = Ng[:, None] - Ne[None, :]  # (n, m)
                Rijl = Rijl_store[key]  # (L, n, m)
                kvecs = kvec_store[key]  # (L, 3)
                scatter = jnp.sum(Rijl * diff_NN[None], axis=(1, 2))
                f_key = (kvecs * scatter[:, None]).T  # (3, L)
                f_lasers[key] = f_key
                F = F + jnp.sum(f_key, axis=1)

            fmag = jnp.zeros(3, dtype=jnp.float64)
            if self.include_mag_forces:
                gradBmag = self.magField.gradFieldMag(r)
                gradBmag = jnp.nan_to_num(gradBmag, nan=0.0)
                for ind1, ind2, mat in mag_mats:
                    if mat is not None:
                        fmag = fmag + jnp.sum(mat @ Neq[ind1:ind2]) * gradBmag
                F = F + fmag

            return Neq, F, f_lasers, fmag, Rijl_store

        # ----------------------------------------------------------------
        # Flatten grid → (N, 3), run vmap, store results
        # ----------------------------------------------------------------
        R_flat = jnp.stack(
            [jnp.asarray(R[i].ravel(), dtype=jnp.float64) for i in range(3)],
            axis=-1,
        )  # (N, 3)
        V_flat = jnp.stack(
            [jnp.asarray(V[i].ravel(), dtype=jnp.float64) for i in range(3)],
            axis=-1,
        )  # (N, 3)

        grid_shape = np.asarray(R[0]).shape

        Neq_all, F_all, f_all, fmag_all, Rijl_all = jax.vmap(single_point)(R_flat, V_flat)

        # Bulk write results (avoids slow Python loop over grid points)
        # Neq_all: (N, n_states) → (grid..., n_states)
        self.profile[name].Neq = np.array(Neq_all.reshape(grid_shape + (-1,)))
        # F_all: (N, 3) → (3, grid...)
        self.profile[name].F = jnp.asarray(F_all.T.reshape((3,) + grid_shape))
        # fmag_all: (N, 3) → (3, grid...)
        self.profile[name].f_mag = jnp.asarray(fmag_all.T.reshape((3,) + grid_shape))
        # f_all[key]: (N, 3, n_beams) → (3, grid..., n_beams)
        ndim_grid = len(grid_shape)
        for key in f_all:
            f_reshaped = f_all[key].reshape(grid_shape + (3, -1))
            self.profile[name].f[key] = jnp.moveaxis(f_reshaped, ndim_grid, 0)
        # Rijl_all[key]: (N, n_beams, n, m) → (grid..., n_beams, n, m)
        for key in Rijl_all:
            self.profile[name].Rijl[key] = np.array(
                Rijl_all[key].reshape(grid_shape + Rijl_all[key].shape[1:])
            )

    def _generate_force_profile_cpu(self, R, V, name, progress_bar=False, **kwargs):
        """Sequential CPU fallback for non-diagonal hamiltonians."""
        it = np.nditer(  # type: ignore[call-overload]
            [R[0], R[1], R[2], V[0], V[1], V[2]],
            flags=["refs_ok", "multi_index"],
            op_flags=[["readonly"]] * 6,  # type: ignore[arg-type]
        )

        if progress_bar:
            progress = progressBar()

        for x, y, z, vx, vy, vz in it:
            r = np.array([x, y, z])
            v = np.array([vx, vy, vz])

            self.set_initial_position_and_velocity(r, v)
            try:
                F, f, Neq, Rijl, f_mag = self.find_equilibrium_force(return_details=True, **kwargs)
            except Exception:
                raise ValueError(
                    "Unable to find solution at "
                    + "r=({0:0.2f},{1:0.2f},{2:0.2f})".format(x, y, z)
                    + " and "
                    + "v=({0:0.2f},{1:0.2f},{2:0.2f})".format(vx, vy, vz)
                )

            if progress_bar:
                progress.update((it.iterindex + 1) / it.itersize)

            self.profile[name].store_data(it.multi_index, Neq, F, f, f_mag, Rijl)
