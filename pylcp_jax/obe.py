"""
Optical Bloch equation (OBE) solver for laser cooling.

Constructs and integrates the full density-matrix equations of motion for an
atom interacting with multiple laser beams and a spatially varying magnetic
field.  Supports GPU-batched trajectory evolution with optional stochastic
photon recoil, equilibrium force finding via chunk-convergence, and vectorised
force-profile generation over position/velocity grids.
"""

import functools
import gc
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .common import (
    base_force_profile,
    cart2spherical,
    progressBar,
    spherical2cart,
)
from .governingeq import governingeq
from .integration_tools_gpu import (
    solve_ivp_dense,
    solve_ivp_random,
)
from .rateeq import rateeq


class force_profile(base_force_profile):
    """
    Optical Bloch equation force profile.

    The force profile object stores all of the calculated quantities created by
    the obe.generate_force_profile() method.  It has the following
    attributes:

    Attributes
    ----------
    R  : array_like, shape (3, ...)
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
    f_q : dictionary of array_like
        The force due to each laser and its :math:`q` component, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp_jax.laserBeams object used to create the governing equation.
    Neq : array_like
        Equilibrium population found.
    """

    def __init__(self, R, V, laserBeams, hamiltonian):
        super().__init__(R, V, laserBeams, hamiltonian)

        self.iterations = np.zeros(self.R[0].shape, dtype="int64")
        self.fq = {}
        for key in laserBeams:
            self.fq[key] = np.zeros(self.R.shape + (3, len(laserBeams[key].beam_vector)))

    def store_data(self, ind, Neq, F, F_laser, F_mag, iterations, F_laser_q):
        """Store force-profile results at a single grid index.

        Parameters
        ----------
        ind : tuple of int
            Multi-dimensional index into the profile arrays.
        Neq : jax.Array or None
            Equilibrium population vector.
        F : jax.Array, shape (3,)
            Total force.
        F_laser : dict of jax.Array
            Per-laser force contributions.
        F_mag : jax.Array, shape (3,)
            Magnetic force contribution.
        iterations : int
            Number of convergence iterations required.
        F_laser_q : dict of jax.Array
            Per-laser, per-q-component force contributions.
        """
        super().store_data(ind, Neq, F, F_laser, F_mag)

        for jj in range(3):
            for key in F_laser_q:
                self.fq[key][(jj,) + ind] = F_laser_q[key][jj]

        self.iterations[ind] = iterations


class obe(governingeq):
    r"""
    The optical Bloch equations.

    This class constructs the optical Bloch equations from the given laser
    beams, magnetic field, and hamiltonian.

    Parameters
    ----------
    laserBeams : dict of pylcp_jax.laserBeams, pylcp_jax.laserBeams, or list of pylcp_jax.laserBeam
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
    transform_into_re_im : boolean
        Optional flag to transform the optical Bloch equations into real and
        imaginary components.  This helps to decrease computation time as it
        uses the symmetry :math:`\\rho_{ji}=\\rho_{ij}^*` to cut the number
        of equations nearly in half.  Default: True
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
        a=jnp.array([0.0, 0.0, 0.0]),
        transform_into_re_im=True,
        include_mag_forces=True,
        r0=jnp.array([0.0, 0.0, 0.0]),
        v0=jnp.array([0.0, 0.0, 0.0]),
    ):

        super().__init__(laserBeams, magField, hamiltonian, a=a, r0=r0, v0=v0)

        # Save the optional arguments:
        self.transform_into_re_im = transform_into_re_im
        self.include_mag_forces = include_mag_forces

        # Set up a dictionary to store any resulting force profiles.
        self.profile = {}

        # Reset the current solution to None
        self.sol: Any = None

        # Pre-build Liouvillian sub-matrices (decay, H0, B, d_q) so the RHS
        # at each time step is just matrix-vector products:
        self.ev_mat = {}
        self.__build_decay_ev()
        self.__build_coherent_ev()

        # If necessary, transform the evolution matrices:
        if self.transform_into_re_im:
            self.__transform_ev_matrices()

        self.__cast_ev_mat_to_jax()

    @property
    def magField(self):
        """Magnetic field object.  Invalidates cached JIT closures when set."""
        return self._magField

    @magField.setter
    def magField(self, value):
        self._magField = value
        # Changing the field invalidates the cached closures so JAX retraces
        # and compiles a new XLA kernel with the updated field.
        self.__dict__.pop("_dydt", None)
        self.__dict__.pop("_motion_dydt", None)

    @property
    def laserBeams(self):
        """Laser beam collection.  Invalidates cached JIT closures when set."""
        return self._laserBeams

    @laserBeams.setter
    def laserBeams(self, value):
        self._laserBeams = value
        self.__dict__.pop("_dydt", None)
        self.__dict__.pop("_motion_dydt", None)

    def update_H0(self, hamiltonian):
        """
        Update only the H0 evolution matrix from a new hamiltonian.

        This is much faster than reconstructing the full OBE when only the
        field-free Hamiltonian (e.g. detunings) has changed, since the decay
        and dipole coupling matrices remain the same.

        Parameters
        ----------
        hamiltonian : pylcp_jax.hamiltonian
            A new hamiltonian whose H_0 block will be used.
        """
        hamiltonian.make_full_matrices()
        self.hamiltonian = hamiltonian
        H0_ev = self.__build_coherent_ev_submatrix(np.array(hamiltonian.H_0))
        if self.transform_into_re_im:
            H0_ev = self.__transform_ev_matrix(H0_ev)
        dtype = jnp.float64 if self.transform_into_re_im else jnp.complex128
        self.ev_mat["H0"] = jnp.asarray(H0_ev, dtype=dtype)
        self.__dict__.pop("_dydt", None)
        self.__dict__.pop("_motion_dydt", None)

    def __cast_ev_mat_to_jax(self):
        """Recursively convert the nested dictionaries of numpy arrays to jax arrays."""
        dtype = jnp.float64 if self.transform_into_re_im else jnp.complex128
        for key in self.ev_mat:
            if isinstance(self.ev_mat[key], dict):
                for subkey in self.ev_mat[key]:
                    self.ev_mat[key][subkey] = jnp.asarray(self.ev_mat[key][subkey], dtype=dtype)
            elif isinstance(self.ev_mat[key], list):
                self.ev_mat[key] = [jnp.asarray(v, dtype=dtype) for v in self.ev_mat[key]]
            else:
                self.ev_mat[key] = jnp.asarray(self.ev_mat[key], dtype=dtype)

    def __density_index(self, ii, jj):
        """Return the index in the rho vector that corresponds to element rho_{ij}."""
        return ii + jj * self.hamiltonian.n

    def __build_coherent_ev_submatrix(self, H):
        """
        Build the coherent evolution Liouvillian from a Hamiltonian submatrix.

        In practice, one must be careful about commutators if one breaks up the
        Hamiltonian.  The density matrix is vectorized column-major:
        rho_flat[i + j*n] = rho[i,j].  The Liouvillian L such that
        d/dt rho_flat = L @ rho_flat is:
            L = i * kron(H.T, I_n) - i * kron(I_n, H)
        """
        n = self.hamiltonian.n
        I = np.eye(n)
        H = np.asarray(H)
        return 1j * np.kron(H.T, I) - 1j * np.kron(I, H)

    # Is only used in construction so can remain in np
    def __build_coherent_ev(self):
        self.ev_mat["H0"] = self.__build_coherent_ev_submatrix(np.array(self.hamiltonian.H_0))

        b_list: list[Any] = [None] * 3
        self.ev_mat["B"] = b_list
        for q in range(3):
            self.ev_mat["B"][q] = self.__build_coherent_ev_submatrix(
                np.array(self.hamiltonian.mu_q[q])
            )

        self.ev_mat["d_q"] = {}
        self.ev_mat["d_q*"] = {}
        for key in self.laserBeams.keys():
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters["gamma"]
            self.ev_mat["d_q"][key] = [None] * 3
            self.ev_mat["d_q*"][key] = [None] * 3
            for q in range(3):
                self.ev_mat["d_q"][key][q] = self.__build_coherent_ev_submatrix(
                    gamma * self.hamiltonian.d_q_bare[key][q] / 4.0
                )
                self.ev_mat["d_q*"][key][q] = self.__build_coherent_ev_submatrix(
                    gamma * self.hamiltonian.d_q_star[key][q] / 4.0
                )
            self.ev_mat["d_q"][key] = np.array(self.ev_mat["d_q"][key])
            self.ev_mat["d_q*"][key] = np.array(self.ev_mat["d_q*"][key])

    # is used only in construction
    def __build_decay_ev(self):
        """
        Construct the decay portion of the OBE.

        Uses the radiation reaction approximation.
        """
        d_q_bare = self.hamiltonian.d_q_bare
        d_q_star = self.hamiltonian.d_q_star

        self.decay_rates = {}
        self.decay_rates_truncated = {}
        self.decay_rho_indices = {}
        self.recoil_velocity = {}

        self.ev_mat["decay"] = np.zeros(
            (self.hamiltonian.n**2, self.hamiltonian.n**2), dtype="complex128"
        )

        # Go through each dipole moment and calculate:
        for key in d_q_bare:
            ev_mat = np.zeros(
                (self.hamiltonian.n**2, self.hamiltonian.n**2),
                dtype="complex128",
            )
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters["gamma"]

            # The first index we want to capture:
            for ii in range(self.hamiltonian.n):
                # The second index we want to capture:
                for jj in range(self.hamiltonian.n):
                    # The first sum index:
                    for kk in range(self.hamiltonian.n):
                        # The second sum index:
                        for ll in range(self.hamiltonian.n):
                            for mm, q in enumerate(np.arange(-1.0, 2.0, 1)):
                                # first term in the commutator, first part:
                                ev_mat[
                                    self.__density_index(ii, jj),
                                    self.__density_index(ll, jj),
                                ] -= d_q_star[key][mm, ll, kk] * d_q_bare[key][mm, kk, ii]
                                # first term in the commutator, second part:
                                ev_mat[
                                    self.__density_index(ii, jj),
                                    self.__density_index(kk, ll),
                                ] += d_q_star[key][mm, kk, ii] * d_q_bare[key][mm, jj, ll]

                                # second term in the commutator, first part:
                                ev_mat[
                                    self.__density_index(ii, jj),
                                    self.__density_index(ll, kk),
                                ] += d_q_star[key][mm, ll, ii] * d_q_bare[key][mm, jj, kk]
                                # second term in the commutator, second part:
                                ev_mat[
                                    self.__density_index(ii, jj),
                                    self.__density_index(ii, ll),
                                ] -= d_q_star[key][mm, jj, kk] * d_q_bare[key][mm, kk, ll]

            # Normalize:
            ev_mat = 0.5 * gamma * ev_mat

            # Save the decay rates for the evolve_motion function:
            self.decay_rates[key] = -np.real(
                np.array(
                    [
                        ev_mat[
                            self.__density_index(ii, ii),
                            self.__density_index(ii, ii),
                        ]
                        for ii in range(self.hamiltonian.n)
                    ]
                )
            )

            # These are useful for the random evolution part:
            self.decay_rates_truncated[key] = self.decay_rates[key][self.decay_rates[key] > 0]
            self.decay_rho_indices[key] = np.array(
                [
                    self.__density_index(ii, ii)
                    for ii, rate in enumerate(self.decay_rates[key])
                    if rate > 0
                ]
            )
            self.recoil_velocity[key] = (
                self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters["k"]
                / self.hamiltonian.mass
            )

            self.ev_mat["decay"] += ev_mat

        return self.ev_mat["decay"]

    # is only ran in setup
    def __build_transform_matrices(self):
        self.U = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2), dtype="complex128")
        self.Uinv = np.zeros((self.hamiltonian.n**2, self.hamiltonian.n**2), dtype="complex128")

        for ii in range(self.hamiltonian.n):
            self.U[self.__density_index(ii, ii), self.__density_index(ii, ii)] = 1.0
            self.Uinv[self.__density_index(ii, ii), self.__density_index(ii, ii)] = 1.0

        for ii in range(self.hamiltonian.n):
            for jj in range(ii + 1, self.hamiltonian.n):
                self.U[self.__density_index(ii, jj), self.__density_index(ii, jj)] = 1.0
                self.U[self.__density_index(ii, jj), self.__density_index(jj, ii)] = 1j

                self.U[self.__density_index(jj, ii), self.__density_index(ii, jj)] = 1.0
                self.U[self.__density_index(jj, ii), self.__density_index(jj, ii)] = -1j

        for ii in range(self.hamiltonian.n):
            for jj in range(ii + 1, self.hamiltonian.n):
                self.Uinv[self.__density_index(ii, jj), self.__density_index(ii, jj)] = 0.5
                self.Uinv[self.__density_index(ii, jj), self.__density_index(jj, ii)] = 0.5

                self.Uinv[self.__density_index(jj, ii), self.__density_index(ii, jj)] = -0.5 * 1j
                self.Uinv[self.__density_index(jj, ii), self.__density_index(jj, ii)] = +0.5 * 1j

    # is only ran in setup
    def __transform_ev_matrix(self, ev_mat):
        if not hasattr(self, "U"):
            self.__build_transform_matrices()

        ev_mat_new = self.Uinv @ ev_mat @ self.U

        # This should remove the imaginary component.
        if np.allclose(np.imag(ev_mat_new), 0):
            return np.real(ev_mat_new)
        else:
            raise ValueError("Something went dreadfully wrong.")

    # is only ran in setup
    def __transform_ev_matrices(self):
        self.ev_mat["decay"] = self.__transform_ev_matrix(self.ev_mat["decay"])
        self.ev_mat["H0"] = self.__transform_ev_matrix(self.ev_mat["H0"])

        self.ev_mat["reE"] = {}
        self.ev_mat["imE"] = {}
        for key in self.ev_mat["d_q"].keys():
            self.ev_mat["reE"][key] = np.array(
                [
                    self.__transform_ev_matrix(
                        self.ev_mat["d_q"][key][jj] + self.ev_mat["d_q*"][key][jj]
                    )
                    for jj in range(3)
                ]
            )
            # Unclear why the following works, I calculate that there should
            # be a minus sign out front.
            self.ev_mat["imE"][key] = np.array(
                [
                    self.__transform_ev_matrix(
                        1j * (self.ev_mat["d_q"][key][jj] - self.ev_mat["d_q*"][key][jj])
                    )
                    for jj in range(3)
                ]
            )

        # Transform Bq back into Bx, By, and Bz (making it real):
        self.ev_mat["B"] = spherical2cart(self.ev_mat["B"])

        self.ev_mat["B"] = np.real(
            np.array([self.__transform_ev_matrix(self.ev_mat["B"][jj]) for jj in range(3)])
        )

        del self.ev_mat["d_q"]
        del self.ev_mat["d_q*"]

    def __reshape_rho(self, rho):
        # Use numpy ops when input is already on CPU to avoid GPU memory
        # accumulation (e.g. when called from __reshape_sol after np.asarray).
        if isinstance(rho, np.ndarray):
            if self.transform_into_re_im:
                rho = rho.astype(np.complex128)
                if len(rho.shape) == 1:
                    rho = np.dot(self.U, rho)
                else:
                    rho = np.tensordot(self.U, rho, axes=([1], [0]))
            rho = np.reshape(rho, (self.hamiltonian.n, self.hamiltonian.n) + rho.shape[1:])
        else:
            rho = jnp.asarray(rho)
            if self.transform_into_re_im:
                rho = rho.astype(jnp.complex128)
                if len(rho.shape) == 1:
                    rho = jnp.dot(self.U, rho)
                else:
                    rho = jnp.tensordot(self.U, rho, axes=([1], [0]))
            rho = jnp.reshape(rho, (self.hamiltonian.n, self.hamiltonian.n) + rho.shape[1:])

        """# If not:
        if self.transform_into_re_im:
            new_rho = np.zeros(rho.shape, dtype='complex128')
            for jj in range(new_rho.shape[2]):
                new_rho[:, :, jj] = (np.diag(np.diagonal(rho[:, :, jj])) +
                                     np.triu(rho[:, :, jj], k=1) +
                                     np.triu(rho[:, :, jj], k=1).T +
                                     1j*np.tril(rho[:, :, jj], k=-1) -
                                     1j*np.tril(rho[:, :, jj], k=-1).T)
            rho = new_rho"""

        return rho

    def __reshape_sol(self):
        """Reshape the solution to have all the proper parts."""
        # Transfer to CPU first to avoid GPU OOM on large batches.
        for sol in self.sols:
            y_cpu = np.asarray(sol.y)  # GPU → CPU transfer
            rho_flat = y_cpu[:-6, :]  # (n^2, n_steps)
            sol.rho = np.asarray(self.__reshape_rho(rho_flat))
            sol.v = np.real(y_cpu[-6:-3, :])  # (3, n_steps)
            sol.r = np.real(y_cpu[-3:, :])  # (3, n_steps)
            del sol.y

    def set_initial_rho(self, rho0):
        r"""
        Set the initial :math:`\\rho` matrix.

        Parameters
        ----------
        rho0 : array_like
            The initial :math:`\\rho`.  It must have :math:`n^2` elements, where :math:`n`
            is the total number of states in the system.  If a flat array, it
            will be reshaped.
        """
        rho0 = jnp.asarray(rho0)
        if jnp.any(jnp.isnan(rho0)) or jnp.any(jnp.isinf(rho0)):
            raise ValueError("rho0 has NaNs or Infs!")

        if rho0.size != self.hamiltonian.n**2:
            raise ValueError("rho0 should have n^2 elements.")

        if rho0.shape == (self.hamiltonian.n, self.hamiltonian.n):
            rho0 = rho0.flatten()

        if self.transform_into_re_im and rho0.dtype == jnp.complex128:
            # self.rho0 = self.Uinv @ rho0
            self.rho0 = jnp.dot(self.Uinv, rho0)
        elif not self.transform_into_re_im and rho0.dtype != jnp.complex128:
            self.rho0 = rho0.astype(jnp.complex128)
        else:
            self.rho0 = rho0

    def set_initial_rho_equally(self):
        r"""
        Set the initial :math:`\\rho` matrix to equal populations.

        All states receive the same population.
        """
        if self.transform_into_re_im:
            self.rho0 = jnp.zeros((self.hamiltonian.n**2,))
        else:
            self.rho0 = jnp.zeros((self.hamiltonian.n**2,), dtype=jnp.complex128)

        # only runs in initialization so is okay with for loop
        for jj in range(self.hamiltonian.ns[0]):
            self.rho0 = self.rho0.at[self.__density_index(jj, jj)].set(1 / self.hamiltonian.ns[0])

    def set_initial_rho_from_populations(self, Npop):
        r"""
        Set the diagonal elements of the initial :math:`\\rho` matrix.

        Parameters
        ----------
        Npop : array_like
            Array of the initial populations of the states in the system.  The
            length must be :math:`n`, where :math:`n` is the number of states.
        """
        Npop = jnp.asarray(Npop)
        if len(Npop) != self.hamiltonian.n:
            raise ValueError(
                "Npop has only %d entries for %d states." % (len(Npop), self.hamiltonian.n)
            )
        if jnp.any(jnp.isnan(Npop)) or jnp.any(jnp.isinf(Npop)):
            raise ValueError("Npop has NaNs or Infs!")

        if self.transform_into_re_im:
            self.rho0 = jnp.zeros((self.hamiltonian.n**2,))
        else:
            self.rho0 = jnp.zeros((self.hamiltonian.n**2,), dtype=jnp.complex128)

        Npop = Npop / jnp.sum(Npop)  # Just make sure it is normalized.
        # okay with for loop for initialization function
        for jj in range(self.hamiltonian.n):
            idx = self.__density_index(jj, jj)
            self.rho0 = self.rho0.at[idx].set(Npop[jj])

    def set_initial_rho_from_rateeq(self):
        r"""
        Set the diagonal elements of the initial :math:`\\rho` matrix.

        Uses the equilibrium populations as determined by pylcp_jax.rateeq.
        """
        if not hasattr(self, "rateeq"):
            self.rateeq = rateeq(self.laserBeams, self.magField, self.hamiltonian)
        Neq = self.rateeq.equilibrium_populations(self.r0, self.v0, t=0)
        self.set_initial_rho_from_populations(Neq)

    def full_OBE_ev_scratch(self, r, t):
        """
        Calculate the evolution for the density matrix.

        This function calculates the OBE evolution matrix at position t and r
        from scratch, first computing the full Hamiltonian, then the
        OBE evolution matrix computed via commutators, then adding in the decay
        matrix evolution.  If `Bq` is `None`, it will compute `Bq`.

        Parameters
        ----------
        r : array_like, shape (3,)
            Position at which to calculate evolution matrix
        t : float
            Time at which to calculate evolution matrix

        Returns
        -------
        ev_mat : array_like
            Evolution matrix for the densities
        """
        Eq = {}
        for key in self.laserBeams.keys():
            Eq[key] = self.laserBeams[key].total_electric_field(r, t)

        B = self.magField.Field(r, t)
        Bq = cart2spherical(B)

        H = self.hamiltonian.return_full_H(Bq, Eq)
        ev_mat = self.__build_coherent_ev_submatrix(H)

        if self.transform_into_re_im:
            return self.__transform_ev_matrix(ev_mat + self.ev_mat["decay"])
        else:
            return ev_mat + self.ev_mat["decay"]

    def full_OBE_ev(self, r, t):
        """
        Calculate the evolution for the density matrix.

        This function calculates the OBE evolution matrix by assembling
        pre-stored versions of the component matries.  This should be
        significantly faster than full_OBE_ev_scratch, but it may suffer bugs
        in the evolution that full_OBE_ev_scratch will not. If Bq is None,
        it will compute Bq based on r, t

        Parameters
        ----------
        r : array_like, shape (3,)
            Position at which to calculate evolution matrix
        t : float
            Time at which to calculate evolution matrix

        Returns
        -------
        ev_mat : array_like
            Evolution matrix for the densities
        """
        ev_mat = self.ev_mat["decay"] + self.ev_mat["H0"]

        # Add in electric fields:
        for key in self.laserBeams.keys():
            if self.transform_into_re_im:
                Eq = self.laserBeams[key].total_electric_field(r, t)
                for ii in range(3):
                    ev_mat -= jnp.real(Eq[ii]) * self.ev_mat["reE"][key][ii]
                    ev_mat -= jnp.imag(Eq[ii]) * self.ev_mat["imE"][key][ii]
            else:
                Eq = self.laserBeams[key].total_electric_field(jnp.real(r), t)
                for ii in range(3):
                    ev_mat -= jnp.conjugate(Eq[ii]) * self.ev_mat["d_q"][key][ii]
                    ev_mat -= Eq[ii] * self.ev_mat["d_q*"][key][ii]

        # Add in magnetic fields:
        B = self.magField.Field(r, t)
        if self.transform_into_re_im:
            for ii in range(3):
                ev_mat = ev_mat - self.ev_mat["B"][ii] * B[ii]
        else:
            Bq = cart2spherical(B)
            for ii in range(3):
                ev_mat = ev_mat - self.ev_mat["B"][ii] * jnp.conjugate(Bq[ii])

        return ev_mat

    def __drhodt(self, r, t, rho):
        """Compute d(rho)/dt via per-component matrix-vector products."""
        drhodt = jnp.dot(self.ev_mat["decay"], rho) + jnp.dot(self.ev_mat["H0"], rho)

        # Add in electric fields:
        if self.transform_into_re_im:
            for key in self.laserBeams.keys():
                Eq = self.laserBeams[key].total_electric_field(r, t)
                for ii, q in enumerate([-1.0, 2.0, 1]):
                    drhodt -= (
                        (-1.0) ** q
                        * jnp.real(Eq[2 - ii])
                        * jnp.dot(self.ev_mat["reE"][key][ii], rho)
                    )
                    drhodt -= (
                        (-1.0) ** q
                        * jnp.imag(Eq[2 - ii])
                        * jnp.dot(self.ev_mat["imE"][key][ii], rho)
                    )
        else:
            for key in self.laserBeams.keys():
                Eq = self.laserBeams[key].total_electric_field(jnp.real(r), t)
                for ii, q in enumerate([-1.0, 2.0, 1]):
                    drhodt -= (-1.0) ** q * Eq[2 - ii] * jnp.dot(self.ev_mat["d_q"][key][ii], rho)
                    drhodt -= (
                        (-1.0) ** q
                        * jnp.conjugate(Eq[2 - ii])
                        * jnp.dot(self.ev_mat["d_q*"][key][ii], rho)
                    )

        # Add in magnetic fields:
        B = self.magField.Field(r, t)
        if self.transform_into_re_im:
            for ii in range(3):
                drhodt -= jnp.dot(self.ev_mat["B"][ii] * B[ii], rho)
        else:
            Bq = cart2spherical(B)
            for ii in range(3):
                drhodt -= jnp.dot(self.ev_mat["B"][ii] * jnp.conjugate(Bq[ii]), rho)

        return drhodt

    @functools.cached_property
    def _dydt(self):
        """Density-only ODE RHS (no motion). Cached for stable JIT keys."""

        def dydt(t, y, _args):
            r = y[-3:]
            v = y[-6:-3]
            rho = y[:-6]
            a = jnp.zeros(3, dtype=y.dtype)
            drhodt = self.__drhodt(r, t, rho)
            return jnp.concatenate((drhodt, a, v))

        return dydt

    @functools.cached_property
    def _motion_dydt(self):
        """Full motion ODE RHS (rho + v + r). Cached for stable JIT keys."""

        def dydt(t, y, args):
            r = y[-3:]
            v = y[-6:-3]
            rho = y[:-6]
            free_axes = args["free_axes"]

            F = self.force(r, t, rho, return_details=False)
            dvdt = (F * free_axes) / self.hamiltonian.mass + self.constant_accel
            drdt = v
            drhodt = self.__drhodt(r, t, rho)
            return jnp.concatenate((drhodt, dvdt, drdt))

        return dydt

    @functools.cached_property
    def _motion_recoil_fn(self):
        """Stochastic recoil function. Cached for stable JIT keys."""
        # Snapshot decay data as JAX arrays at first access.
        decay_channels = []
        n_excited_per_channel = []
        for dk in self.decay_rates:
            decay_channels.append(
                (
                    jnp.asarray(self.decay_rates_truncated[dk]),
                    jnp.asarray(self.decay_rho_indices[dk]),
                    jnp.asarray(self.recoil_velocity[dk]),
                )
            )
            n_excited_per_channel.append(len(self.decay_rates_truncated[dk]))

        def recoil_fn(t, y, dt, key, args):
            """Apply stochastic spontaneous-emission recoil kicks (OBE path)."""
            free_axes = args["free_axes"]
            max_scatter_probability = args["max_scatter_probability"]

            def _rand_vec(k):
                """Return a random unit vector restricted to free axes."""
                k1, k2 = jax.random.split(k)
                phi = 2.0 * jnp.pi * jax.random.uniform(k1)
                z = 2.0 * jax.random.uniform(k2) - 1.0
                r = jnp.sqrt(1.0 - z**2)
                return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z]) * free_axes

            # 3 keys per excited state (dice, vec1, vec2) + 1 new key
            total_excited = sum(n_excited_per_channel)
            all_keys = jax.random.split(key, 3 * total_excited + 1)
            key_new = all_keys[0]
            ki = 1

            y_jump = y
            num_of_scatters = jnp.int32(0)
            total_P = 0.0

            for ch_idx, (rates, indices, recoil_v) in enumerate(decay_channels):
                P = dt * rates * jnp.real(y[indices])

                for ii in range(n_excited_per_channel[ch_idx]):
                    dice = jax.random.uniform(all_keys[ki])
                    vec1 = _rand_vec(all_keys[ki + 1])
                    vec2 = _rand_vec(all_keys[ki + 2])
                    ki += 3

                    scattered = dice < P[ii]
                    kick = recoil_v * (vec1 + vec2)
                    delta_v = jnp.where(
                        scattered,
                        kick * free_axes,
                        jnp.zeros(3, dtype=jnp.float64),
                    )
                    y_jump = y_jump.at[-6:-3].add(delta_v)
                    num_of_scatters += jnp.where(scattered, jnp.int32(1), jnp.int32(0))
                    total_P += P[ii]

            new_dt_max = jnp.where(total_P > 0, (max_scatter_probability / total_P) * dt, jnp.inf)
            return y_jump, num_of_scatters, new_dt_max, key_new

        return recoil_fn

    @staticmethod
    def _no_recoil(t, y, dt, key, args):
        """No-op recoil function with stable identity for JIT caching."""
        return y, 0, jnp.inf, key

    def evolve_density(self, t_span, y0_batch=None, n_points=1000, **kwargs):
        r"""
        Evolve the density operators :math:`\\rho_{ij}` in time.

        This function integrates the optical Bloch equations to determine how
        the populations evolve in time.  Any initial velocity is kept constant
        while the atom potentially moves through the light field.  This function
        is therefore useful in determining average forces.  Any constant
        acceleration set when the OBEs were generated is ignored. It is
        analogous to rateeq.evolve_populations().

        Parameters
        ----------
        t_span : list or array_like
            A two element list or array that specify the initial and final time
            of integration.
        y0_batch : array_like, shape (n_atoms, state_dim), optional
            Batch of initial state vectors. If None, uses ``self.rho0``,
            ``self.v0``, and ``self.r0``. Default: None.
        n_points : int, optional
            Number of output time points. Default: 1000.
        **kwargs :
            Additional keyword arguments:

            rtol : float, optional
                Relative tolerance. Default: 1e-5.
            atol : float, optional
                Absolute tolerance. Default: 1e-6.
            max_steps : int, optional
                Maximum solver steps. Default: 4096.

        Returns
        -------
        sol : Bunch
            Object with the following fields:

                * t: integration times, shape ``(n_points,)``
                * rho: density matrix, shape ``(n, n, n_points)``
                * y: raw batched state array, shape ``(n_atoms, n_points, state_dim)``
        """
        rtol = kwargs.get("rtol", 1e-5)
        atol = kwargs.get("atol", 1e-6)
        max_steps = kwargs.get("max_steps", 4096)
        method = kwargs.get("method", "Dopri5")

        if y0_batch is None:
            y0 = jnp.concatenate([self.rho0, self.v0, self.r0])
            y0_batch = y0[None, :]  # (1, state_dim)

        ts_grid, batched_ys = solve_ivp_dense(
            self._dydt,
            t_span,
            y0_batch,
            n_points=n_points,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
            solver_type=method,
            args=None,
        )
        # batched_ys shape: (n_atoms, n_points, state_dim)

        class Bunch:
            """Simple namespace for OBE solution arrays (t, rho, y)."""

            t: Any
            y: Any
            rho: Any

        self.sol = Bunch()
        self.sol.t = ts_grid
        self.sol.y = batched_ys
        # reconstructing the rho for output
        self.sol.rho = self.__reshape_rho(batched_ys[0, :, : self.hamiltonian.n**2].T)

        return self.sol

    def evolve_motion(
        self,
        t_span,
        n_points,
        y0_batch=None,
        keys_batch=None,
        freeze_axis=[False, False, False],
        random_recoil=False,
        max_scatter_probability=0.1,
        backend="auto",
        **kwargs,
    ):
        r"""
        Evolve :math:`\\rho_{ij}` and the motion of the atom in time.

        This function evolves the optical Bloch equations, moving the atom
        along given the instantaneous force, for some period of time. The
        integration is performed using a JAX/diffrax adaptive solver with a
        custom outer loop that records the state at each step.

        Parameters
        ----------
        t_span : list or array_like
            A two element list or array that specify the initial and final time
            of integration.
        n_points : int
            Number of evenly-spaced output time points.  The solver takes
            as many internal adaptive steps as needed and saves the state
            at each output time.
        freeze_axis : list of boolean
            Freeze atomic motion along the specified axis.
            Default: [False, False, False]
        random_recoil : boolean
            Allow the atom to randomly recoil from scattering events.
            Default: False
        backend : str, optional
            Execution backend for the ODE solver.

            * ``'auto'`` *(default)* — use GPU (batched vmap) when a CUDA
              device is available, otherwise fall back to CPU serial.
            * ``'gpu'`` — always run the full batch through ``solve_ivp_random``
              in one vmapped call.  Best for large batches on GPU.
            * ``'cpu'`` — iterate over atoms and call ``solve_ivp_random``
              one at a time.  No vmap overhead; suitable for CPU workers or
              when the batch is small enough that serial is faster.
        max_scatter_probability : float
            When undergoing random recoils, this sets the maximum time step such
            that the maximum scattering probability is less than or equal to
            this number during the next time step.  Default: 0.1
        **kwargs :
            Additional keyword arguments passed to ``solve_ivp_random``.
            Important options include:

            max_step : float, optional
                Ceiling on the recoil-limited step size.  When
                ``random_recoil=True`` and ``max_step`` is not given, it is
                auto-computed from the on-resonance scattering rate so that
                all atoms take similarly-sized steps (eliminates GPU warp
                divergence).  Pass explicitly to override.
            rtol : float, optional
                Relative tolerance for the adaptive step controller.
                Default: 1e-5.
            atol : float, optional
                Absolute tolerance for the adaptive step controller.
                Default: 1e-6.
            solver_type : str, optional
                Diffrax solver to use: ``'Dopri5'``, ``'Bosh3'``, or
                ``'Kvaerno5'``. Default: ``'Dopri5'``.

        Returns
        -------
        sols : list of RandomOdeResult
            A list with one result per trajectory in the batch. Each result
            is also accessible as ``self.sol`` (first trajectory) or
            ``self.sols``. Each result contains:

                * t: integration times, shape ``(n_steps,)``
                * rho: density matrix, shape ``(n, n, n_steps)``
                * v: atomic velocity, shape ``(3, n_steps)``
                * r: atomic position, shape ``(3, n_steps)``
        """
        if y0_batch is None:
            y0 = jnp.concatenate([self.rho0, self.v0, self.r0])
            y0_batch = y0[jnp.newaxis, :]
        if keys_batch is None:
            keys_batch = jax.random.split(
                jax.random.PRNGKey(np.random.randint(0, 2**31)),
                y0_batch.shape[0],
            )

        free_axes = jnp.bitwise_not(jnp.asarray(freeze_axis, dtype=bool))

        # Auto-compute max_step from on-resonance scattering rate to
        # reduce GPU warp divergence when random_recoil is enabled.
        if random_recoil and "max_step" not in kwargs:
            total_s: float = sum(
                beam._s if not callable(beam._s) else beam._s(np.zeros(3), 0.0)  # type: ignore[arg-type]
                for beams in self.laserBeams.values()
                for beam in beams.beam_vector
            )
            # On-resonance max scattering rate: gamma/2 * S_tot / (1 + S_tot)
            # This is a theoretical upper bound; actual rates are always ≤ this.
            gamma = self.hamiltonian.blocks[0, 1].parameters["gamma"]
            R_max = (gamma / 2) * total_s / (1 + total_s)
            if R_max > 0:
                kwargs["max_step"] = max_scatter_probability / R_max

        # Per-call parameters as a JAX pytree (read by cached closures).
        args = {
            "free_axes": free_axes,
            "max_scatter_probability": jnp.asarray(max_scatter_probability, dtype=jnp.float64),
        }

        dydt = self._motion_dydt
        recoil_func = self._no_recoil if not random_recoil else self._motion_recoil_fn

        # Resolve backend: 'auto' uses GPU when a CUDA device is present.
        resolved = backend
        if resolved == "auto":
            resolved = "gpu" if jax.default_backend() == "gpu" else "cpu"

        if resolved == "gpu":
            self.sols = solve_ivp_random(
                fun=dydt,
                random_func=recoil_func,
                t_span=t_span,
                y0_batch=y0_batch,
                keys_batch=keys_batch,
                n_points=n_points,
                args=args,
                **kwargs,
            )
        else:
            # CPU serial: one atom at a time — no vmap across the batch.
            self.sols = []
            for i in range(y0_batch.shape[0]):
                sol_list = solve_ivp_random(
                    fun=dydt,
                    random_func=recoil_func,
                    t_span=t_span,
                    y0_batch=y0_batch[i : i + 1],
                    keys_batch=keys_batch[i : i + 1],
                    n_points=n_points,
                    args=args,
                    **kwargs,
                )
                self.sols.extend(sol_list)

        # Remake the solution:
        self.__reshape_sol()

        # For convenience, expose the first trajectory as self.sol:
        if self.sols:
            self.sol = self.sols[0]

        return self.sols

    def observable(self, O, rho=None):
        """
        Calculate the expectation value of the observable O given density matrix rho.

        This method computes the trace of the product of the observable operator
        and the density matrix. It natively supports evaluating single states
        as well as batched/time-series density matrices using JAX.

        Parameters
        ----------
        O : array_like
            The matrix form of the observable operator. Can have any shape,
            representing scalar, vector, or tensor operators, but the last two
            axes must correspond to the matrix of the operator and have the
            same dimensions as the generating Hamiltonian. For example,
            a vector operator might have the shape (3, n, n), where n
            is the number of states and the first axis corresponds to x, y,
            and z.
        rho : array_like, optional
            The density matrix. The first two dimensions must have sizes
            (n, n), but there may be multiple instances of the density matrix
            tiled in higher dimensions (e.g., (n, n, m) for a time series).
            Alternatively, a flat 1D array of length n**2 can be provided
            and will be automatically reshaped. If not specified, the method
            will default to the current solution stored in memory (`self.sols`
            or `self.sol`).

        Returns
        -------
        observable : jax.Array
            The calculated expectation value(s). The output shape is
            `O.shape[:-2] + rho.shape[2:]`.
        """
        if rho is None:  # handle if rho is missing
            if hasattr(self, "sols") and isinstance(self.sols, list) and len(self.sols) > 0:
                rho = self.sols[0].rho
            elif hasattr(self, "sol"):
                rho = self.sol.rho
            else:
                raise ValueError("No solution found in memory. Please provide 'rho' explicitly.")
        if rho is None:
            raise ValueError("No rho available in the stored solution.")

        O = jnp.asarray(O)  # convert to jax array in case
        if rho.shape[0] == self.hamiltonian.n**2 and rho.ndim == 1:
            rho_mat = self.__reshape_rho(rho)
        else:
            rho_mat = jnp.asarray(rho)

        if rho_mat.shape[:2] != (self.hamiltonian.n, self.hamiltonian.n):
            raise ValueError(
                "rho must have dimensions (n, n,...), where n "
                + "corresponds to the number of states in the "
                + "generating Hamiltonian. "
                + "Instead, shape of rho is %s." % str(rho.shape)
            )
        elif O.shape[-2:] != (self.hamiltonian.n, self.hamiltonian.n):
            raise ValueError(
                "O must have dimensions (..., n, n), where n "
                + "corresponds to the number of states in the "
                + "generating Hamiltonian. "
                + "Instead, shape of O is %s." % str(O.shape)
            )
        return jnp.real(jnp.tensordot(O, rho_mat, axes=[(-2, -1), (0, 1)]))

    def force(self, r, t, rho, return_details=False):
        """
        Calculate the instantaneous force on the atom.

        This method computes the gradient of the laser and magnetic fields
        and traces them with the density matrix to find the instantaneous
        expectation value of the force. It supports evaluating single points
        as well as batched/time-series data arrays natively using JAX.

        Parameters
        ----------
        r : array_like, shape (3,) or (3, n_pts)
            Position(s) at which to calculate the force.
        t : float or array_like, shape (n_pts,)
            Time(s) at which to calculate the force.
        rho : array_like
            Density matrix (or flat state vector representation) with which to
            calculate the force.
        return_details : boolean, optional
            If true, returns the detailed components of the force broken down
            by laser, polarization, and magnetic fields. Default: False.

        Returns
        -------
        f : jax.Array
            Total instantaneous force experienced by the atom.
        f_laser : dict of jax.Array
            (Returned if `return_details=True`) The forces due to each laser,
            indexed by the manifold the laser addresses.
        f_laser_q : dict of jax.Array
            (Returned if `return_details=True`) The forces due to each laser
            and its q component of the polarization, indexed by the manifold
            the laser addresses.
        f_mag : jax.Array
            (Returned if `return_details=True`) The forces due to the
            magnetic field.
        """
        rho_mat = self.__reshape_rho(rho)

        f = jnp.zeros((3,) + rho_mat.shape[2:])
        f_mag = jnp.zeros_like(f)
        if return_details:
            f_laser_q = {}
            f_laser = {}

        # Precomputed constants for the q-component vectorisation.
        _q_signs = jnp.array([-1.0, 1.0, -1.0])  # (-1)^q for q = -1, 2, 1
        _q_col = jnp.array([2, 1, 0])  # column index 2-jj

        # Helper: evaluate field gradient at r (3,) or (3, n_pts) and t scalar/(n_pts,)
        def _grad(beam_func, r_in, t_in):
            if jnp.asarray(r_in).ndim == 2:  # time series
                t_arr = jnp.asarray(t_in).ravel()
                dE = jax.vmap(lambda ri, ti: beam_func(ri, ti))(
                    jnp.real(r_in).T,
                    t_arr,  # (n_pts, 3)
                )
                return jnp.moveaxis(dE, 0, -1)  # (..., n_pts)
            else:
                return beam_func(jnp.real(r_in), t_in)

        for key in self.laserBeams:
            # Compute the complex average of each d_q component.  We need the
            # full complex value (not just Re) because the force is
            # Re[<d_q> * ∇E] where ∇E is also complex (e.g. ∝ ik for a plane
            # wave). Using observable() would apply jnp.real() prematurely and
            # zero out the imaginary-coherence contribution to the force.
            mu_q_av = jnp.tensordot(
                jnp.asarray(self.hamiltonian.d_q_bare[key]),
                rho_mat,
                axes=[(-2, -1), (0, 1)],
            )
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters["gamma"]

            if not return_details:
                delE = _grad(self.laserBeams[key].total_electric_field_gradient, r, t)

                # F = Re[∇(d_q · E†)] * gamma/2  (factor from dipole + RWA units)
                for jj, q in enumerate([-1.0, 2.0, 1.0]):
                    f += jnp.real((-1) ** q * gamma * mu_q_av[jj] * delE[:, 2 - jj]) / 2
            else:
                # All beam gradients in one call: (n_beams, 3_spatial, 3_grad, [n_pts])
                delE_all = _grad(self.laserBeams[key].electric_field_gradient, r, t)

                delE_q = delE_all[:, :, _q_col]  # select q columns
                delE_q = jnp.moveaxis(delE_q, 0, 2)  # -> (3_s, 3_q, n_b, [n_pts])

                # Broadcast signs and mu_q_av for vectorised multiply
                if rho_mat.ndim == 2:  # single point
                    s = _q_signs[None, :, None]  # (1, 3_q, 1)
                    m = mu_q_av[None, :, None]  # (1, 3_q, 1)
                else:  # time series
                    s = _q_signs[None, :, None, None]  # (1, 3_q, 1, 1)
                    m = mu_q_av[None, :, None, :]  # (1, 3_q, 1, n_pts)

                f_laser_q_key = jnp.real(s * gamma * m * delE_q) / 2
                # f_laser_q_key: (3_s, 3_q, n_beams, [n_pts])

                f_laser_key = jnp.sum(f_laser_q_key, axis=1)  # (3_s, n_beams, [n_pts])
                f = f + jnp.sum(f_laser_key, axis=1)  # (3_s, [n_pts])

                f_laser_q[key] = f_laser_q_key
                f_laser[key] = f_laser_key

        # Are we including magnetic forces?
        if self.include_mag_forces:
            av_mu = self.observable(self.hamiltonian.mu, rho_mat)

            if rho_mat.ndim == 3:  # time series: r shape (3, n_pts)
                delB = jax.vmap(lambda ri: self.magField.gradField(ri))(jnp.real(r).T)
                # delB: (n_pts, 3, 3) -> vmap over time, contract i
                f_mag = jnp.real(jnp.einsum("it,tij->jt", av_mu, delB))
            else:
                delB = self.magField.gradField(jnp.real(r))
                f_mag = jnp.real(jnp.einsum("...i,...ij->...j", av_mu, delB))

            f += f_mag

        if return_details:
            return f, f_laser, f_laser_q, f_mag
        else:
            return f

    def find_equilibrium_force(
        self,
        deltat=500,
        itermax=100,
        Npts=5001,
        rel=1e-5,
        abs=1e-9,
        debug=False,
        initial_rho="rateeq",
        return_details=False,
        **kwargs,
    ):
        r"""
        Find the equilibrium force at the initial position.

        This method works by solving the OBEs in a chunk of time
        $\\Delta T$, calculating the force during that chunk, continuing
        the integration for another chunk, calculating the force during that
        subsequent chunk, and comparing the average of the forces of the two
        chunks to see if they have converged.

        Parameters
        ----------
        deltat : float, optional
            Chunk time $\\Delta T$ to integrate over before checking for convergence. Default: 500.
        itermax : int, optional
            Maximum number of chunk iterations. Default: 100.
        Npts : int, optional
            Number of points to divide the chunk into. Default: 5001.
        rel : float, optional
            Relative convergence parameter. Default: 1e-5.
        abs : float, optional
            Absolute convergence parameter. Default: 1e-9.
        debug : boolean, optional
            If true, prints out debug information and stores piecewise solutions
            in `self.piecewise_sols` as the integration proceeds. Default: False.
        initial_rho : {'rateeq', 'equally', 'frompops'}, optional
            Determines how to set the initial density matrix $\\rho$ at the start
            of the calculation. Default: 'rateeq'.
        return_details : boolean, optional
            If true, returns the detailed components of the force (laser, magnetic,
            polarization projections) and equilibrium populations. Default: False.

        Other Parameters
        ----------------
        init_pop : array_like, optional
            Initial populations to use if `initial_rho='frompops'`.
        **kwargs :
            Additional keyword arguments are passed directly to `evolve_density`
            (e.g., `rtol` and `atol` for the ODE solver).

        Returns
        -------
        F : jax.Array, shape (3,)
            Total equilibrium force experienced by the atom.
        F_laser : dict of jax.Array
            (Returned if `return_details=True`) The forces due to each laser, indexed
            by the manifold the laser addresses.
        F_laser_q : dict of jax.Array
            (Returned if `return_details=True`) The forces due to each laser and its
            q component, indexed by the manifold the laser addresses.
        F_mag : jax.Array, shape (3,)
            (Returned if `return_details=True`) The forces due to the magnetic field.
        Neq : jax.Array
            (Returned if `return_details=True`) The equilibrium populations.
        ii : int
            (Returned if `return_details=True`) Number of iterations needed to converge.
        """
        if initial_rho == "rateeq":
            self.set_initial_rho_from_rateeq()
        elif initial_rho == "equally":
            self.set_initial_rho_equally()
        elif initial_rho == "frompops":
            Npop = kwargs.pop("init_pop", None)
            self.set_initial_rho_from_populations(Npop)
        else:
            raise ValueError("Argument initial_rho=%s not understood" % initial_rho)

        old_f_avg = jnp.full((3,), jnp.inf)

        if debug:
            print(
                "Finding equilibrium force at "
                + "r=(%.2f, %.2f, %.2f) " % (self.r0[0], self.r0[1], self.r0[2])
                + "v=(%.2f, %.2f, %.2f) " % (self.v0[0], self.v0[1], self.v0[2])
                + "with deltat = %.2f, itermax = %d, Npts = %d, " % (deltat, itermax, Npts)
                + "rel = %.1e and abs = %.1e" % (rel, abs)
            )
            self.piecewise_sols = []

        ii = 0
        while True:
            # Build single-atom state vector [rho_flat, v, r] and wrap as batch
            y0 = jnp.concatenate([self.rho0, self.v0, self.r0])
            y0_batch = y0[None, :]  # shape (1, state_dim)

            self.evolve_density(
                [ii * deltat, (ii + 1) * deltat],
                y0_batch,
                n_points=int(Npts),
                **kwargs,
            )

            # Extract atom 0: self.sol.y shape (1, n_points, state_dim)
            y_atom = self.sol.y[0]  # (n_points, state_dim)
            rho_flat = y_atom[:, :-6].T  # (n^2, n_points)
            r = jnp.real(y_atom[:, -3:]).T  # (3, n_points)

            f, f_laser, f_laser_q, f_mag = self.force(r, self.sol.t, rho_flat, return_details=True)
            f_avg = jnp.mean(f, axis=1)  # (3,)

            if debug:
                print(ii, f_avg, jnp.sum(f_avg**2))
                self.piecewise_sols.append(self.sol)

            f_sq = jnp.sum(f_avg**2)
            diff_sq = jnp.sum((old_f_avg - f_avg) ** 2)
            converged = bool((f_sq < abs) or (diff_sq / jnp.maximum(f_sq, 1e-30) < rel))
            if converged or ii >= itermax - 1:
                break

            old_f_avg = f_avg
            # Seed next chunk: rho_flat[:, -1] is already in the correct flat
            # representation (re/im if transform_into_re_im, complex otherwise)
            self.set_initial_rho(rho_flat[:, -1])
            self.set_initial_position_and_velocity(r[:, -1], jnp.real(y_atom[-1, -6:-3]))
            ii += 1

        if return_details:
            f_laser_avg = {key: jnp.mean(f_laser[key], axis=2) for key in f_laser}
            f_laser_avg_q = {key: jnp.mean(f_laser_q[key], axis=3) for key in f_laser_q}
            f_mag_avg = jnp.mean(f_mag, axis=1)
            rho_mat = self.__reshape_rho(rho_flat)  # (n, n, n_points)
            Neq = jnp.real(jnp.diagonal(jnp.mean(rho_mat, axis=2)))
            return (f_avg, f_laser_avg, f_laser_avg_q, f_mag_avg, Neq, ii)
        else:
            return f_avg

    def generate_force_profile(self, R, V, name=None, **kwargs):
        r"""
        Map out the equilibrium force vs. position and velocity using batched JAX integration.

        This method solves the Optical Bloch Equations (OBEs) simultaneously across a
        grid of initial positions and velocities. It integrates the evolution in chunks
        of time, comparing the time-averaged force of successive chunks until the force
        has converged for all grid points.

        Parameters
        ----------
        R : array_like, shape(3, ...)
            Position vector. First dimension of the array must be length 3, and
            corresponds to :math:`x`, :math:`y`, and :math:`z` components,
            respectively.
        V : array_like, shape(3, ...)
            Velocity vector. First dimension of the array must be length 3, and
            corresponds to :math:`v_x`, :math:`v_y`, and :math:`v_z` components,
            respectively.
        name : str, optional
            Name for the profile. Stored in the profile dictionary in this object.
            If None, uses the next integer, cast as a string, (i.e., '0') as
            the name.

        Other Parameters
        ----------------
        deltat : float, optional
            Chunk time :math:`\\Delta T` to integrate over before checking for convergence.
            Default: 500.
        itermax : int, optional
            Maximum number of chunk iterations to perform. Default: 100.
        Npts : int, optional
            Number of points to divide the integration chunk into. Default: 5001.
        rel : float, optional
            Relative convergence parameter. Default: 1e-5.
        abs : float, optional
            Absolute convergence parameter. Default: 1e-9.
        npts_conv_divisor : int, optional
            ``Npts`` is divided by this value to get the number of output
            points used during the convergence loop (``Npts_conv``).  A
            larger value means fewer points per convergence chunk and faster
            iterations, but noisier force estimates.  Set to 1 to use the
            full ``Npts`` resolution for convergence (smoothest profiles,
            slowest).  Default: 10.
        initial_rho : {'rateeq', 'equally'}, optional
            Determines how to set the initial density matrix :math:`\\rho` at the start
            of the calculation. Default: 'rateeq'.
        deltat_r : float, optional
            Dynamic deltat scaling factor based on spatial position.
        deltat_v : float, optional
            Dynamic deltat scaling factor based on velocity.
        deltat_tmax : float, optional
            Maximum allowed deltat if dynamically calculated via `deltat_r` or `deltat_v`.
            Default: np.inf.
        deltat_func : callable, optional
            A custom function `f(r, v)` to dynamically determine the chunk time `deltat`
            for each grid point. The method will use the minimum valid deltat returned
            across the grid to ensure alignment.

        Returns
        -------
        profile : pylcp_jax.obe.force_profile
            Resulting force profile containing the equilibrium forces, detailed laser/mag
            forces, and equilibrium populations for the specified grid.
        """
        # Pin all JAX operations to the CPU device so that large
        # grids do not exhaust GPU memory.
        cpu_device = jax.devices("cpu")[0]

        _cpu_ctx = jax.default_device(cpu_device)
        _cpu_ctx.__enter__()

        # Pop deltat-shaping kwargs
        deltat_r = kwargs.pop("deltat_r", None)
        deltat_v = kwargs.pop("deltat_v", None)
        deltat_tmax = kwargs.pop("deltat_tmax", np.inf)
        deltat_func = kwargs.pop("deltat_func", None)
        kwargs.pop("return_details", None)  # always True here

        # Pop find_equilibrium_force kwargs so they don't leak into evolve_density
        chunk_deltat = kwargs.pop("deltat", 500)
        itermax = kwargs.pop("itermax", 100)
        Npts = kwargs.pop("Npts", 5001)
        rel = kwargs.pop("rel", 1e-5)
        abs_tol = kwargs.pop("abs", 1e-9)
        npts_conv_divisor = kwargs.pop("npts_conv_divisor", 10)
        initial_rho = kwargs.pop("initial_rho", "rateeq")
        progress_bar = kwargs.pop("progress_bar", False)

        if not name:
            name = "{0:d}".format(len(self.profile))

        self.profile[name] = force_profile(R, V, self.laserBeams, self.hamiltonian)

        # Flatten the position/velocity grid: (3, N)
        R_np = np.array(R).reshape(3, -1)
        V_np = np.array(V).reshape(3, -1)
        N = R_np.shape[1]

        # Build initial rho for every atom
        rho0_list = []
        for i in range(N):
            self.set_initial_position_and_velocity(R_np[:, i], V_np[:, i])
            if initial_rho == "rateeq":
                self.set_initial_rho_from_rateeq()
            elif initial_rho == "equally":
                self.set_initial_rho_equally()
            else:
                raise ValueError(
                    f"initial_rho={initial_rho!r} not supported in generate_force_profile"
                )
            rho0_list.append(self.rho0)

        rho0_batch = jnp.stack(rho0_list)  # (N, n²)
        V_jnp = jnp.asarray(V_np.T)  # (N, 3)
        R_jnp = jnp.asarray(R_np.T)  # (N, 3)
        y0_batch = jnp.concatenate([rho0_batch, V_jnp, R_jnp], axis=1)  # (N, state_dim)

        # Group atoms by their ideal chunk_deltat so that slow atoms get long
        # chunks (fast convergence) and fast atoms get short chunks (accuracy).
        # This avoids forcing ALL atoms to use the smallest chunk_deltat.
        per_atom_deltat = np.full(N, chunk_deltat, dtype=float)
        if deltat_func is not None:
            for i in range(N):
                d = deltat_func(R_np[:, i], V_np[:, i])
                if d is not None:
                    per_atom_deltat[i] = float(d)
        elif deltat_v is not None or deltat_r is not None:
            for i in range(N):
                r_i, v_i = R_np[:, i], V_np[:, i]
                d = None
                if deltat_v is not None:
                    vabs = np.sqrt(np.sum(v_i**2))
                    d = (
                        float(deltat_tmax)
                        if vabs == 0
                        else min(2 * np.pi * deltat_v / vabs, float(deltat_tmax))
                    )
                if deltat_r is not None:
                    rabs = np.sqrt(np.sum(r_i**2))
                    d_r = (
                        float(deltat_tmax)
                        if rabs == 0
                        else min(2 * np.pi * deltat_r / rabs, float(deltat_tmax))
                    )
                    d = d_r if d is None else min(d, d_r)
                if d is not None:
                    per_atom_deltat[i] = d

        # Group atoms by chunk_deltat; atoms with the same dt are batched.
        if deltat_v is not None or deltat_func is not None:
            rounded_deltat = np.round(per_atom_deltat, decimals=6)
            unique_deltats = np.unique(rounded_deltat)
            groups = []
            for dt in unique_deltats:
                indices = np.where(np.abs(rounded_deltat - dt) < 1e-9)[0]
                groups.append((indices, float(dt)))
        else:
            # deltat_r only (or bare deltat): single batch, shortest chunk
            shared_dt = float(np.min(per_atom_deltat))
            groups = [(np.arange(N), shared_dt)]

        # Allocate per-atom result storage
        n_rho = rho0_batch.shape[1]
        final_f_avg = np.zeros((N, 3))
        final_rho_flat_mean = np.zeros((N, n_rho))
        final_f_mag_avg = np.zeros((N, 3))
        final_f_laser_avg = None
        final_f_laser_q_avg = None
        final_iters = np.zeros(N, dtype=int)

        if progress_bar:
            progress = progressBar()
            atoms_done = 0

        # Single max_steps across groups avoids repeated JIT recompilation.
        max_group_deltat = max(dt for _, dt in groups)
        fixed_max_steps = max(int(np.ceil(max_group_deltat * 16)), 4096)

        for gi, (group_indices, group_chunk_deltat) in enumerate(groups):
            Ng = len(group_indices)
            y0_group = y0_batch[group_indices]  # (Ng, state_dim)

            # Coarse grid for convergence; full Npts on final pass.
            Npts_conv = max(int(Npts) // max(npts_conv_divisor, 1), 101)

            # Per-atom convergence via consecutive chunk comparison.
            # was_decreasing guards against false convergence during dark-state decay.
            old_f_chunk = jnp.full((Ng, 3), jnp.inf)
            old_f_sq = jnp.zeros(Ng)
            was_decreasing = jnp.zeros(Ng, dtype=bool)
            atom_converged = jnp.zeros(Ng, dtype=bool)
            converged_f = jnp.zeros((Ng, 3))
            converged_rho = jnp.zeros((Ng, n_rho))

            ii = 0
            while True:
                self.evolve_density(
                    [ii * group_chunk_deltat, (ii + 1) * group_chunk_deltat],
                    y0_group,
                    n_points=Npts_conv,
                    max_steps=fixed_max_steps,
                    **kwargs,
                )
                t = self.sol.t
                n_pts = t.shape[0]

                rho_flat_all = self.sol.y[:, :, :-6].transpose(0, 2, 1)
                r_all = jnp.real(self.sol.y[:, :, -3:]).transpose(0, 2, 1)

                f_all = jax.vmap(
                    lambda r_i, rho_i: self.force(r_i, t, rho_i, return_details=False)
                )(r_all, rho_flat_all)

                f_chunk = jnp.sum(f_all, axis=2) / n_pts  # type: ignore[arg-type]  # (Ng, 3)
                rho_chunk = jnp.sum(rho_flat_all, axis=2) / n_pts  # (Ng, n_rho)

                f_sq = jnp.sum(f_chunk**2, axis=1)
                diff_sq = jnp.sum((old_f_chunk - f_chunk) ** 2, axis=1)

                # Track monotonic force decay (dark-state guard).
                is_decreasing = f_sq < old_f_sq
                was_decreasing = was_decreasing | is_decreasing

                # Block abs_tol criterion while force is still decaying.
                newly = ~atom_converged & (
                    (f_sq < abs_tol)
                    | (diff_sq / jnp.maximum(f_sq, 1e-30) < rel)
                    | (~was_decreasing & (diff_sq < abs_tol))
                )
                converged_f = jnp.where(newly[:, None], f_chunk, converged_f)  # type: ignore[arg-type]
                converged_rho = jnp.where(newly[:, None], rho_chunk, converged_rho)  # type: ignore[arg-type]
                atom_converged = atom_converged | newly

                if bool(jnp.all(atom_converged)) or ii >= itermax - 1:
                    break

                old_f_chunk = f_chunk
                old_f_sq = f_sq
                y0_group = self.sol.y[:, -1, :]
                ii += 1

            # Atoms that hit itermax without converging fall back to last chunk.
            f_conv = jnp.where(atom_converged[:, None], converged_f, f_chunk)
            rho_conv = jnp.where(atom_converged[:, None], converged_rho, rho_chunk)

            # Final pass at full Npts with return_details=True.
            self.evolve_density(
                [ii * group_chunk_deltat, (ii + 1) * group_chunk_deltat],
                y0_group,
                n_points=int(Npts),
                max_steps=fixed_max_steps,
                **kwargs,
            )
            t = self.sol.t
            n_pts = t.shape[0]
            rho_flat_all = self.sol.y[:, :, :-6].transpose(0, 2, 1)
            r_all = jnp.real(self.sol.y[:, :, -3:]).transpose(0, 2, 1)

            f_all, f_laser_all, f_laser_q_all, f_mag_all = jax.vmap(
                lambda r_i, rho_i: self.force(r_i, t, rho_i, return_details=True)
            )(r_all, rho_flat_all)

            # Use the convergence estimate for f_avg so the convergence test
            # and the reported force are consistent.
            f_avg_np = np.array(f_conv)
            rho_flat_mean = np.array(rho_conv)
            f_mag_avg_np = np.array(jnp.sum(f_mag_all, axis=2) / n_pts)

            for local_idx, global_idx in enumerate(group_indices):
                final_f_avg[global_idx] = f_avg_np[local_idx]
                final_rho_flat_mean[global_idx] = rho_flat_mean[local_idx]
                final_f_mag_avg[global_idx] = f_mag_avg_np[local_idx]
                final_iters[global_idx] = ii

            # Per-beam/per-q averages from the final (converged) chunk only
            f_laser_group = {
                k: np.array(jnp.sum(v, axis=-1) / n_pts) for k, v in f_laser_all.items()
            }
            f_laser_q_group = {
                k: np.array(jnp.sum(v, axis=-1) / n_pts) for k, v in f_laser_q_all.items()
            }
            if final_f_laser_avg is None:
                final_f_laser_avg = {
                    k: np.zeros((N,) + v.shape[1:]) for k, v in f_laser_group.items()
                }
                final_f_laser_q_avg = {
                    k: np.zeros((N,) + v.shape[1:]) for k, v in f_laser_q_group.items()
                }
            for k in f_laser_group:
                for local_idx, global_idx in enumerate(group_indices):
                    final_f_laser_avg[k][global_idx] = f_laser_group[k][local_idx]  # type: ignore[index]
                    final_f_laser_q_avg[k][global_idx] = f_laser_q_group[k][  # pyright: ignore[reportOptionalSubscript]
                        local_idx
                    ]

            # Free large JAX arrays to keep memory bounded across groups.
            del (
                rho_flat_all,
                r_all,
                f_all,
                f_laser_all,
                f_laser_q_all,
                f_mag_all,
            )
            gc.collect()

            if progress_bar:
                atoms_done += Ng
                progress.update(atoms_done / N)

        if progress_bar:
            progress.update(1.0)

        # Convert final arrays back to jnp for downstream compatibility
        f_avg = jnp.array(final_f_avg)
        assert final_f_laser_avg is not None
        assert final_f_laser_q_avg is not None
        f_laser_avg = {k: jnp.array(v) for k, v in final_f_laser_avg.items()}
        f_laser_avg_q = {k: jnp.array(v) for k, v in final_f_laser_q_avg.items()}
        f_mag_avg = jnp.array(final_f_mag_avg)

        # Equilibrium populations: cumulative mean rho → diagonal
        rho_flat_mean_jnp = jnp.array(final_rho_flat_mean)

        def get_Neq(rho_flat_i):
            """Extract diagonal populations from a flat density-matrix vector."""
            return jnp.real(jnp.diagonal(self.__reshape_rho(rho_flat_i)))

        Neq_all = jax.vmap(get_Neq)(rho_flat_mean_jnp)  # (N, n)

        # Write results back into the force_profile using nditer for multi-index
        it = np.nditer(
            [R[0], R[1], R[2], V[0], V[1], V[2]],
            flags=["refs_ok", "multi_index"],
            op_flags=[["readonly"]] * 6,  # type: ignore[arg-type]
        )
        for atom_idx, _ in enumerate(it):
            mi = it.multi_index
            self.profile[name].store_data(
                mi,
                Neq_all[atom_idx],
                f_avg[atom_idx],
                {key: f_laser_avg[key][atom_idx] for key in f_laser_avg},
                f_mag_avg[atom_idx],
                int(final_iters[atom_idx]),
                {key: f_laser_avg_q[key][atom_idx] for key in f_laser_avg_q},
            )

        _cpu_ctx.__exit__(None, None, None)
        return self.profile[name]
