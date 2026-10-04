"""
Heuristic force equation for laser cooling simulations.

Treats the atom as a single F=0 -> F'=1 cycling transition and computes
the scattering force from per-beam saturation parameters projected onto
the local magnetic field axis.  No internal-state dynamics are evolved,
so the force is instantaneous at every point in phase space.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.integrate import solve_ivp as _scipy_solve_ivp

from .common import base_force_profile
from .governingeq import governingeq
from .integration_tools_gpu import solve_ivp_random as solve_ivp_random_gpu


class heuristiceq(governingeq):
    """
    Heuristic force equation (JAX/GPU implementation).

    The heuristic equation governs the atom or molecule as if it has a single
    transition between an :math:`F=0` ground state to an :math:`F'=1` excited
    state.

    Parameters
    ----------
    laserBeams : dictionary of pylcp_jax.laserBeams, pylcp_jax.laserBeams, or list of pylcp_jax.laserBeam
        The laserBeams that will be used in constructing the heuristic
        equations.  Must contain only a single key of ``'g->e'``.
    magField : pylcp_jax.magField or callable
        The function or object that defines the magnetic field.
    a : array_like, shape (3,), optional
        Constant acceleration (e.g. gravity). Default: [0., 0., 0.]
    mass : float, optional
        Mass of the atom or molecule. Default: 100
    gamma : float, optional
        Decay rate of the single transition. Default: 1
    k : float, optional
        Magnitude of the k vector for the transition. Default: 1
    r0 : array_like, shape (3,), optional
        Initial position. Default: [0., 0., 0.]
    v0 : array_like, shape (3,), optional
        Initial velocity. Default: [0., 0., 0.]
    """

    def __init__(
        self,
        laserBeams,
        magField,
        a=jnp.array([0.0, 0.0, 0.0]),
        mass=100,
        gamma=1,
        k=1,
        r0=jnp.array([0.0, 0.0, 0.0]),
        v0=jnp.array([0.0, 0.0, 0.0]),
    ):
        super().__init__(laserBeams, magField, a=a, r0=r0, v0=v0)

        for key in self.laserBeams:
            if key != "g->e":
                raise KeyError(
                    "laserBeam dictionary should only contain "
                    "a single key of 'g->e' for the heuristiceq."
                )

        self.mass = float(mass)
        self.gamma = float(gamma)
        self.k = float(k)

    def scattering_rate(self, r, v, t, return_kvecs=False):
        """
        Calculate the scattering rate for each laser beam.

        Parameters
        ----------
        r : array_like, shape (3,)
            Position.
        v : array_like, shape (3,)
            Velocity.
        t : float
            Time.
        return_kvecs : bool, optional
            If True, also return the k-vectors. Default: False.

        Returns
        -------
        R : jax.Array, shape (n_beams,)
            Scattering rate for each beam.
        kvecs : jax.Array, shape (n_beams, 3)
            Only returned if ``return_kvecs=True``.
        """
        B = self.magField.Field(r, t)
        Bmag = jnp.linalg.norm(B)
        Bhat = jnp.where(
            Bmag > 1e-10,
            B / Bmag,
            jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
        )

        kvecs = self.laserBeams["g->e"].kvec(r, t)  # (n_beams, 3)
        intensities = self.laserBeams["g->e"].intensity(r, t)  # (n_beams,)
        pols = self.laserBeams["g->e"].project_pol(Bhat, r, t)  # (n_beams, 3)
        deltas = self.laserBeams["g->e"].delta(t)  # (n_beams,)

        totintensity = jnp.sum(intensities)
        q_vals = jnp.array([-1.0, 0.0, 1.0])

        polsqrd = jnp.abs(pols) ** 2  # (n_beams, 3)

        # Doppler shift per beam: k·v, shape (n_beams,)
        kdotv = jnp.einsum("bi,i->b", kvecs, v)

        # Detuning for each beam and each q: (n_beams, 3)
        det = deltas[:, None] - kdotv[:, None] - q_vals[None, :] * Bmag

        # Scattering rate summed over q components: (n_beams,)
        R = jnp.sum(
            self.gamma
            / 2
            * intensities[:, None]
            * polsqrd
            / (1 + totintensity + 4 * det**2 / self.gamma**2),
            axis=1,
        )

        if return_kvecs:
            return R, kvecs
        return R

    def force(self, r, v, t):
        """
        Calculate the instantaneous force.

        Parameters
        ----------
        r : array_like, shape (3,)
            Position.
        v : array_like, shape (3,)
            Velocity.
        t : float
            Time.

        Returns
        -------
        F : jax.Array, shape (3,)
            Total force on the atom.
        F_laser : dict
            Per-beam force contributions, keyed by transition.
            ``F_laser['g->e']`` has shape ``(3, n_beams)``.
        """
        R, kvecs = self.scattering_rate(r, v, t, return_kvecs=True)
        # kvecs: (n_beams, 3),  R: (n_beams,)
        F_laser_ge = (kvecs * R[:, None]).T  # (3, n_beams)
        F = jnp.sum(F_laser_ge, axis=1)  # (3,)
        return F, {"g->e": F_laser_ge}

    def evolve_motion(self, t_span, freeze_axis=[False, False, False], **kwargs):
        """
        Evolve the motion of a single atom using scipy.

        Uses ``self.r0`` and ``self.v0`` as initial conditions (set via
        :meth:`set_initial_position_and_velocity`).  Delegates to
        :func:`scipy.integrate.solve_ivp`, so all scipy kwargs are supported,
        including ``events`` for early termination and ``max_step``.

        Parameters
        ----------
        t_span : tuple
            ``(t0, tf)`` integration interval.
        freeze_axis : list of bool
            Freeze motion along the specified axes. Default: [False, False, False]
        **kwargs
            Passed to :func:`scipy.integrate.solve_ivp` (e.g. ``events``,
            ``max_step``, ``method``, ``dense_output``).

        Returns
        -------
        sol : scipy OdeResult
            Also stored in ``self.sol``.  Has attributes ``t``, ``y``, ``r``,
            ``v``, and (if events were given) ``t_events``, ``y_events``.
        """
        free_axes = np.array([not f for f in freeze_axis], dtype=np.float64)
        mass = self.mass
        constant_accel = np.array(self.constant_accel)

        # Cache JIT-compiled force on the instance so it's only traced once.
        # Safe as long as beam parameters don't change on the same instance.
        # If you mutate beam parameters, delete self._jit_force first.
        if not hasattr(self, "_jit_force"):

            @jax.jit
            def _jit_force(r, v, t):
                return self.force(r, v, t)

            self._jit_force = _jit_force

        _jit_force = self._jit_force

        def dydt(t, y):
            """ODE RHS for a single atom trajectory (scipy interface)."""
            v = y[:3]
            r = y[3:6]
            F, _ = _jit_force(jnp.asarray(r), jnp.asarray(v), jnp.float64(t))
            F = np.asarray(F)
            dvdt = F / mass * free_axes + constant_accel
            return np.concatenate([dvdt, v])

        y0 = np.concatenate([np.asarray(self.v0), np.asarray(self.r0)])

        self.sol = _scipy_solve_ivp(fun=dydt, t_span=t_span, y0=y0, **kwargs)

        self.sol.v = self.sol.y[:3]
        self.sol.r = self.sol.y[3:]

        return self.sol

    def evolve_motion_batch(
        self,
        t_span,
        y0_batch=None,
        keys_batch=None,
        freeze_axis=[False, False, False],
        random_recoil=False,
        max_scatter_probability=0.1,
        **kwargs,
    ):
        """
        Evolve the motion of many atoms in parallel using JAX/diffrax.

        State vector layout per atom: ``[v (3), r (3)]`` → shape ``(6,)``.

        Parameters
        ----------
        t_span : tuple
            ``(t0, tf)`` integration interval.
        y0_batch : jax.Array, shape (N, 6)
            Initial conditions ``[v, r]`` for N atoms.
        keys_batch : jax.Array, shape (N, ...), optional
            JAX PRNG keys, one per atom.  If None, keys are generated
            automatically from a random seed.
        freeze_axis : list of bool
            Freeze motion along the specified axes. Default: [False, False, False]
        random_recoil : bool
            Apply stochastic photon recoil kicks. Default: False
        max_scatter_probability : float
            Maximum scattering probability per step. Default: 0.1
        **kwargs
            Passed to :func:`solve_ivp_random` (e.g. ``rtol``, ``atol``,
            ``max_steps``, ``solver_type``, ``n_points``).

        Returns
        -------
        sols : list of RandomOdeResult
            One result per atom with ``t``, ``y``, ``r``, ``v``.
        """
        if y0_batch is None:
            y0 = jnp.concatenate([self.v0, self.r0])
            y0_batch = y0[jnp.newaxis, :]
        if keys_batch is None:
            keys_batch = jax.random.split(
                jax.random.PRNGKey(np.random.randint(0, 2**31)),
                y0_batch.shape[0],
            )

        free_axes = jnp.asarray([not f for f in freeze_axis], dtype=jnp.float64)
        mass = self.mass
        constant_accel = self.constant_accel

        def dydt(t, y):
            """ODE RHS for batched atom trajectories (JAX/diffrax interface)."""
            v = y[:3]
            r = y[3:6]
            F, _ = self.force(r, v, t)
            dvdt = F / mass * free_axes + constant_accel
            return jnp.concatenate([dvdt, v])

        def _random_unit_vector(key):
            """Return a random unit vector restricted to the free axes."""
            key_phi, key_z = jax.random.split(key)
            phi = 2.0 * jnp.pi * jax.random.uniform(key_phi)
            z = 2.0 * jax.random.uniform(key_z) - 1.0
            r_xy = jnp.sqrt(1.0 - z**2)
            return jnp.array([r_xy * jnp.cos(phi), r_xy * jnp.sin(phi), z]) * free_axes

        def random_recoil_fn(t, y, dt, key, args=None):
            """Apply a stochastic photon recoil kick if a scattering event occurs."""
            v = y[:3]
            r = y[3:6]
            R_rates = self.scattering_rate(r, v, t)
            total_P = jnp.sum(R_rates) * dt  # type: ignore[arg-type]

            key, key_dice, key_v1, key_v2 = jax.random.split(key, 4)
            did_scatter = jax.random.uniform(key_dice) < total_P

            kick = self.k / mass * (_random_unit_vector(key_v1) + _random_unit_vector(key_v2))
            y_jump = jnp.where(did_scatter, y.at[:3].add(kick), y)
            n_scatters = jnp.where(did_scatter, 1, 0)
            new_dt_max = jnp.where(total_P > 0, max_scatter_probability / total_P * dt, jnp.inf)
            return y_jump, n_scatters, new_dt_max, key

        def dummy_recoil(t, y, dt, key, args=None):
            """No-op recoil function used when random_recoil=False."""
            return y, jnp.int32(0), jnp.inf, key

        random_func = random_recoil_fn if random_recoil else dummy_recoil

        self.sols = solve_ivp_random_gpu(
            fun=dydt,
            random_func=random_func,
            t_span=t_span,
            y0_batch=jnp.asarray(y0_batch),
            keys_batch=jnp.asarray(keys_batch),
            **kwargs,
        )

        # Attach r and v as named attributes for convenience (y is (state_dim, n_steps))
        for sol in self.sols:
            assert sol.y is not None
            sol.v = sol.y[:3]
            sol.r = sol.y[3:]

        return self.sols

    def find_equilibrium_force(self, return_details=False):
        """
        Find the equilibrium force at the initial position and velocity.

        Since the heuristic force is instantaneous (no internal state to
        converge), this is a direct evaluation of :meth:`force`.

        Parameters
        ----------
        return_details : bool, optional
            If True, also return per-beam forces and scattering rates.

        Returns
        -------
        F : jax.Array, shape (3,)
            Total force.
        F_laser : dict
            Only if ``return_details=True``. Per-beam forces.
        R : jax.Array, shape (n_beams,)
            Only if ``return_details=True``. Scattering rates.
        """
        F, F_laser = self.force(self.r0, self.v0, t=0.0)
        if return_details:
            R_rates = self.scattering_rate(self.r0, self.v0, t=0.0)
            return F, F_laser, R_rates
        return F

    def generate_force_profile(self, R, V, name=None, t=0.0):
        """
        Map out the equilibrium force vs. position and velocity.

        Since the heuristic force is instantaneous, all grid points are
        evaluated in a single :func:`jax.vmap` call (no convergence loop).

        Parameters
        ----------
        R : array_like, shape (3, ...)
            Position grid. First dimension must be 3.
        V : array_like, shape (3, ...)
            Velocity grid. First dimension must be 3.
        name : str, optional
            Key under which to store the profile. Defaults to the current
            count of profiles as a string.
        t : float, optional
            Time at which to evaluate the force. Default: 0.

        Returns
        -------
        profile : pylcp_jax.common.base_force_profile
            Resulting force profile.
        """
        if not name:
            name = "{0:d}".format(len(self.profile))

        self.profile[name] = base_force_profile(R, V, self.laserBeams, None)

        grid_shape = np.asarray(R[0]).shape  # e.g. (Nx, Ny) or (N,)

        # Flatten position/velocity grids to (3, N) then transpose to (N, 3)
        R_jnp = jnp.asarray(R).reshape(3, -1).T  # (N, 3)
        V_jnp = jnp.asarray(V).reshape(3, -1).T  # (N, 3)

        # Evaluate force for all (r, v) pairs in parallel
        F_all, F_laser_all = jax.vmap(lambda r_i, v_i: self.force(r_i, v_i, t))(R_jnp, V_jnp)
        # F_all: (N, 3),  F_laser_all: {'g->e': (N, 3, n_beams)}

        # Write results back in bulk (reshape flat arrays to grid shape)
        self.profile[name].F = F_all.T.reshape((3,) + grid_shape)
        for key in F_laser_all:
            # F_laser_all[key]: (N, 3, n_beams) → (3, ..grid.., n_beams)
            f_flat = F_laser_all[key].reshape(grid_shape + (3, -1))
            self.profile[name].f[key] = jnp.moveaxis(f_flat, len(grid_shape), 0)

        return self.profile[name]
