"""
Base class for all governing equations in pylcp_jax.

Defines the common interface (initial conditions, force profiles, equilibrium
finding, trapping frequencies, damping coefficients) shared by the heuristic
equation, rate equations, and optical Bloch equations.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.optimize import root, root_scalar

from .fields import laserBeams as laserBeamsObject
from .fields import magField as magFieldObject


class governingeq(object):
    """
    Governing equation base class.

    This class is the basis for making all the governing equations in `pylcp_jax`,
    including the rate equations, heuristic equation, and the optical Bloch
    equations.  Its methods are available to other governing equations.

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
    hamiltonian : pylcp_jax.hamiltonian or None
        The internal hamiltonian of the particle.
    a : array_like, shape (3,), optional
        A default acceleration to apply to the particle's motion, usually
        gravity. Default: [0., 0., 0.]
    r0 : array_like, shape (3,)
        Initial position.  Default: [0.,0.,0.]
    v0 : array_like, shape (3,)
        Initial velocity.  Default: [0.,0.,0.]
    """

    def __init__(
        self,
        laserBeams: dict[str, laserBeamsObject] | laserBeamsObject | list[Any],
        magField: magFieldObject | npt.ArrayLike,
        hamiltonian: Any = None,
        a: npt.ArrayLike = jnp.array([0.0, 0.0, 0.0]),
        r0: npt.ArrayLike = jnp.array([0.0, 0.0, 0.0]),
        v0: npt.ArrayLike = jnp.array([0.0, 0.0, 0.0]),
    ) -> None:

        a = jnp.asarray(a, dtype=jnp.float64)  # cast to jax if not already given
        r0 = jnp.asarray(r0, dtype=jnp.float64)
        v0 = jnp.asarray(v0, dtype=jnp.float64)

        self.set_initial_position_and_velocity(r0, v0)

        # Normalise laserBeams into a dict keyed by transition label (e.g. 'g->e').
        # Lists and bare laserBeams objects default to the 'g->e' key.
        self.laserBeams = {}
        if isinstance(laserBeams, list):
            self.laserBeams["g->e"] = laserBeamsObject(laserBeams)
        elif isinstance(laserBeams, laserBeamsObject):
            self.laserBeams["g->e"] = copy.copy(laserBeams)
        elif isinstance(laserBeams, dict):
            for key in laserBeams.keys():
                if not isinstance(laserBeams[key], laserBeamsObject):
                    raise TypeError(
                        "Key %s in dictionary laserBeams " % key + "is not of type laserBeams."
                    )
            self.laserBeams = copy.copy(laserBeams)
        else:
            raise TypeError("laserBeams is not a valid type.")

        # Add in magnetic field:
        if callable(magField):
            self.magField = magFieldObject(magField)
        elif isinstance(magField, (np.ndarray, jnp.ndarray)):
            self.magField = magFieldObject(jnp.asarray(magField))
        elif isinstance(magField, magFieldObject):
            self.magField = copy.copy(magField)
        else:
            raise TypeError(
                "The magnetic field must be either a lambda " + "function or a magField object."
            )

        # Add the Hamiltonian:
        if hamiltonian is not None:
            self.hamiltonian = copy.copy(hamiltonian)
            self.hamiltonian.make_full_matrices()

            # Next, check to see if there is consistency in k:
            self.__check_consistency_in_lasers_and_d_q()

        # Check the acceleration:
        if a.size != 3:
            raise ValueError("Constant acceleration must have length 3.")
        self.constant_accel = a

        # Set up a dictionary to store any resulting force profiles.
        self.profile = {}

        # No solution computed yet:
        self.sol = None

        # Set an attribute for the equilibrium position:
        self.r_eq = None

    def __check_consistency_in_lasers_and_d_q(self):
        # Check that laser beam keys and Hamiltonian keys match.
        for laser_key in self.laserBeams.keys():
            if laser_key not in self.hamiltonian.laser_keys.keys():
                raise ValueError(
                    "laserBeams dictionary keys %s " % laser_key
                    + "does not have a corresponding key the "
                    + "Hamiltonian d_q."
                )

    def set_initial_position_and_velocity(self, r0: npt.ArrayLike, v0: npt.ArrayLike) -> None:
        """
        Set the initial position and velocity.

        Parameters
        ----------
        r0 : array_like, shape (3,)
            Initial position.  Default: [0.,0.,0.]
        v0 : array_like, shape (3,)
            Initial velocity.  Default: [0.,0.,0.]
        """
        self.set_initial_position(r0)
        self.set_initial_velocity(v0)

    def set_initial_position(self, r0: npt.ArrayLike) -> None:
        """
        Set the initial position.

        Parameters
        ----------
        r0 : array_like, shape (3,)
            Initial position.  Default: [0.,0.,0.]
        """
        self.r0 = jnp.asarray(r0, dtype=jnp.float64)
        self.sol = None

    def set_initial_velocity(self, v0: npt.ArrayLike) -> None:
        """
        Set the initial velocity.

        Parameters
        ----------
        v0 : array_like, shape (3,)
            Initial velocity.  Default: [0.,0.,0.]
        """
        self.v0 = jnp.asarray(v0, dtype=jnp.float64)
        self.sol = None

    def evolve_motion(self):
        """Evolve the equations of motion (implemented by subclasses)."""
        pass

    def find_equilibrium_force(self):
        """
        Find the equilibrium force at the initial conditions.

        Returns
        -------
        force : array_like
            Equilibrium force experienced by the atom
        """
        pass

    def force(self):
        """
        Find the instantaneous force.

        Returns
        -------
        force : array_like
            Force experienced by the atom
        """
        pass

    def generate_force_profile(self):
        """
        Map out the equilibrium force vs. position and velocity.

        Parameters
        ----------
        R : array_like, shape(3, ...)
            Position vector.  First dimension of the array must be length 3, and
            corresponds to :math:`x`, :math:`y`, and :math:`z` components,
            respectively.
        V : array_like, shape(3, ...)
            Velocity vector.  First dimension of the array must be length 3, and
            corresponds to :math:`v_x`, :math:`v_y`, and :math:`v_z` components,
            respectively.
        name : str, optional
            Name for the profile.  Stored in profile dictionary in this object.
            If None, uses the next integer, cast as a string, (i.e., '0') as
            the name.
        progress_bar : boolean, optional
            Displays a progress bar as the proceeds.  Default: False

        Returns
        -------
        profile : pylcp_jax.common.base_force_profile
            Resulting force profile.
        """
        pass

    def find_equilibrium_position(self, axes: Sequence[int], **kwargs: Any) -> jax.Array:
        r"""
        Find the equilibrium position.

        Uses the find_equilibrium force() method to calculate the where the
        :math:`\\mathbf{f}(\\mathbf{r}, \\mathbf{v}=0)=0`.

        Parameters
        ----------
        axes : array_like
            A list of axis indices to compute the trapping frequencies along.
            Here, :math:`\\hat{x}` is index 0, :math:`\\hat{y}` is index 1, and
            :math:`\\hat{z}` is index 2.  For example, `axes=[2]` calculates
            the trapping frequency along :math:`\\hat{z}`.
        kwargs :
            Any additional keyword arguments to pass to find_equilibrium_force()

        Returns
        -------
        r_eq : list or float
            The equilibrium positions along the selected axes.
        """
        if self.r_eq is None:
            self.r_eq = jnp.zeros((3,), dtype=jnp.float64)

        def _set_and_eval(r_list):
            r_wrap = jnp.array(r_list, dtype=jnp.float64)
            self.set_initial_position_and_velocity(r_wrap, jnp.zeros(3))
            return self.find_equilibrium_force()

        if len(axes) > 1:

            def multi_wrapper(r_changing):
                r_vals = list(np.array(self.r_eq, dtype=float))
                for i, ax in enumerate(axes):
                    r_vals[int(ax)] = float(r_changing[i])
                F = _set_and_eval(r_vals)
                assert F is not None
                return np.array([float(F[int(ax)]) for ax in axes])

            result = root(multi_wrapper, **kwargs)
            for i, ax in enumerate(axes):
                self.r_eq = self.r_eq.at[int(ax)].set(float(result.x[i]))
        else:
            ax0 = int(axes[0])

            def scalar_wrapper(r_changing):
                r_vals = list(np.array(self.r_eq, dtype=float))
                r_vals[ax0] = float(r_changing)
                F = _set_and_eval(r_vals)
                assert F is not None
                return float(F[ax0])

            result = root_scalar(scalar_wrapper, **kwargs)
            self.r_eq = self.r_eq.at[ax0].set(float(result.root))

        return self.r_eq

    def trapping_frequencies(
        self,
        axes: Sequence[int],
        r: npt.ArrayLike | None = None,
        eps: float = 0.01,
        **kwargs: Any,
    ) -> jax.Array | float:
        r"""
        Find the trapping frequency.

        Uses the find_equilibrium force() method to calculate the trapping
        frequency for the particular configuration.

        Parameters
        ----------
        axes : array_like
            A list of axis indices to compute the trapping frequencies along.
            Here, :math:`\\hat{x}` is index 0, :math:`\\hat{y}` is index 1, and
            :math:`\\hat{z}` is index 2.  For example, `axes=[2]` calculates
            the trapping frequency along :math:`\\hat{z}`.
        r : array_like, optional
            The position at which to calculate the damping coefficient.  By
            default r=None, which defaults to calculating at the equilibrium
            position as found by the find_equilibrium_position() method.  If
            this method has not been run, it defaults to the origin.
        eps : float, optional
            The small numerical :math:`\\epsilon` parameter used for calculating
            the :math:`df/dr` derivative.  Default: 0.01
        kwargs :
            Any additional keyword arguments to pass to find_equilibrium_force()

        Returns
        -------
        omega : list or float
            The trapping frequencies along the selected axes.
        """
        self.omega = jnp.zeros(
            3,
        )

        eps_arr: jax.Array = jnp.array([eps] * 3) if isinstance(eps, float) else jnp.asarray(eps)

        if r is None and self.r_eq is None:
            r = jnp.array([0.0, 0.0, 0.0])
        elif r is None:
            r = self.r_eq if self.r_eq is not None else jnp.array([0.0, 0.0, 0.0])

        assert r is not None
        r_arr: jax.Array = jnp.asarray(r)

        mass = getattr(self, "mass", None) or self.hamiltonian.mass

        for axis in axes:
            if not jnp.isnan(r_arr[axis]):
                rpmdri = jnp.tile(r_arr, (2, 1)).T
                # rpmdri[axis, 1] += eps_arr[axis]
                # rpmdri[axis, 0] -= eps_arr[axis]
                rpmdri = rpmdri.at[axis, 1].add(eps_arr[axis])
                rpmdri = rpmdri.at[axis, 0].subtract(eps_arr[axis])

                F = np.zeros((2,))
                for jj in range(2):
                    self.set_initial_position_and_velocity(rpmdri[:, jj], jnp.zeros((3,)))
                    f = self.find_equilibrium_force(**kwargs)

                    assert f is not None
                    F[jj] = f[axis]

                dF = jnp.diff(jnp.array(F))[0]
                if dF < 0:
                    self.omega = self.omega.at[axis].set(jnp.sqrt(-dF / (2 * eps_arr[axis] * mass)))
                else:
                    self.omega = self.omega.at[axis].set(0.0)
            else:
                self.omega = self.omega.at[axis].set(0.0)

        result = self.omega[jnp.asarray(axes)]
        if len(axes) == 1:
            return float(result[0])
        return result

    def damping_coeff(
        self,
        axes: Sequence[int],
        r: npt.ArrayLike | None = None,
        eps: float = 0.01,
        **kwargs: Any,
    ) -> jax.Array | float:
        r"""
        Find the damping coefficent.

        Uses the find_equilibrium force() method to calculate the damping
        coefficient for the particular configuration.

        Parameters
        ----------
        axes : array_like
            A list of axis indices to compute the damping coefficient(s) along.
            Here, :math:`\\hat{x}` is index 0, :math:`\\hat{y}` is index 1, and
            :math:`\\hat{z}` is index 2.  For example, `axes=[2]` calculates
            the damping parameter along :math:`\\hat{z}`.
        r : array_like, optional
            The position at which to calculate the damping coefficient.  By
            default r=None, which defaults to calculating at the equilibrium
            position as found by the find_equilibrium_position() method.  If
            this method has not been run, it defaults to the origin.
        eps : float
            The small numerical :math:`\\epsilon` parameter used for calculating
            the :math:`df/dv` derivative.  Default: 0.01
        kwargs :
            Any additional keyword arguments to pass to find_equilibrium_force()

        Returns
        -------
        beta : list or float
            The damping coefficients along the selected axes.
        """
        self.beta = jnp.zeros(
            3,
        )

        eps_arr: jax.Array = jnp.array([eps] * 3) if isinstance(eps, float) else jnp.asarray(eps)

        if r is None and self.r_eq is None:
            r = jnp.array([0.0, 0.0, 0.0])
        elif r is None:
            r = self.r_eq if self.r_eq is not None else jnp.array([0.0, 0.0, 0.0])

        assert r is not None
        r_arr: jax.Array = jnp.asarray(r)

        for axis in axes:
            if not jnp.isnan(r_arr[axis]):
                vpmdvi = jnp.zeros((3, 2))
                vpmdvi = vpmdvi.at[axis, 1].add(eps_arr[axis])
                vpmdvi = vpmdvi.at[axis, 0].subtract(eps_arr[axis])

                F = np.zeros((2,))
                for jj in range(2):
                    self.set_initial_position_and_velocity(r_arr, vpmdvi[:, jj])
                    f = self.find_equilibrium_force(**kwargs)

                    assert f is not None
                    F[jj] = f[axis]

                dF = jnp.diff(jnp.asarray(F))[0]
                self.beta = self.beta.at[axis].set(-dF / (2 * eps_arr[axis]))
            else:
                self.beta = self.beta.at[axis].set(0)

        result = self.beta[jnp.asarray(axes)]
        if len(axes) == 1:
            return float(result[0])
        return result
