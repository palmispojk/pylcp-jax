"""
Common utilities shared across the pylcp_jax package.

Includes coordinate-basis conversions (Cartesian <-> spherical), a terminal
progress bar, a random unit-vector generator, and the base force-profile
storage class used by all governing-equation solvers.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy.typing as npt


class progressBar(object):
    """
    A terminal progress bar with completion percentage and ETA.

    Displays completion percentage, a visual bar, and estimated time remaining.
    Call ``update(fraction)`` with a value between 0 and 1 to refresh the
    display; it automatically prints a completion message when the fraction
    reaches 1.
    """

    def __init__(
        self,
        decimals: int = 1,
        fill: str = "█",
        prefix: str = "Progress:",
        suffix: str = "",
        time_remaining_prefix: str = " time left",
        length: int = 30,
        update_rate: float = 0.5,
    ) -> None:
        self.tic = time.time()
        self.decimals = decimals
        self.fill = fill
        self.length = length
        self.prefix = prefix
        self.suffix = suffix
        self.time_remaining_prefix = time_remaining_prefix
        self.finished = False
        self.max_written_length = 0
        self.last_update = 0.0
        self.update_rate = update_rate

    def format_time(self, tic_toc: float) -> str:
        """Format a duration in seconds as a human-readable string.

        Parameters
        ----------
        tic_toc : float
            Duration in seconds.

        Returns
        -------
        time_str : str
            ``H:MM:SS`` for durations >= 1 hour, ``M:SS`` for >= 1 minute,
            or ``X.XX s`` for shorter durations.
        """
        if tic_toc > 3600:
            time_str = "%d:%02d:%02d" % (
                (tic_toc) / 3600.0,
                ((tic_toc) / 60.0) % 60.0,
                (tic_toc) % 60.0,
            )
        elif tic_toc > 60:
            time_str = "%d:%02d" % ((tic_toc) / 60.0, (tic_toc) % 60.0)
        else:
            time_str = "%.2f s" % (tic_toc)

        return time_str

    def print_string(self, string1: str) -> None:
        """Print a string in-place, padding to erase any longer previous output.

        Tracks the maximum printed length so that shorter subsequent strings
        do not leave stale characters on the terminal line.

        Parameters
        ----------
        string1 : str
            The string to print.
        """
        # Update the maximum length of string written:
        self.max_written_length = max(self.max_written_length, len(string1))
        pad = " " * (self.max_written_length - len(string1))
        print(string1 + pad, end="\r")

    def update(self, percentage: float) -> None:
        """Refresh the progress bar display.

        Redraws the bar only when enough wall-clock time has elapsed since the
        last update (controlled by ``self.update_rate``).  When ``percentage``
        reaches 1 the bar is replaced by a completion message showing total
        elapsed time.

        Parameters
        ----------
        percentage : float
            Completion fraction in [0, 1].
        """
        toc = time.time()
        if percentage > 0 and percentage < 1 and (toc - self.last_update) > self.update_rate:
            percent = ("{0:." + str(self.decimals) + "f}").format(100 * percentage)
            filledLength = int(self.length * percentage)
            bar = self.fill * filledLength + "-" * (self.length - filledLength)

            remaining_time = (1 - percentage) * ((toc - self.tic) / percentage)
            if remaining_time > 0:
                time_str = self.format_time(remaining_time)
            else:
                time_str = "0.00 s"
            self.print_string(
                "%s |%s| %s%%%s;%s: %s"
                % (
                    self.prefix,
                    bar,
                    percent,
                    self.suffix,
                    self.time_remaining_prefix,
                    time_str,
                )
            )
            self.last_update = toc
        elif percentage >= 1:
            if not self.finished:
                self.finished = True
                time_str = self.format_time(toc - self.tic)
                self.print_string("Completed in %s." % time_str)
                print()


def cart2spherical(A: jax.Array) -> jax.Array:
    """Convert a Cartesian vector to the spherical (circular) basis.

    Maps (x, y, z) to (e_{-1}, e_0, e_{+1}) using the standard atomic-physics
    convention for polarisation components.
    """
    return jnp.array(
        [
            (A[0] - 1j * A[1]) / jnp.sqrt(2),
            A[2],
            -(A[0] + 1j * A[1]) / jnp.sqrt(2),
        ]
    )


def spherical2cart(A: jax.Array) -> jax.Array:
    """Convert a spherical (circular) basis vector to Cartesian.

    Maps (e_{-1}, e_0, e_{+1}) back to (x, y, z).  Inverse of
    :func:`cart2spherical`.
    """
    return jnp.array(
        [
            1 / jnp.sqrt(2) * (-A[2] + A[0]),
            1j / jnp.sqrt(2) * (A[2] + A[0]),
            A[1],
        ]
    )


def spherical_dot(A: jax.Array, B: jax.Array) -> jax.Array:
    """Compute the dot product of two spherical-basis vectors.

    The metric tensor in this basis introduces alternating signs:
    A·B = -A_{-1}B_{+1} + A_0 B_0 - A_{+1}B_{-1}.
    """
    return jnp.tensordot(A, jnp.array([-1.0, 1.0, -1.0]) * B[::-1], axes=(0, 0))


class base_force_profile:
    """
    Base force profile.

    The force profile object stores all of the calculated quantities created by
    the governingeq.generate_force_profile() method.  It has the following
    attributes:

    Attributes
    ----------
    R : jnp.Array, shape (3, ...)
        Positions at which the force profile was calculated.
    V : jnp.Array, shape (3, ...)
        Velocities at which the force profile was calculated.
    F : jnp.Array, shape (3, ...)
        Total equilibrium force at position R and velocity V.
    f_mag : jnp.Array, shape (3, ...)
        Magnetic force at position R and velocity V.
    f : dictionary of jnp.Array
        The forces due to each laser, indexed by the
        manifold the laser addresses.  The dictionary is keyed by the transition
        driven, and individual lasers are in the same order as in the
        pylcp_jax.laserBeams object used to create the governing equation.
    Neq : jnp.Array
        Equilibrium population found.
    """

    def __init__(
        self,
        R: npt.ArrayLike,
        V: npt.ArrayLike,
        laserBeams: dict[str, Any],
        hamiltonian: Any,
    ) -> None:
        if not isinstance(R, jnp.ndarray):
            R = jnp.array(R)
        if not isinstance(V, jnp.ndarray):
            V = jnp.array(V)

        if R.shape[0] != 3 or V.shape[0] != 3:
            raise TypeError("R and V must have first dimension of 3.")

        self.R = R
        self.V = V

        if hamiltonian is None:
            self.Neq = None
        else:
            self.Neq = jnp.zeros(R[0].shape + (hamiltonian.n,))

        self.f = {}
        for key in laserBeams:
            self.f[key] = jnp.zeros(R.shape + (len(laserBeams[key].beam_vector),))

        self.f_mag = jnp.zeros(R.shape)

        self.F = jnp.zeros(R.shape)

    def store_data(
        self,
        ind: tuple[int, ...],
        Neq: jax.Array | None,
        F: jax.Array,
        F_laser: dict[str, jax.Array],
        F_mag: jax.Array,
    ) -> None:
        """Store computed force results at a single grid index.

        Parameters
        ----------
        ind : tuple of int
            Multi-dimensional index into the profile arrays.
        Neq : jax.Array or None
            Equilibrium population vector; skipped when None.
        F : jax.Array, shape (3,)
            Total force vector.
        F_laser : dict of jax.Array
            Per-laser force contributions, keyed by transition label.
        F_mag : jax.Array, shape (3,)
            Magnetic force contribution.
        """
        if Neq is not None and self.Neq is not None:
            self.Neq = self.Neq.at[ind].set(Neq)

        for jj in range(3):
            self.F = self.F.at[(jj,) + ind].set(F[jj])
            for key in F_laser:
                self.f[key] = self.f[key].at[(jj,) + ind].set(F_laser[key][jj])

            self.f_mag = self.f_mag.at[(jj,) + ind].set(F_mag[jj])


def random_vector(key: jax.Array, free_axes: Sequence[bool] = [True, True, True]) -> jax.Array:
    """
    Return a random vector in either 1D, 2D, or 3D.

    Parameters
    ----------
        key: jax.random.PRNGKeyArray
            A JAX pseudo-random number generator key. Because JAX functions
            must be pure and stateless for GPU acceleration, randomness requires
        passing an explicit PRNG state.
        free_axes: list of 3 booleans (optional)
            Which axes (x, y, z) are considered free for the random
            direction.  Default: [True, True, True], i.e., a full 3D
            unit vector.

    Returns
    -------
        vector: jax.Array of shape (3,)
            Random vector with unit length.
    """
    free_axes_arr = jnp.array(free_axes)
    axes_count = sum(free_axes)

    key1, key2 = jax.random.split(key)
    a = jax.random.uniform(key1)
    b = jax.random.uniform(key2)

    if axes_count == 1:
        val = jnp.sign(a - 0.5)
        return val * free_axes_arr

    elif axes_count == 2:
        phi = 2 * jnp.pi * a
        x = jnp.array([jnp.cos(phi), jnp.sin(phi)])
        y = jnp.zeros(3)
        y = y.at[free_axes_arr].set(x)
        return y

    elif axes_count == 3:
        th = jnp.arccos(2 * b - 1)
        phi = 2 * jnp.pi * a
        return jnp.array(
            [
                jnp.sin(th) * jnp.cos(phi),
                jnp.sin(th) * jnp.sin(phi),
                jnp.cos(th),
            ]
        )

    else:
        raise ValueError(
            f"free_axes must be a boolean list of length 1 to 3. Got {axes_count} True entries."
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    master_key = jax.random.PRNGKey(
        42
    )  # can be a random seed as well, is kept as a chosen number here

    keys = jax.random.split(master_key, 500)  # splits the random key 500 times

    batched_random_vector = jax.vmap(random_vector, in_axes=(0, None))

    vectors = batched_random_vector(
        keys, (True, True, True)
    )  # generates all random vectors at once

    vectors = np.array(vectors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2])  # type: ignore[arg-type]
    ax.view_init(elev=-90.0, azim=0.0)
    plt.show()
