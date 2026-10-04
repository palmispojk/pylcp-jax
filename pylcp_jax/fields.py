"""
Magnetic field and laser beam classes for pylcp_jax.

Provides classes for defining magnetic fields (constant, quadrupole,
Ioffe-Pritchard) and laser beams (plane wave, Gaussian, clipped Gaussian)
used in laser cooling and trapping simulations.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from pylcp_jax.common import cart2spherical, spherical2cart


def return_constant_val(R: npt.ArrayLike, t: float, val: Any) -> Any:
    """Return a constant scalar value, ignoring the position and time arguments.

    Used internally to promote constant parameters to callable ``(R, t)`` form.

    Parameters
    ----------
    R : array_like
        Not used; position argument kept for API compatibility.
    t : float
        Not used; time argument kept for API compatibility.
    val : array_like or scalar
        The constant value to return.

    Returns
    -------
    val : array_like or scalar
        The value passed in, unchanged.
    """
    return val


def return_constant_vector(R: npt.ArrayLike, t: float, vector: npt.ArrayLike) -> jax.Array:
    """Return a constant vector as a JAX array, ignoring position and time.

    Used internally to promote constant vector parameters to callable ``(R, t)``
    form, analogous to :func:`return_constant_val` for scalars.

    Parameters
    ----------
    R : array_like
        Not used; position argument kept for API compatibility.
    t : float
        Not used; time argument kept for API compatibility.
    vector : array_like, shape (3,)
        The constant vector to return.

    Returns
    -------
    vector : jax.Array, shape (3,)
        The input vector converted to a JAX array.
    """
    return jnp.array(vector)


def return_constant_val_t(t: float | jax.Array, val: float) -> float | jax.Array:
    """Return a constant value, broadcasting over a time array when needed.

    Parameters
    ----------
    t : float or jax.Array
        Time argument.  If a JAX array, the constant ``val`` is broadcast to
        match its shape.
    val : float
        The constant value to return.

    Returns
    -------
    val : float or jax.Array
        ``val`` unchanged when ``t`` is a scalar; an array of shape ``t.shape``
        filled with ``val`` when ``t`` is an array.
    """
    if isinstance(t, jnp.ndarray):
        return val * jnp.ones(t.shape)
    else:
        return val


def promote_to_lambda(
    val: Any,
    var_name: str = "",
    kind: str = "Rt",
) -> tuple[Callable[..., Any], str]:
    """
    Promotes a constant or callable to a lambda function with proper arguments.

    Parameters
    ----------
        val : array_like or callable
            The value to promote.  Can either be a function, array_like
            (vector), or a scalar (constant).
        var_name : str, optional
            Name of the variable attempting to be promoted.  Useful for error
            messages. Default: empty string.
        kind : str, optional
            The arguments of the lambda function we are creating.  If `Rt`,
            the lambda function returned has ``(R, t)`` as its arguments.  If
            `t`, it has only `t` as its arguments.  Default: `Rt`.


    Returns
    -------
        func : callable
            lambda function created by this function
        sig : string
            Either `(R,t)` or `(t)`
    """
    if kind == "Rt":
        if not callable(val):
            if isinstance(val, list) or isinstance(val, jnp.ndarray):
                func = lambda R=jnp.array([0.0, 0.0, 0.0]), t=0.0: return_constant_vector(R, t, val)
            else:
                func = lambda R=jnp.array([0.0, 0.0, 0.0]), t=0.0: return_constant_val(R, t, val)
            sig = "()"
        else:
            sig = str(signature(val))
            if "(R)" in sig or "(r)" in sig or "(x)" in sig:
                func = lambda R=jnp.array([0.0, 0.0, 0.0]), t=0.0: val(R)
                sig = "(R)"
            elif "(R, t)" in sig or "(r, t)" in sig or "(x, t)" in sig:
                func = lambda R=jnp.array([0.0, 0.0, 0.0]), t=0.0: val(R, t)
                sig = "(R, t)"
            elif "(t)" in sig:
                func = lambda R=jnp.array([0.0, 0.0, 0.0]), t=0.0: val(t)
                sig = "(R, t)"
            else:
                raise TypeError(f"Signature [{sig}] of function {var_name} not understood.")

        return func, sig
    elif kind == "t":
        if not callable(val):
            func = lambda t=0.0: return_constant_val_t(t, val)
            sig = "()"
        else:
            sig = str(signature(val))
            if "(t)" in sig:
                func = lambda t=0.0: val(t)
            else:
                raise TypeError(f"Signature [{sig}] of function {var_name} not understood.")

        return func, sig
    else:
        raise TypeError("kind must be 'Rt' or 't', got %r" % kind)


class magField(object):
    """
    Base magnetic field class.

    Stores a magnetic field and calculates useful derivatives
    for `pylcp_jax`.

    Parameters
    ----------
    field : array_like with shape (3,) or callable
        If constant, the magnetic field vector, specified as either as an array_like
        with shape (3,).  If a callable, it must have a signature like (R, t), (R),
        or (t) where R is an array_like with shape (3,) and t is a float and it
        must return an array_like with three elements.
    eps : float, optional
        Small distance to use in calculation of numerical derivatives.  By default
        `eps=1e-5`.

    Attributes
    ----------
    eps : float
        small epsilon used for computing derivatives
    """

    def __init__(self, field: npt.ArrayLike | Callable[..., Any], eps: float = 1e-5) -> None:
        self.eps = eps

        # Promote it to a lambda func:
        self.Field, self.FieldSig = promote_to_lambda(field, var_name="for field")

        self.gradField = jax.jacfwd(self.Field, argnums=0)

    def FieldMag(self, R=jnp.array([0.0, 0.0, 0.0]), t=0):
        """
        Magnetic field magnitude at R and t.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        B : float
            the magnetic field magnitude at position R and time t.
        """
        return jnp.linalg.norm(self.Field(R, t))

    def gradFieldMag(self, R=jnp.array([0.0, 0.0, 0.0]), t=0):
        r"""
        Gradient of the magnetic field magnitude at R and t.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dB : array_like, shape (3,)
            :math:`\\nabla|B|`, the gradient of the magnetic field magnitude
            at position :math:`R` and time :math:`t`.
        """
        return jax.grad(self.FieldMag, argnums=0)(R, t)


class iPMagneticField(magField):
    r"""
    Ioffe-Pritchard trap magnetic field.

    Generates a magnetic field of the form

    .. math::
      \\mathbf{B} = B_1 x \\hat{x} - B_1 y \\hat{y} + \\left(B_0 + \\frac{B_2}{2}z^2\\right)\\hat{z}

    Parameters
    ----------
    B0 : float
        Constant offset field
    B1 : float
        Magnetic field gradient in x-y plane
    B2 : float
        Magnetic quadratic component along z direction.

    Notes
    -----
    It is currently missing extra terms that are required for it to fulfill
    Maxwell's equations at second order.
    """

    def __init__(self, B0: float, B1: float, B2: float, eps: float = 1e-5) -> None:
        self.B0 = B0
        self.B1 = B1
        self.B2 = B2

        super().__init__(
            lambda R, t: jnp.array(
                [
                    B1 * R[0] - B2 * R[0] * R[2] / 2,
                    -R[1] * B1 - B2 * R[1] * R[2] / 2,
                    B0 + B2 / 2 * (R[2] ** 2 - (R[0] ** 2 + R[1] ** 2) / 2),
                ]
            )
        )


class constantMagneticField(magField):
    r"""
    Spatially constant magnetic field.

    Represents a magnetic field of the form

    .. math::
      \\mathbf{B} = \\mathbf{B}_0

    Parameters
    ----------
    val : array_like with shape (3,)
        The three-vector defintion of the constant magnetic field.
    """

    def __init__(self, B0: npt.ArrayLike) -> None:
        self.B0 = B0
        super().__init__(lambda R, t: B0)


class quadrupoleMagneticField(magField):
    r"""
    Spherical quadrupole magnetic field.

    Represents a magnetic field of the form

    .. math::
      \\mathbf{B} = \\alpha\\left(- \\frac{x\\hat{x}}{2} - \\frac{y\\hat{y}}{2} + z\\hat{z}\\right)

    Parameters
    ----------
    alpha : float
        strength of the magnetic field gradient.
    """

    def __init__(self, alpha: float, eps: float = 1e-5) -> None:
        self.alpha = alpha

        super().__init__(lambda R, t: alpha * jnp.array([-0.5 * R[0], -0.5 * R[1], R[2]]))


class laserBeam(object):
    r"""
    The base class for a single laser beam.

    Attempts to represent a laser beam as

    .. math::
        \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}}(r, t) E_0(r, t)
        e^{i\\mathbf{k}(r,t)\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}

    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the detuning,
    :math:`t` is the time, and :math:`\\phi` is the phase.


    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array or as callable function.  If a callable, it
        must have a signature like (R, t), (R), or (t) where R is an array_like with
        shape (3,) and t is a float and it must return an array_like with three
        elements.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,), or as callable function.  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`. If a callable, it must
        have a signature like (R, t), (R), or (t) where R is an array_like with
        shape (3,) and t is a float and it must return an array_like with three
        elements.
    s : float or callable
        The intensity of the laser beam, normalized to the saturation intensity,
        specified as either a float or as callable function.  If a callable,
        it must have a signature like (R, t), (R), or (t) where R is an
        array_like with shape (3,) and t is a float and it must return a float.
    delta: float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    phase : float, optional
        Phase of laser beam.  By default, zero.
    pol_coord : string, optional
        Polarization basis of the input polarization vector: 'cartesian'
        or 'spherical' (default).
    eps : float, optional
        Small distance to use in calculation of numerical derivatives.  By default
        `eps=1e-5`.

    Attributes
    ----------
    eps : float
        Small epsilon used for computing derivatives
    phase : float
        Overall phase of the laser beam.
    """

    def __init__(
        self,
        kvec: npt.ArrayLike | None = None,
        s: float | Callable[..., Any] | None = None,
        pol: int | float | npt.ArrayLike | Callable[..., jax.Array] | None = None,
        delta: float | Callable[[float], float] | None = None,
        phase: float | Callable[[float], float] = 0.0,
        pol_coord: str = "spherical",
        eps: float = 1e-5,
    ) -> None:
        self._kvec = jnp.array(kvec) if kvec is not None else jnp.array([0.0, 0.0, 1.0])
        self._s = s if callable(s) else (float(jnp.real(s)) if s is not None else 1.0)
        self._delta = (
            delta if callable(delta) else (float(jnp.real(delta)) if delta is not None else 0.0)
        )
        self._phase = (
            phase if callable(phase) else (float(jnp.real(phase)) if phase is not None else 0.0)
        )
        self.eps = eps

        if pol is not None:
            parsed_pol = self.__parse_constant_polarization(pol, pol_coord)
            self._pol = jnp.array(parsed_pol, dtype=jnp.complex64)
        else:
            self._pol = jnp.array([0.0, 0.0, 1.0], dtype=jnp.complex64)

    def __parse_constant_polarization(self, pol, pol_coord):
        if isinstance(pol, (float, int)):
            # If the polarization is defined by just a single number (+/-1),
            # we assume that the polarization is defined as sigma^+ or sigma^-
            # using the k-vector of the light as the axis defining z.  In this
            # case, we want to project onto the actual z axis, which is
            # relatively simple as there is only one angle.

            # Set the polarization in this direction:
            if jnp.sign(pol) < 0:
                base_pol = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex64)
            else:
                base_pol = jnp.array([0.0, 0.0, 1.0], dtype=jnp.complex64)

            k_norm = jnp.linalg.norm(self._kvec)
            # jnp.where with 3 args always returns Array; cast avoids stub ambiguity
            k_dir = self._kvec / cast(jax.Array, jnp.where(k_norm == 0, 1.0, k_norm))

            parsed_pol = self.project_pol(k_dir, pol=base_pol, invert=True)

        elif isinstance(pol, (jnp.ndarray, np.ndarray, list)):
            pol_arr = jnp.array(pol)
            if pol_arr.shape != (3,):
                raise ValueError("pol, when a vector, must be a (3,) array")

            # The user has specified a single polarization vector in
            # cartesian coordinates:
            if pol_coord == "cartesian":
                # Check for transverseness:
                if jnp.abs(jnp.dot(self.kvec(), pol_arr)) > 1e-9:
                    raise ValueError("I'm sorry; light is a transverse wave")

                parsed_pol = cart2spherical(pol_arr)

            # The user has specified a single polarization vector in
            # spherical basis:
            elif pol_coord == "spherical":
                pol_cart = spherical2cart(pol_arr)

                # Check for transverseness:
                if jnp.abs(jnp.dot(self.kvec(), pol_cart)) > 1e-9:
                    raise ValueError("I'm sorry; light is a transverse wave")
                parsed_pol = pol_arr
            else:
                raise ValueError(f"Unknown pol_coord: {pol_coord}")

            parsed_pol = parsed_pol / jnp.linalg.norm(parsed_pol)
        else:
            raise ValueError("pol must be +1, -1, or a numpy array")

        return jnp.array(parsed_pol, dtype=jnp.complex64)

    def kvec(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """
        Return the k-vector of the laser beam.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        kvec : array_like, size(3,)
            the k vector at position R and time t.
        """
        return self._kvec

    def intensity(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """
        Return the intensity of the laser beam at position R and t.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        s : float or array_like
            Saturation parameter of the laser beam at R and t.
        """
        if callable(self._s):
            return self._s(R, t)
        return self._s

    def pol(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """
        Return the polarization of the laser beam at position R and t.

        The polarization is returned in the spherical basis.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : array_like, size (3,)
            polarization of the laser beam at R and t in spherical basis.
        """
        return self._pol

    def delta(self, t=0.0):
        """
        Return the detuning of the laser beam at time t.

        Parameters
        ----------
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        delta : float or array like
            detuning of the laser beam at time t
        """
        if callable(self._delta):
            return self._delta(t)
        return self._delta

    def phase(self, t=0.0):
        """Return the static phase offset of the laser beam at time t.

        Parameters
        ----------
        t : float, optional
            Time at which to evaluate the phase.  Default: 0.

        Returns
        -------
        phase : float
            Phase of the laser beam at time t.
        """
        if callable(self._phase):
            return self._phase(t)
        return self._phase

    def delta_phase(self, t=0.0):
        """Return the cumulative phase accumulated by detuning up to time t.

        Computes ``delta * t``.  Raises :exc:`NotImplementedError` if ``delta``
        is callable, because numerical integration of a time-dependent detuning
        is too slow inside a JIT-compiled kernel.  In that case, supply the
        analytic phase integral as a callable ``phase`` argument instead.

        Parameters
        ----------
        t : float, optional
            Time at which to evaluate the cumulative phase.  Default: 0.

        Returns
        -------
        delta_phase : float
            ``delta * t``.

        Raises
        ------
        NotImplementedError
            If ``delta`` was supplied as a callable.
        """
        if callable(self._delta):
            raise NotImplementedError(
                "Callable delta cannot be used to compute delta_phase inside the OBE "
                "E-field because numerical integration is too slow inside a JIT kernel. "
                "Instead, set delta=0 and supply the analytic phase integral as a "
                "callable phase, e.g.:\n"
                "  'delta': 0,\n"
                "  'phase': lambda t: Delta0/omegam * jnp.sin(omegam * t)"
            )
        return self._delta * t

    # TODO: add testing of kvec/pol orthogonality.
    def project_pol(
        self,
        quant_axis,
        R=jnp.array([0.0, 0.0, 0.0]),
        t=0.0,
        treat_nans=False,
        calculate_norm=False,
        invert=False,
        pol=None,
        **kwargs,
    ):
        """
        Project the polarization onto a quantization axis.

        Parameters
        ----------
        quant_axis : array_like, shape (3,)
            A normalized 3-vector of the quantization axis direction.
        R : array_like, shape (3,), optional
            If polarization is a function of R is the
            3-vectors at which the polarization shall be calculated.
        calculate_norm : bool, optional
            If true, renormalizes the quant_axis.  By default, False.
        treat_nans : bool, optional
            If true, every place that nan is encountered, replace with the
            $hat{z}$ axis as the quantization axis.  By default, False.
        invert : bool, optional
            If true, invert the process to project the quantization axis
            onto the specified polarization.

        Returns
        -------
        projected_pol : array_like, shape (3,)
            The polarization projected onto the quantization axis.
        """
        p_vec = jnp.array(pol) if pol is not None else self.pol(R, t)

        if calculate_norm:
            quant_axis_norm = jnp.linalg.norm(quant_axis, axis=0)

            safe_norm = jnp.where(quant_axis_norm == 0, 1.0, quant_axis_norm)
            quant_axis = quant_axis / safe_norm

            fallback = jnp.zeros_like(quant_axis).at[-1].set(1.0)
            quant_axis = jnp.where(quant_axis_norm == 0, fallback, quant_axis)

        elif treat_nans:
            nan_mask = jnp.isnan(quant_axis[-1])

            fallback = jnp.zeros_like(quant_axis).at[-1].set(1.0)
            quant_axis = jnp.where(nan_mask, fallback, quant_axis)

        cosbeta = quant_axis[2]
        sinbeta = jnp.sqrt(jnp.clip(1.0 - cosbeta**2, 0.0, 1.0))
        gamma = jnp.arctan2(quant_axis[1], quant_axis[0])
        alpha = 0.0

        D = jnp.array(
            [
                [
                    (1 + cosbeta) / 2 * jnp.exp(-1j * alpha + 1j * gamma),
                    -sinbeta / jnp.sqrt(2) * jnp.exp(-1j * alpha),
                    (1 - cosbeta) / 2 * jnp.exp(-1j * alpha - 1j * gamma),
                ],
                [
                    sinbeta / jnp.sqrt(2) * jnp.exp(1j * gamma),
                    cosbeta,
                    -sinbeta / jnp.sqrt(2) * jnp.exp(-1j * gamma),
                ],
                [
                    (1 - cosbeta) / 2 * jnp.exp(1j * alpha + 1j * gamma),
                    sinbeta / jnp.sqrt(2),
                    (1 + cosbeta) / 2 * jnp.exp(1j * alpha - 1j * gamma),
                ],
            ],
            dtype=jnp.complex64,
        )

        if invert:
            D = jnp.conjugate(jnp.swapaxes(D, 0, 1))

        return jnp.einsum("ij...,j...->i...", D, p_vec)

    def cartesian_pol(self, R=jnp.array([0.0, 0.0, 0.0]), t=0):
        """
        Return the polarization in Cartesian coordinates.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : array_like, size (3,)
            polarization of the laser beam at R and t in Cartesian basis.
        """
        pol = self.pol(R, t)
        return spherical2cart(pol)

    def jones_vector(self, xp, yp, R=jnp.array([0.0, 0.0, 0.0]), t=0):
        """
        Return the Jones vector at position.

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the Jones vector.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the Jones vector.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : array_like, size (2,)
            Jones vector of the laser beam at R and t in Cartesian basis.
        """
        # First, run some basic checks.
        if jnp.abs(jnp.dot(xp, yp)) > 1e-10:
            raise ValueError("xp and yp must be orthogonal.")
        if jnp.abs(jnp.dot(xp, self.kvec(R, t))) > 1e-10:
            raise ValueError("xp and k must be orthogonal.")
        if jnp.abs(jnp.dot(yp, self.kvec(R, t))) > 1e-10:
            raise ValueError("yp and k must be orthogonal.")
        if jnp.sum(jnp.abs(jnp.cross(xp, yp) - self.kvec(R, t))) > 1e-10:
            raise ValueError("xp, yp, and k must form a right-handed" + "coordinate system.")

        pol_cart = self.cartesian_pol(R, t)

        if jnp.abs(jnp.dot(pol_cart, self.kvec(R, t))) > 1e-9:
            raise ValueError("Something is terribly, terribly wrong.")

        return jnp.array([jnp.dot(pol_cart, xp), jnp.dot(pol_cart, yp)])

    def stokes_parameters(self, xp, yp, R=jnp.array([0.0, 0.0, 0.0]), t=0):
        """
        Return the Stokes Parameters of the laser beam at R and t.

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the Stokes parameters.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the Stokes parameters.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : array_like, shape (3,)
            Stokes parameters for the laser beam, [Q, U, V]
        """
        jones_vector = self.jones_vector(xp, yp, R, t)

        Q = jnp.abs(jones_vector[0]) ** 2 - jnp.abs(jones_vector[1]) ** 2
        U = 2 * jnp.real(jones_vector[0] * jnp.conj(jones_vector[1]))
        V = -2 * jnp.imag(jones_vector[0] * jnp.conj(jones_vector[1]))

        return jnp.array([Q, U, V])

    def polarization_ellipse(self, xp, yp, R=jnp.array([0.0, 0.0, 0.0]), t=0):
        r"""
        Return the polarization ellipse parameters of the laser beam at R and t.

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the polarization ellipse.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the polarization ellipse.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        psi : float
            :math:`\\psi` parameter of the polarization ellipse
        chi : float
            :math:`\\chi` parameter of the polarization ellipse
        """
        Q, U, V = self.stokes_parameters(xp, yp, R, t)

        psi = (jnp.arctan2(U, Q) % (2 * jnp.pi)) / 2
        if jnp.sqrt(Q**2 + U**2) > 1e-10:
            chi = 0.5 * jnp.arctan(V / jnp.sqrt(Q**2 + U**2))
        else:
            chi = jnp.pi / 4 * jnp.sign(V)

        return (psi, chi)

    def electric_field(
        self,
        R=jnp.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        t=0.0,
    ):
        """
        Return the electric field at position R and t.

        Parameters
        ----------
        R : jnp.Array, size (3,)
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        Eq : jnp.Array, shape (3,)
            electric field in the spherical basis.
        """
        kvec = self.kvec(R, t)
        s = self.intensity(R, t)
        pol = self.pol(R, t)
        delta_phase = self.delta_phase(t)
        phase = self.phase(t)

        amp = jnp.sqrt(2 * s)

        phi = -1j * (jnp.dot(kvec, R) - delta_phase + phase)  # phase term

        return pol * amp * jnp.exp(phi)

    def electric_field_gradient(
        self,
        R=jnp.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        t=0.0,
    ):
        """Return the Jacobian of the electric field with respect to position.

        Computes the gradient numerically via forward-mode automatic
        differentiation.  The returned matrix follows the convention
        ``M[i, j] = dE_j / dR_i``, so that each row ``i`` gives the
        spatial derivative along direction ``i``.

        Parameters
        ----------
        R : jnp.Array, shape (3,), optional
            Position at which to evaluate the gradient.  Default: origin.
        t : float, optional
            Time at which to evaluate the gradient.  Default: 0.

        Returns
        -------
        grad_E : jnp.Array, shape (3, 3)
            Jacobian of the electric field in the spherical basis.
        """

        def e_field_R(R_val):
            return self.electric_field(R_val, t)

        # Transpose so convention is [grad_dir, field_comp], i.e.
        # M[i, j] = dE_j / dR_i, matching infinitePlaneWaveBeam.
        return jax.jacfwd(e_field_R)(R).T


class infinitePlaneWaveBeam(laserBeam):
    r"""
    Infinte plane wave beam.

    A beam which has spatially constant intensity, k-vector, and polarization.

    .. math::
        \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}} E_0e^{i\\mathbf{k}\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}
    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the detuning,
    :math:`t` is the time, and :math:`\\phi` is the phase.

    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,).  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`.
    s : float or callable
        The intensity of the laser beam, specified as either a float or as
        callable function.
    delta: float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    **kwargs :
        Additional keyword arguments to pass to laserBeam superclass.

    Notes
    -----
    This implementation is much faster, when it can be used, compared to the
    base laserBeam class.
    """

    def __init__(
        self,
        kvec: npt.ArrayLike,
        pol: int | float | npt.ArrayLike,
        s: float,
        delta: float | Callable[[float], float],
        **kwargs: Any,
    ) -> None:
        if callable(kvec):
            raise TypeError("kvec cannot be a function for an infinite plane wave.")

        if callable(pol):
            raise TypeError("Polarization cannot be a function for an infinite plane wave.")

        self.con_kvec = jnp.array(kvec)

        super().__init__(kvec=self.con_kvec, pol=pol, s=s, delta=delta, **kwargs)

    def electric_field_gradient(
        self,
        R=jnp.array(
            [
                0.0,
                0.0,
                0.0,
            ]
        ),
        t=0.0,
    ):
        """Return the analytic Jacobian of the electric field for a plane wave.

        For an infinite plane wave the gradient is exact:
        ``dE_j / dR_i = -i k_i E_j``, i.e. ``-1j * outer(k, E)``.
        The convention matches the base class: ``M[i, j] = dE_j / dR_i``.

        Parameters
        ----------
        R : jnp.Array, shape (3,), optional
            Position at which to evaluate the gradient.  Default: origin.
        t : float, optional
            Time at which to evaluate the gradient.  Default: 0.

        Returns
        -------
        grad_E : jnp.Array, shape (3, 3)
            Jacobian of the electric field in the spherical basis.
        """
        E = self.electric_field(R, t)
        k = self.kvec(R, t)

        return -1j * jnp.outer(k, E)


class gaussianBeam(laserBeam):
    r"""
    Collimated Gaussian beam.

    A beam which has spatially constant k-vector and polarization, with a
    Gaussian intensity modulation.  Specifically,

    .. math::
      \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}} E_0 e^{-\\mathbf{r}^2/w_b^2} e^{i\\mathbf{k}\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}
    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the detuning,
    :math:`t` is the time, and :math:`\\phi` is the phase.  Note that because
    :math:`I\\propto E^2`, :math:`w_b` is the :math:`1/e^2` radius.

    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,).  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`.
    s : float or callable
        The maximum intensity of the laser beam at the center, specified as
        either a float or as callable function.
    delta : float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    wb : float
        The :math:`1/e^2` radius of the beam.
    **kwargs:
        Additional keyword arguments to pass to the laserBeam superclass.
    """

    def __init__(
        self,
        kvec: npt.ArrayLike,
        pol: int | float | npt.ArrayLike,
        s: float,
        delta: float | Callable[[float], float],
        wb: float,
        **kwargs: Any,
    ) -> None:
        if callable(kvec):
            raise TypeError("kvec cannot be a function for a Gaussian beam.")

        if callable(pol):
            raise TypeError("Polarization cannot be a function for a Gaussian beam.")

        # # Use super class to define kvec(R, t), pol(R, t), and delta(t)
        # super().__init__(kvec=kvec, pol=pol, delta=delta, **kwargs)

        # # Save the constant values (might be useful):
        # self.con_kvec = kvec
        # self.con_khat = kvec/np.linalg.norm(kvec)
        # self.con_pol = self.pol(np.array([0., 0., 0.]), 0.)

        # # Save the parameters specific to the Gaussian beam:
        # self.s_max = s # central saturation parameter
        # self.wb = wb # 1/e^2 radius
        # self.define_rotation_matrix()

        self.con_kvec = jnp.array(kvec)
        self.con_khat = self.con_kvec / jnp.linalg.norm(self.con_kvec)
        th = jnp.arccos(self.con_khat[2])
        phi = jnp.arctan2(self.con_khat[1], self.con_khat[0])

        self.rmat = jnp.array(Rotation.from_euler("ZY", [phi, th]).inv().as_matrix())

        self.s_max = s
        self.wb = wb
        super().__init__(kvec=self.con_kvec, pol=pol, delta=delta, s=s, **kwargs)

    def intensity(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """Return the Gaussian intensity profile at position R and time t.

        Evaluates ``s_max * exp(-2 * rho^2 / wb^2)`` where ``rho`` is the
        transverse distance from the beam axis.

        Parameters
        ----------
        R : jnp.Array, shape (3,), optional
            Position at which to evaluate the intensity.  Default: origin.
        t : float, optional
            Time (unused for a static Gaussian beam).  Default: 0.

        Returns
        -------
        s : float
            Saturation parameter (intensity) at position R.
        """
        # Rotate up to the z-axis where we can apply formulas:
        Rp = jnp.matmul(self.rmat, R)

        rho_sq = Rp[0] ** 2 + Rp[1] ** 2

        return self.s_max * jnp.exp(-2 * rho_sq / self.wb**2)


class clippedGaussianBeam(gaussianBeam):
    r"""
    Clipped, collimated Gaussian beam.

    A beam which has spatially constant k-vector and polarization, with a
    Gaussian intensity modulation.  Specifically,

    .. math::
      \\frac{1}{2}\\hat{\\boldsymbol{\\epsilon}} E_0 e^{-\\mathbf{r}^2/w_b^2} (|\\mathbf{r}|<r_s) e^{i\\mathbf{k}\\cdot\\mathbf{r}-i \\int dt\\Delta(t) + i\\phi(r, t)}
    where :math:`\\hat{\\boldsymbol{\\epsilon}}` is the polarization, :math:`E_0`
    is the electric field magnitude, :math:`r_s` is the radius of the stop,
    :math:`\\mathbf{k}(r,t)` is the k-vector,
    :math:`\\mathbf{r}` is the position, :math:`\\Delta(t)` is the detuning,
    :math:`t` is the time, and :math:`\\phi` is the phase. Note that because
    :math:`I\\propto E^2`, :math:`w_b` is the :math:`1/e^2` radius.

    Parameters
    ----------
    kvec : array_like with shape (3,) or callable
        The k-vector of the laser beam, specified as either a three-element
        list or numpy array.
    pol : int, float, array_like with shape (3,), or callable
        The polarization of the laser beam, specified as either an integer, float
        array_like with shape(3,).  If an integer or float,
        if `pol<0` the polarization will be left circular polarized relative to
        the k-vector of the light.  If `pol>0`, the polarization will be right
        circular polarized.  If array_like, polarization will be specified by the
        vector, whose basis is specified by `pol_coord`.
    s : float or callable
        The maximum intensity of the laser beam at the center, specified as
        either a float or as callable function.
    delta : float or callable
        Detuning of the laser beam.  If a callable, it must have a
        signature like (t) where t is a float and it must return a float.
    wb : float
        The :math:`1/e^2` radius of the beam.
    rs : float
        The radius of the stop.
    **kwargs:
        Additional keyword arguments to pass to the laserBeam superclass.
    """

    def __init__(
        self,
        kvec: npt.ArrayLike,
        pol: int | float | npt.ArrayLike,
        s: float,
        delta: float | Callable[[float], float],
        wb: float,
        rs: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(kvec=kvec, pol=pol, s=s, delta=delta, wb=wb, **kwargs)

        self.rs = rs  # Save the radius of the stop.

    def intensity(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """Return the clipped Gaussian intensity profile at position R and time t.

        Evaluates the Gaussian profile ``s_max * exp(-2 * rho^2 / wb^2)`` and
        hard-clips it to zero outside the aperture radius ``rs``.

        Parameters
        ----------
        R : jnp.Array, shape (3,), optional
            Position at which to evaluate the intensity.  Default: origin.
        t : float, optional
            Time (unused for a static beam).  Default: 0.

        Returns
        -------
        s : float
            Saturation parameter (intensity) at position R; zero when the
            transverse distance from the beam axis exceeds ``rs``.
        """
        Rp = jnp.matmul(self.rmat, R)
        rho_sq = Rp[0] ** 2 + Rp[1] ** 2

        # standard gaussian
        s_gaussian = self.s_max * jnp.exp(-2 * rho_sq / self.wb**2)
        # hard clipping vectorized
        return jnp.where(rho_sq <= self.rs**2, s_gaussian, 0.0)


class laserBeams(object):
    """
    The base class for a collection of laser beams.

    Parameters
    ----------
    laserbeamparams : array_like of laserBeam or array_like of dictionaries
        If array_like contains laserBeams, the laserBeams in the array will be joined
        together to form a collection.  If array_like is a list of dictionaries, the
        dictionaries will be passed as keyword arguments to beam_type
    beam_type : laserBeam or laserBeam subclass, optional
        Type of beam to use in the collection of laserBeams.  By default
        `beam_type=laserBeam`.
    """

    def __init__(
        self,
        laserbeamparams: list[laserBeam | dict[str, Any]] | None = None,
        beam_type: type[laserBeam] = laserBeam,
    ) -> None:
        self.beam_vector = []

        if laserbeamparams is not None:
            if not isinstance(laserbeamparams, list):
                raise ValueError("laserbeamparams must be a list.")

            for laserbeamparam in laserbeamparams:
                if isinstance(laserbeamparam, dict):
                    self.beam_vector.append(beam_type(**laserbeamparam))

                elif isinstance(laserbeamparam, laserBeam):
                    self.beam_vector.append(laserbeamparam)
                else:
                    raise TypeError(
                        "Each element of laserbeamparams must either "
                        + "be a list of dictionaries or list of "
                        + "laserBeams"
                    )

        self.num_of_beams = len(self.beam_vector)

    def __iadd__(self, other):
        self.beam_vector += other.beam_vector
        self.num_of_beams = len(self.beam_vector)

        return self

    def __add__(self, other):
        return laserBeams(self.beam_vector + other.beam_vector)

    def add_laser(self, new_laser: laserBeam | dict[str, Any]) -> None:
        """
        Add a laser to the collection.

        Parameters
        ----------
        new_laser : laserBeam or laserBeam subclass
        """
        if isinstance(new_laser, laserBeam):
            self.beam_vector.append(new_laser)
            self.num_of_beams = len(self.beam_vector)
        elif isinstance(new_laser, dict):
            self.beam_vector.append(laserBeam(**new_laser))
            self.num_of_beams = len(self.beam_vector)
        else:
            raise TypeError(
                "new_laser should by type laserBeam or a dictionary"
                + "of arguments to initialize the laserBeam class."
            )

    def pol(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """
        Return the polarization of each of the laser beams at position R and t.

        The polarization is returned in the spherical basis.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        pol : list of array_like, size (3,)
            polarization of each laser beam at R and t in spherical basis.
        """
        if self.num_of_beams == 0:
            return jnp.zeros((0, 3), dtype=jnp.complex64)

        return jnp.stack([beam.pol(R, t) for beam in self.beam_vector])

    def intensity(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """
        Return the intensity of each of the laser beams at position R and t.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        s : list of float or array_like
            Saturation parameters of all laser beams at R and t.
        """
        if self.num_of_beams == 0:
            return jnp.array([], dtype=jnp.float32)
        return jnp.stack([beam.intensity(R, t) for beam in self.beam_vector])

    def kvec(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """
        Return the k-vectors of each of the laser beams.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        kvec : list of array_like, size(3,)
            the k vector at position R and time t for each laser beam.
        """
        if self.num_of_beams == 0:
            return jnp.zeros((0, 3), dtype=jnp.float32)
        return jnp.stack([beam.kvec(R, t) for beam in self.beam_vector])

    def delta(self, t=0.0):
        """
        Return the detuning of each of the laser beams at time t.

        Parameters
        ----------
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        delta : float or array like
            detuning of the laser beam at time t for all laser beams
        """
        if self.num_of_beams == 0:
            return jnp.array([], dtype=jnp.float32)
        return jnp.stack([beam.delta(t) for beam in self.beam_vector])

    def electric_field(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """
        Return the electric field of each of the laser beams.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        E : list of array_like, size(3,)
            the electric field vectors at position R and time t for each laser beam.
        """
        if self.num_of_beams == 0:
            return jnp.zeros((0, 3), dtype=jnp.complex64)

        return jnp.stack([beam.electric_field(R, t) for beam in self.beam_vector])

    def electric_field_gradient(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """
        Return the gradient of the electric field of each of the laser beams.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dE : list of array_like, size(3,)
            the electric field gradient matrices at position R and time t for each laser beam.
        """
        if self.num_of_beams == 0:
            return jnp.zeros((0, 3, 3), dtype=jnp.complex64)

        return jnp.stack([beam.electric_field_gradient(R, t) for beam in self.beam_vector])

    def total_electric_field(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """
        Return the total electric field of the laser beams.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        E : array_like, size(3,)
            the total electric field vector at position R and time t of all
            the laser beams
        """
        if self.num_of_beams == 0:
            return jnp.zeros(3, dtype=jnp.complex64)

        return jnp.sum(self.electric_field(R, t), axis=0)

    def total_electric_field_gradient(self, R=jnp.array([0.0, 0.0, 0.0]), t=0.0):
        """
        Return the total gradient of the electric field of the laser beams.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dE : array_like, size(3,)
            the total electric field gradient matrices at position R and time t
            of all laser beams.
        """
        return jnp.sum(self.electric_field_gradient(R, t), axis=0)

    def project_pol(self, quant_axis, R=jnp.array([0.0, 0.0, 0.0]), t=0.0, **kwargs):
        """
        Project the polarization onto a quantization axis.

        Parameters
        ----------
        quant_axis : jnp.Array, shape (3,)
            A normalized 3-vector of the quantization axis direction.
        R : jnp.Array, shape (3,), optional
            If polarization is a function of R is the
            3-vectors at which the polarization shall be calculated.
        calculate_norm : bool, optional
            If true, renormalizes the quant_axis.  By default, False.
        treat_nans : bool, optional
            If true, every place that nan is encountered, replace with the
            $hat{z}$ axis as the quantization axis.  By default, False.
        invert : bool, optional
            If true, invert the process to project the quantization axis
            onto the specified polarization.

        Returns
        -------
        projected_pol : list of array_like, shape (3,)
            The polarization projected onto the quantization axis for all
            laser beams
        """
        # apply norm so it doesnt rely on it being normalised before
        if self.num_of_beams == 0:
            return jnp.zeros((0, 3), dtype=jnp.complex64)

        norm = jnp.linalg.norm(quant_axis)
        q = quant_axis / jnp.where(norm == 0, 1.0, norm)
        cosbeta = q[2]
        sinbeta = jnp.sqrt(jnp.clip(1 - cosbeta**2, 0, 1))
        gamma = jnp.arctan2(q[1], q[0])
        alpha = 0.0

        D = jnp.array(
            [
                [
                    (1 + cosbeta) / 2 * jnp.exp(-1j * alpha + 1j * gamma),
                    -sinbeta / jnp.sqrt(2) * jnp.exp(-1j * alpha),
                    (1 - cosbeta) / 2 * jnp.exp(-1j * alpha - 1j * gamma),
                ],
                [
                    sinbeta / jnp.sqrt(2) * jnp.exp(1j * gamma),
                    cosbeta,
                    -sinbeta / jnp.sqrt(2) * jnp.exp(-1j * gamma),
                ],
                [
                    (1 - cosbeta) / 2 * jnp.exp(1j * alpha + 1j * gamma),
                    sinbeta / jnp.sqrt(2),
                    (1 + cosbeta) / 2 * jnp.exp(1j * alpha - 1j * gamma),
                ],
            ]
        )

        pols = self.pol(R, t)

        return jnp.einsum("ij, bj -> bi", D, pols)

    def cartesian_pol(self, R=jnp.array([0.0, 0.0, 0.0]), t=0):
        """
        Return the polarization of all laser beams in Cartesian coordinates.

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the polarization.  By default,
            the origin.
        t : float, optional
            time at which to return the polarization.  By default, t=0.

        Returns
        -------
        pol : array_like, shape (num_of_beams, 3)
            polarization of the laser beam at R and t in Cartesian basis.
        """
        if self.num_of_beams == 0:
            return jnp.zeros((0, 3), dtype=jnp.complex64)

        return jnp.stack([beam.cartesian_pol(R, t) for beam in self.beam_vector])

    def jones_vector(self, xp, yp, R=jnp.array([0.0, 0.0, 0.0]), t=0):
        """
        Jones vector at position R and time t.

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the Jones vector.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the Jones vector.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to evaluate the Jones vector.  By default,
            the origin.
        t : float, optional
            time at which to evaluate the Jones vector.  By default, t=0.

        Returns
        -------
        pol : jnp.Array, size (num_of_beams, 2)
            Jones vector of the laser beams at R and t in Cartesian basis.
        """
        if self.num_of_beams == 0:
            return jnp.zeros((0, 2), dtype=jnp.complex64)

        return jnp.stack([beam.jones_vector(xp, yp, R, t) for beam in self.beam_vector])

    def stokes_parameters(self, xp, yp, R=jnp.array([0.0, 0.0, 0.0]), t=0):
        """
        Return the Stokes Parameters of the laser beam at R and t.

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the Stokes parameters.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the Stokes parameters.
            Must be orthogonal to k and `xp`.
        R : jnp.Array, size (3,), optional
            vector of the position at which to calculate the Stokes parameters.
            By default, the origin.
        t : float, optional
            time at which to calculate the Stokes parameters.  By default, t=0.

        Returns
        -------
        pol : jnp.Array, shape (num_of_beams, 3)
            Stokes parameters for the laser beams, [Q, U, V]
        """
        if self.num_of_beams == 0:
            return jnp.zeros((0, 3), dtype=jnp.complex64)

        return jnp.stack([beam.stokes_parameters(xp, yp, R, t) for beam in self.beam_vector])

    def polarization_ellipse(self, xp, yp, R=jnp.array([0.0, 0.0, 0.0]), t=0):
        r"""
        Return the polarization ellipse parameters of the laser beam at R and t.

        Parameters
        ----------
        xp : array_like, shape (3,)
            The x vector of the basis in which to calculate the polarization ellipse.
            Must be orthogonal to k.
        yp : array_like, shape (3,)
            The y vector of the basis in which to calculate the polarization ellipse.
            Must be orthogonal to k and `xp`.
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        list of (psi, chi) : list of tuples
            list of (:math:`\\psi`, :math:`\\chi`) parameters of the
            polarization ellipses for each laser beam
        """
        if self.num_of_beams == 0:
            return jnp.zeros((0, 3), dtype=jnp.complex64)
        return jnp.stack([beam.polarization_ellipse(xp, yp, R, t) for beam in self.beam_vector])


class conventional3DMOTBeams(laserBeams):
    r"""
    A collection of laser beams for 6-beam MOT.

    The standard geometry is to generate counter-progagating beams along all
    orthogonal axes :math:`(\\hat{x}, \\hat{y}, \\hat{z})`.

    Parameters
    ----------
    k : float, optional
        Magnitude of the k-vector for the six laser beams.  Default: 1
    pol : int or float, optional
        Sign of the circular polarization for the beams moving along
        :math:`\\hat{z}`.  Default: +1.  Orthogonal beams have opposite
        polarization by default.
    rotation_angles : array_like
        List of angles to define a rotated MOT.  Default: [0., 0., 0.]
    rotation_spec : str
        String to define the convention of the Euler rotations.  Default: 'ZYZ'
    beam_type : pylcp_jax.laserBeam or subclass
        Type of beam to generate.
    **kwargs :
        other keyword arguments to pass to beam_type
    """

    def __init__(
        self,
        k=1.0,
        pol=+1,
        rotation_angles=[0.0, 0.0, 0.0],
        rotation_spec="ZYZ",
        beam_type=laserBeam,
        **kwargs,
    ):
        super().__init__()

        rot_mat = jnp.array(Rotation.from_euler(rotation_spec, rotation_angles).as_matrix())

        kvecs = [
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([-1.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
            jnp.array([0.0, -1.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
            jnp.array([0.0, 0.0, -1.0]),
        ]
        pols = [-pol, -pol, -pol, -pol, +pol, +pol]

        for kvec, pol in zip(kvecs, pols):
            rotated_kvec = jnp.matmul(rot_mat, k * kvec)
            self.add_laser(beam_type(kvec=rotated_kvec, pol=pol, **kwargs))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    test_field = magField(lambda R: jnp.array([-0.5 * R[0], -0.5 * R[1], 1 * R[2]]))
    print("B-field at origin:\n", test_field.Field(jnp.array([0.0, 0.0, 0.0])))
    print("B-field gradient:\n", test_field.gradField(jnp.array([5.0, 2.0, 1.0])))

    example_beams = laserBeams(
        [
            {
                "kvec": jnp.array([0.0, 0.0, 1.0]),
                "pol": jnp.array([0.0, 0.0, 1.0]),
                "pol_coord": "spherical",
                "delta": -2.0,
                "s": 1.0,
            },
            {
                "kvec": jnp.array([0.0, 0.0, -1.0]),
                "pol": jnp.array([0.0, 0.0, 1.0]),
                "pol_coord": "spherical",
                "delta": -2.0,
                "s": 1.0,
            },
        ],
        beam_type=infinitePlaneWaveBeam,
    )

    print(
        "Jones Vector:\n",
        example_beams.beam_vector[0].jones_vector(
            jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])
        ),
    )
    # 3. Testing the "Stacked" Collection Returns
    print("All kvecs shape:\n", example_beams.kvec().shape)
    print("All pols shape:\n", example_beams.pol().shape)
    print("All intensities shape:\n", example_beams.intensity().shape)
    print(
        "All gradients shape:\n",
        example_beams.electric_field_gradient(jnp.array([0.0, 0.0, 0.0]), 0.5).shape,
    )

    # 4. Gaussian Beam Test
    example_beam_gauss = gaussianBeam(
        kvec=jnp.array([1.0, 0.0, 0.0]), pol=+1, s=5.0, delta=-2.0, wb=1000.0
    )
    print(
        "Gaussian Intensity at edge:\n",
        example_beam_gauss.intensity(jnp.array([0.0, 1000 / jnp.sqrt(2), 1000 / jnp.sqrt(2)])),
    )

    # 5. Infinite Plane Wave Test
    example_beam_plane = infinitePlaneWaveBeam(
        kvec=jnp.array([1.0, 0.0, 0.0]), pol=+1, s=5.0, delta=-2.0
    )
    print(
        "Plane Wave Gradient at origin:\n",
        example_beam_plane.electric_field_gradient(jnp.array([0.0, 0.0, 0.0]), 0.0),
    )

    R_batch = jnp.array(np.random.rand(101, 3))
    t_batch = jnp.linspace(0, 10, 101)

    # jax.vmap passes Array scalars; t: float annotation is correct for normal use
    vmapped_gradient = jax.vmap(example_beam_plane.electric_field_gradient)(
        R_batch,
        t_batch,  # pyright: ignore[reportArgumentType]
    )
    print("Batch Gradient shape (Atoms, 3, 3):\n", vmapped_gradient.shape)

    # 7. MOT Beams Initialization
    MOT_beams = conventional3DMOTBeams(
        k=1.0, pol=1, beam_type=gaussianBeam, wb=1000.0, s=5.0, delta=-2.0
    )
    print("MOT Beam 1 kvec:\n", MOT_beams.beam_vector[1].kvec())

    OT_beams = conventional3DMOTBeams(
        k=1.0, pol=1, beam_type=gaussianBeam, wb=1000.0, s=5.0, delta=-2.0
    )
    print("MOT Beam 1 kvec:\n", MOT_beams.beam_vector[1].kvec())

    # --- Let's actually use plt to plot the Gaussian beam profile! ---

    # Create an array of 400 points along the y-axis from -2000 to +2000
    y_vals = jnp.linspace(-2000, 2000, 400)

    # Stack them into coordinates [0, y, 0] so the shape is (400, 3)
    R_plot = jnp.stack([jnp.zeros_like(y_vals), y_vals, jnp.zeros_like(y_vals)], axis=1)

    # Use vmap to calculate the intensity for all 400 points at once
    intensities = jax.vmap(example_beam_gauss.intensity)(R_plot)

    # Plot it
    plt.figure(figsize=(8, 5))
    plt.plot(y_vals, intensities, label=f"waist = {example_beam_gauss.wb}")
    plt.title("Gaussian Beam Intensity Profile (JAX vmapped)")
    plt.xlabel("Transverse Position (y)")
    plt.ylabel("Saturation Parameter (s)")
    plt.grid(True)
    plt.legend()
    plt.show()
