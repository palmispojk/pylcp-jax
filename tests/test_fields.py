import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pylcp_jax.fields import (
    clippedGaussianBeam,
    constantMagneticField,
    conventional3DMOTBeams,
    gaussianBeam,
    infinitePlaneWaveBeam,
    iPMagneticField,
    laserBeam,
    laserBeams,
    magField,
    promote_to_lambda,
    quadrupoleMagneticField,
)

R0 = jnp.array([0.0, 0.0, 0.0])
R1 = jnp.array([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# promote_to_lambda
# ---------------------------------------------------------------------------


class TestPromoteToLambda:
    def test_scalar_Rt(self):
        func, sig = promote_to_lambda(3.0)
        assert sig == "()"
        assert float(func()) == pytest.approx(3.0)

    def test_list_Rt(self):
        func, sig = promote_to_lambda([0.0, 0.0, 1.0])
        assert sig == "()"
        result = func()
        assert result.shape == (3,)
        assert float(result[2]) == pytest.approx(1.0)

    def test_callable_R_only(self):
        func, sig = promote_to_lambda(lambda R: R * 2.0)
        assert sig == "(R)"
        result = func(R=jnp.array([1.0, 0.0, 0.0]))
        assert float(result[0]) == pytest.approx(2.0)

    def test_callable_Rt(self):
        func, sig = promote_to_lambda(lambda R, t: R + t)
        assert sig == "(R, t)"
        result = func(R=jnp.array([1.0, 0.0, 0.0]), t=1.0)
        assert float(result[0]) == pytest.approx(2.0)

    def test_callable_t_only(self):
        func, sig = promote_to_lambda(lambda t: t * 2)
        assert sig == "(R, t)"
        result = func(t=3.0)
        assert float(result) == pytest.approx(6.0)

    def test_scalar_t_kind(self):
        func, sig = promote_to_lambda(5.0, kind="t")
        assert sig == "()"
        assert float(func()) == pytest.approx(5.0)

    def test_callable_t_kind(self):
        func, sig = promote_to_lambda(lambda t: t**2, kind="t")
        assert float(func(t=3.0)) == pytest.approx(9.0)

    def test_unknown_callable_raises(self):
        with pytest.raises(TypeError):
            promote_to_lambda(lambda a, b, c: a, kind="Rt")


# ---------------------------------------------------------------------------
# magField
# ---------------------------------------------------------------------------


class TestMagField:
    def test_constant_array(self):
        B = magField(jnp.array([0.0, 0.0, 1.0]))
        result = B.Field(R0, 0.0)
        assert jnp.allclose(result, jnp.array([0.0, 0.0, 1.0]))

    def test_callable_Rt(self):
        B = magField(lambda R, t: jnp.array([R[0], 0.0, 0.0]))
        result = B.Field(jnp.array([3.0, 0.0, 0.0]), 0.0)
        assert float(result[0]) == pytest.approx(3.0)

    def test_field_magnitude(self):
        B = magField(jnp.array([3.0, 4.0, 0.0]))
        assert float(B.FieldMag(R0, 0.0)) == pytest.approx(5.0)

    def test_grad_field_constant_zero(self):
        # Gradient of a constant field should be zero
        B = magField(jnp.array([0.0, 0.0, 1.0]))
        grad = B.gradField(R0)
        assert jnp.allclose(grad, jnp.zeros((3, 3)), atol=1e-5)

    def test_grad_field_mag_constant_zero(self):
        B = magField(jnp.array([0.0, 0.0, 1.0]))
        dB = B.gradFieldMag(R0, 0.0)
        assert jnp.allclose(dB, jnp.zeros(3), atol=1e-5)


class TestConstantMagneticField:
    def test_returns_array(self):
        B = constantMagneticField(jnp.array([1.0, 2.0, 3.0]))
        result = B.Field(R1, 1.0)
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))

    def test_zero_gradient(self):
        B = constantMagneticField(jnp.array([0.0, 1.0, 0.0]))
        grad = B.gradField(R1)
        assert jnp.allclose(grad, jnp.zeros((3, 3)), atol=1e-5)


class TestQuadrupoleMagneticField:
    """Quadrupole magnetic field B⃗ = α·(−x/2, −y/2, z).

    The field vanishes at the origin and increases linearly in all
    directions. Maxwell's ∇·B⃗ = 0 constraint fixes the radial gradient
    to half the axial gradient with opposite sign."""

    def test_origin_is_zero(self):
        B = quadrupoleMagneticField(1.0)
        assert jnp.allclose(B.Field(R0, 0.0), jnp.zeros(3))

    def test_linear_scaling(self):
        B = quadrupoleMagneticField(2.0)
        r = jnp.array([1.0, 0.0, 0.0])
        result = B.Field(r, 0.0)
        assert float(result[0]) == pytest.approx(-1.0)  # alpha * (-0.5 * 1)
        assert float(result[2]) == pytest.approx(0.0)

    def test_z_component(self):
        B = quadrupoleMagneticField(1.0)
        r = jnp.array([0.0, 0.0, 1.0])
        assert float(B.Field(r, 0.0)[2]) == pytest.approx(1.0)


class TestIPMagneticField:
    """Ioffe-Pritchard magnetic field.

    At the origin the field is a uniform bias B₀ along ẑ.  Away from
    the origin, linear (B₁) and quadratic (B₂) gradients add radial
    and axial confinement, producing the characteristic IP trap
    potential used for magnetic trapping of neutral atoms."""

    def test_origin_gives_B0_along_z(self):
        B = iPMagneticField(B0=1.0, B1=0.5, B2=0.1)
        result = B.Field(R0, 0.0)
        assert float(result[2]) == pytest.approx(1.0)
        assert float(result[0]) == pytest.approx(0.0)
        assert float(result[1]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# laserBeam
# ---------------------------------------------------------------------------


class TestLaserBeam:
    """Single laser beam: polarization, intensity, detuning, and fields.

    A laserBeam is defined by its wave vector k⃗ (propagation direction
    and magnitude |k| = 2π/λ), polarization (σ⁺, σ⁻, or arbitrary
    spherical/Cartesian vector), saturation parameter s, and detuning δ
    from resonance.  Polarization must be transverse (ε⃗ ⊥ k⃗)."""

    def test_default_construction(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=-2.0)
        assert float(beam.intensity()) == pytest.approx(1.0)
        assert float(beam.delta()) == pytest.approx(-2.0)

    def test_pol_int_positive(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        pol = beam.pol()
        assert pol.shape == (3,)
        # For +1 pol along z, sigma+ -> spherical component [1] should dominate
        assert jnp.linalg.norm(pol) == pytest.approx(1.0, abs=1e-5)

    def test_pol_int_negative(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=-1, s=1.0, delta=0.0)
        pol = beam.pol()
        assert pol.shape == (3,)
        assert jnp.linalg.norm(pol) == pytest.approx(1.0, abs=1e-5)

    def test_pol_spherical_array(self):
        # Pure z-component in spherical: [0, 1, 0] represents pi polarization
        pol_vec = jnp.array([0.0, 1.0, 0.0], dtype=jnp.complex64)
        beam = laserBeam(kvec=[1.0, 0.0, 0.0], pol=pol_vec, s=1.0, delta=0.0, pol_coord="spherical")
        assert jnp.linalg.norm(beam.pol()) == pytest.approx(1.0, abs=1e-5)

    def test_pol_cartesian_array_converts_to_spherical(self):
        # Cartesian z-hat [0, 0, 1] should become spherical [0, 1, 0] (pi light)
        beam_cart = laserBeam(
            kvec=[1.0, 0.0, 0.0],
            pol=np.array([0.0, 0.0, 1.0]),
            s=1.0,
            delta=0.0,
            pol_coord="cartesian",
        )
        beam_sph = laserBeam(
            kvec=[1.0, 0.0, 0.0],
            pol=np.array([0.0, 1.0, 0.0]),
            s=1.0,
            delta=0.0,
            pol_coord="spherical",
        )
        assert jnp.allclose(beam_cart.pol(), beam_sph.pol(), atol=1e-5)

    def test_callable_intensity(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=lambda R, t: float(R[2] ** 2), delta=0.0)
        assert float(beam.intensity(jnp.array([0.0, 0.0, 2.0]), 0.0)) == pytest.approx(4.0)

    def test_callable_delta(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=lambda t: -t)
        assert float(beam.delta(3.0)) == pytest.approx(-3.0)

    def test_delta_phase(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=-2.0)
        assert float(beam.delta_phase(t=1.0)) == pytest.approx(-2.0)

    def test_delta_phase_callable_raises(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=lambda t: -t)
        with pytest.raises(NotImplementedError):
            beam.delta_phase(t=1.0)

    def test_kvec_returned(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        assert jnp.allclose(beam.kvec(), jnp.array([0.0, 0.0, 1.0]))

    def test_cartesian_pol_shape(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        cp = beam.cartesian_pol()
        assert cp.shape == (3,)

    def test_jones_vector(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        jv = beam.jones_vector(jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
        assert jv.shape == (2,)

    def test_stokes_parameters(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        Q, U, V = beam.stokes_parameters(jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
        # For pure circular polarization, Q=U=0 and V has magnitude 1
        assert float(jnp.abs(V)) == pytest.approx(1.0, abs=1e-4)

    def test_polarization_ellipse(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        psi, chi = beam.polarization_ellipse(jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
        # Pure circular pol: chi should be ±pi/4
        assert float(jnp.abs(chi)) == pytest.approx(jnp.pi / 4, abs=1e-4)

    def test_electric_field_shape(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        E = beam.electric_field(R0, 0.0)
        assert E.shape == (3,)

    def test_electric_field_gradient_shape(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        dE = beam.electric_field_gradient(R0, 0.0)
        assert dE.shape == (3, 3)

    def test_project_pol_shape(self):
        beam = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        pp = beam.project_pol(jnp.array([0.0, 0.0, 1.0]))
        assert pp.shape == (3,)

    def test_project_pol_identity_z(self):
        # Projecting onto z-axis should leave z-axis polarization unchanged
        beam = laserBeam(
            kvec=[1.0, 0.0, 0.0],
            pol=jnp.array([0.0, 1.0, 0.0], dtype=jnp.complex64),
            s=1.0,
            delta=0.0,
            pol_coord="spherical",
        )
        pp = beam.project_pol(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.allclose(jnp.abs(pp), jnp.abs(beam.pol()), atol=1e-5)


# ---------------------------------------------------------------------------
# infinitePlaneWaveBeam
# ---------------------------------------------------------------------------


class TestInfinitePlaneWaveBeam:
    def test_construction(self):
        beam = infinitePlaneWaveBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=2.0, delta=-1.0)
        assert float(beam.intensity()) == pytest.approx(2.0)

    def test_callable_kvec_raises(self):
        with pytest.raises(TypeError):
            infinitePlaneWaveBeam(
                kvec=lambda R, t: jnp.array([0.0, 0.0, 1.0]), pol=+1, s=1.0, delta=0.0
            )

    def test_callable_s_accepted(self):
        # Callable s(R, t) is allowed (used by the SF red MOT power ramp).
        infinitePlaneWaveBeam(
            kvec=[0.0, 0.0, 1.0],
            pol=+1,
            s=lambda R, t: 1.0,
            delta=0.0,
        )

    def test_callable_s_intensity(self):
        beam = infinitePlaneWaveBeam(
            kvec=[0.0, 0.0, 1.0],
            pol=+1,
            s=lambda R, t: 2.0 * t,
            delta=0.0,
        )
        assert float(beam.intensity(R0, 1.5)) == pytest.approx(3.0)

    def test_callable_s_ramp_over_time(self):
        # Linear ramp from s_start -> s_end over t_ramp, then held flat
        # (mirrors the SF red MOT power ramp).
        s_start, s_end, t_ramp = 1270.0, 127.0, 3000.0

        def s_ramp(R, t):
            frac = jnp.minimum(t / t_ramp, 1.0)
            return s_start + (s_end - s_start) * frac

        beam = infinitePlaneWaveBeam(
            kvec=[0.0, 0.0, 1.0],
            pol=+1,
            s=s_ramp,
            delta=0.0,
        )
        assert float(beam.intensity(R0, 0.0)) == pytest.approx(s_start)
        assert float(beam.intensity(R0, t_ramp / 2)) == pytest.approx((s_start + s_end) / 2)
        assert float(beam.intensity(R0, t_ramp)) == pytest.approx(s_end)
        assert float(beam.intensity(R0, 2 * t_ramp)) == pytest.approx(s_end)

    def test_electric_field_gradient_shape(self):
        beam = infinitePlaneWaveBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        dE = beam.electric_field_gradient(R0, 0.0)
        assert dE.shape == (3, 3)

    def test_gradient_analytic_vs_jax(self):
        # The analytic gradient (-i * outer(k, E)) should equal jacfwd transposed
        # to match the [grad_dir, field_comp] convention: M[i,j] = dE_j/dR_i.
        # Use x-beam so the gradient matrix is non-symmetric (z-beam is diagonal
        # and passes trivially even with wrong convention).
        beam = infinitePlaneWaveBeam(kvec=[1.0, 0.0, 0.0], pol=+1, s=1.0, delta=0.0)
        dE_analytic = beam.electric_field_gradient(R0, 0.0)
        dE_jac = jax.jacfwd(lambda R: beam.electric_field(R, 0.0))(R0)
        assert jnp.allclose(dE_analytic, dE_jac.T, atol=1e-5)

    @pytest.mark.parametrize("kvec", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    def test_gradient_matches_base_laserBeam(self, kvec):
        # infinitePlaneWaveBeam and base laserBeam must return the same gradient
        # convention [grad_dir, field_comp] for all beam directions.
        ipw = infinitePlaneWaveBeam(kvec=kvec, pol=+1, s=1.0, delta=0.0)
        base = laserBeam(kvec=kvec, pol=+1, s=1.0, delta=0.0)
        dE_ipw = ipw.electric_field_gradient(R0, 0.0)
        dE_base = base.electric_field_gradient(R0, 0.0)
        assert jnp.allclose(dE_ipw, dE_base, atol=1e-5), (
            f"Gradient mismatch for kvec={kvec}:\n"
            f"infinitePlaneWaveBeam:\n{dE_ipw}\nlaserBeam:\n{dE_base}"
        )


# ---------------------------------------------------------------------------
# gaussianBeam
# ---------------------------------------------------------------------------


class TestGaussianBeam:
    """Gaussian beam with intensity profile I(r) = s · exp(−2r²/w²).

    The beam waist w defines the 1/e² intensity radius.  At
    r = w/√2 the intensity drops to s·e⁻¹."""

    def test_peak_intensity(self):
        beam = gaussianBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=5.0, delta=0.0, wb=1.0)
        assert float(beam.intensity(R0)) == pytest.approx(5.0)

    def test_half_max_intensity(self):
        # At r = wb/sqrt(2) from axis, I = s_max * exp(-1)
        wb = 2.0
        beam = gaussianBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0, wb=wb)
        r = jnp.array([wb / jnp.sqrt(2), 0.0, 0.0])
        expected = float(jnp.exp(-1.0))
        assert float(beam.intensity(r)) == pytest.approx(expected, rel=1e-4)

    def test_callable_kvec_raises(self):
        with pytest.raises(TypeError):
            gaussianBeam(
                kvec=lambda R, t: jnp.array([0.0, 0.0, 1.0]), pol=+1, s=1.0, delta=0.0, wb=1.0
            )

    def test_electric_field_gradient_shape(self):
        beam = gaussianBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0, wb=10.0)
        dE = beam.electric_field_gradient(R0, 0.0)
        assert dE.shape == (3, 3)


# ---------------------------------------------------------------------------
# clippedGaussianBeam
# ---------------------------------------------------------------------------


class TestClippedGaussianBeam:
    def test_inside_clip(self):
        beam = clippedGaussianBeam(
            kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0, wb=100.0, rs=10.0
        )
        # At origin: well inside clip, intensity ≈ s_max
        assert float(beam.intensity(R0)) == pytest.approx(1.0, rel=1e-4)

    def test_outside_clip(self):
        beam = clippedGaussianBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0, wb=100.0, rs=1.0)
        # At r=5 >> rs=1: intensity should be zero
        r = jnp.array([5.0, 0.0, 0.0])
        assert float(beam.intensity(r)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# laserBeams collection
# ---------------------------------------------------------------------------


class TestLaserBeams:
    def setup_method(self):
        self.beams = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": -2.0},
                {"kvec": [0.0, 0.0, -1.0], "pol": -1, "s": 2.0, "delta": -2.0},
            ]
        )

    def test_num_of_beams(self):
        assert self.beams.num_of_beams == 2

    def test_kvec_shape(self):
        assert self.beams.kvec().shape == (2, 3)

    def test_pol_shape(self):
        assert self.beams.pol().shape == (2, 3)

    def test_intensity_shape(self):
        s = self.beams.intensity()
        assert s.shape == (2,)
        assert float(s[0]) == pytest.approx(1.0)
        assert float(s[1]) == pytest.approx(2.0)

    def test_delta_shape(self):
        d = self.beams.delta()
        assert d.shape == (2,)
        assert jnp.allclose(d, jnp.array([-2.0, -2.0]))

    def test_electric_field_shape(self):
        E = self.beams.electric_field(R0, 0.0)
        assert E.shape == (2, 3)

    def test_electric_field_gradient_shape(self):
        dE = self.beams.electric_field_gradient(R0, 0.0)
        assert dE.shape == (2, 3, 3)

    def test_total_electric_field_shape(self):
        E = self.beams.total_electric_field(R0, 0.0)
        assert E.shape == (3,)

    def test_total_electric_field_gradient_shape(self):
        dE = self.beams.total_electric_field_gradient(R0, 0.0)
        assert dE.shape == (3, 3)

    def test_total_equals_sum(self):
        E_total = self.beams.total_electric_field(R0, 0.0)
        E_sum = jnp.sum(self.beams.electric_field(R0, 0.0), axis=0)
        assert jnp.allclose(E_total, E_sum)

    def test_project_pol_shape(self):
        pp = self.beams.project_pol(jnp.array([0.0, 0.0, 1.0]))
        assert pp.shape == (2, 3)

    def test_cartesian_pol_shape(self):
        cp = self.beams.cartesian_pol()
        assert cp.shape == (2, 3)

    def test_jones_vector_shape(self):
        # jones_vector requires xp, yp, k to form a right-handed system;
        # use a collection where all beams share the same +z direction
        same_dir = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": -2.0},
                {"kvec": [0.0, 0.0, 1.0], "pol": -1, "s": 2.0, "delta": -2.0},
            ]
        )
        jv = same_dir.jones_vector(jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
        assert jv.shape == (2, 2)

    def test_stokes_parameters_shape(self):
        same_dir = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": -2.0},
                {"kvec": [0.0, 0.0, 1.0], "pol": -1, "s": 2.0, "delta": -2.0},
            ]
        )
        sp = same_dir.stokes_parameters(jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
        assert sp.shape == (2, 3)

    def test_empty_beams_shapes(self):
        empty = laserBeams([])
        assert empty.kvec().shape == (0, 3)
        assert empty.pol().shape == (0, 3)
        assert empty.intensity().shape == (0,)
        assert empty.delta().shape == (0,)
        assert empty.electric_field(R0, 0.0).shape == (0, 3)  # each beam returns (3,)
        assert empty.electric_field_gradient(R0, 0.0).shape == (0, 3, 3)
        assert empty.total_electric_field(R0, 0.0).shape == (3,)
        assert empty.project_pol(jnp.array([0.0, 0.0, 1.0])).shape == (0, 3)

    def test_add_operator(self):
        beams2 = laserBeams([{"kvec": [1.0, 0.0, 0.0], "pol": +1, "s": 1.0, "delta": 0.0}])
        combined = self.beams + beams2
        assert combined.num_of_beams == 3

    def test_iadd_operator(self):
        beams_copy = laserBeams(
            [
                {"kvec": [0.0, 0.0, 1.0], "pol": +1, "s": 1.0, "delta": -2.0},
            ]
        )
        beams2 = laserBeams([{"kvec": [1.0, 0.0, 0.0], "pol": +1, "s": 1.0, "delta": 0.0}])
        beams_copy += beams2
        assert beams_copy.num_of_beams == 2

    def test_add_laser_beam_instance(self):
        b = laserBeams([])
        b.add_laser(laserBeam(kvec=[1.0, 0.0, 0.0], pol=+1, s=1.0, delta=0.0))
        assert b.num_of_beams == 1

    def test_add_laser_dict(self):
        b = laserBeams([])
        b.add_laser({"kvec": [1.0, 0.0, 0.0], "pol": +1, "s": 1.0, "delta": 0.0})
        assert b.num_of_beams == 1

    def test_add_laser_invalid_raises(self):
        b = laserBeams([])
        with pytest.raises(TypeError):
            b.add_laser("not_a_beam")

    def test_invalid_param_type_raises(self):
        with pytest.raises(TypeError):
            laserBeams(["not_a_beam_or_dict"])


# ---------------------------------------------------------------------------
# conventional3DMOTBeams
# ---------------------------------------------------------------------------


class TestConventional3DMOTBeams:
    """Standard 3D magneto-optical trap (MOT) beam configuration.

    Six beams form three counter-propagating pairs along ±x̂, ±ŷ, ±ẑ.
    The sum of all k⃗-vectors vanishes (balanced radiation pressure at
    rest). Each pair has opposite circular polarization to create the
    position-dependent force needed for trapping in a quadrupole B-field."""

    def test_six_beams(self):
        mot = conventional3DMOTBeams(k=1.0, pol=+1, s=1.0, delta=-2.0)
        assert mot.num_of_beams == 6

    def test_counter_propagating_pairs(self):
        mot = conventional3DMOTBeams(k=1.0, pol=+1, s=1.0, delta=-2.0)
        kvecs = mot.kvec()  # shape (6, 3)
        # Sum of all k-vectors should be zero (counter-propagating pairs)
        assert jnp.allclose(jnp.sum(kvecs, axis=0), jnp.zeros(3), atol=1e-6)

    def test_k_magnitude(self):
        k = 2.5
        mot = conventional3DMOTBeams(k=k, pol=+1, s=1.0, delta=-2.0)
        kvecs = mot.kvec()
        norms = jnp.linalg.norm(kvecs, axis=1)
        assert jnp.allclose(norms, k * jnp.ones(6), atol=1e-5)

    def test_with_gaussian_beam_type(self):
        mot = conventional3DMOTBeams(
            k=1.0, pol=+1, beam_type=gaussianBeam, wb=100.0, s=1.0, delta=-2.0
        )
        assert mot.num_of_beams == 6
        # Peak intensity at origin should be s_max
        assert float(mot.beam_vector[0].intensity(R0)) == pytest.approx(1.0, rel=1e-4)

    def test_rotation(self):
        # A 90 deg rotation around Z should permute the x/y beam directions
        mot_rot = conventional3DMOTBeams(
            k=1.0, pol=+1, s=1.0, delta=-2.0, rotation_angles=[90.0, 0.0, 0.0], rotation_spec="ZYZ"
        )
        mot_orig = conventional3DMOTBeams(k=1.0, pol=+1, s=1.0, delta=-2.0)
        # The z beams (indices 4, 5) should be unchanged
        assert jnp.allclose(
            mot_rot.beam_vector[4].kvec(), mot_orig.beam_vector[4].kvec(), atol=1e-5
        )


# ---------------------------------------------------------------------------
# vmap compatibility
# ---------------------------------------------------------------------------


class TestVmapCompatibility:
    def test_vmap_electric_field(self):
        beam = infinitePlaneWaveBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        Rs = jnp.ones((10, 3))
        ts = jnp.zeros(10)
        E_batch = jax.vmap(beam.electric_field)(Rs, ts)
        assert E_batch.shape == (10, 3)

    def test_vmap_gaussian_intensity(self):
        beam = gaussianBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0, wb=10.0)
        Rs = jnp.zeros((5, 3))
        intensities = jax.vmap(beam.intensity)(Rs)
        assert intensities.shape == (5,)
        assert jnp.allclose(intensities, jnp.ones(5), atol=1e-5)

    def test_vmap_electric_field_gradient(self):
        beam = infinitePlaneWaveBeam(kvec=[0.0, 0.0, 1.0], pol=+1, s=1.0, delta=0.0)
        Rs = jnp.zeros((4, 3))
        ts = jnp.zeros(4)
        dE_batch = jax.vmap(beam.electric_field_gradient)(Rs, ts)
        assert dE_batch.shape == (4, 3, 3)


# ---------------------------------------------------------------------------
# Polarization projection – comprehensive physics tests
# ---------------------------------------------------------------------------


class TestPolarizationProjection:
    """Polarization projection onto a quantization axis.

    When a laser beam interacts with an atom, the relevant quantity is the
    decomposition of its polarization into spherical components
    (ε₋₁, ε₀, ε₊₁) — i.e. (σ⁻, π, σ⁺) — defined with respect to the
    local quantization axis (usually set by the magnetic field direction).

    The decomposition is performed via Wigner rotation matrices (spherical
    tensor algebra) and must satisfy:

    - Unitarity: |ε₋₁|² + |ε₀|² + |ε₊₁|² = 1 for any quantization axis.
    - Consistency: different but mathematically equivalent polarization
      definitions (integer flag, spherical vector, Cartesian vector) must
      yield the same physical projection.
    - Correct limits: σ⁺ light propagating along ẑ with quantization axis
      ẑ must project purely onto ε₊₁.

    Adapted from tests/lasers/00_polarization_projection.py.
    """

    def test_two_sigma_plus_definitions_agree(self):
        """Two equivalent ways to define σ⁺ light (integer flag and spherical
        coordinates) must give the same polarization projection."""
        laser_int = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1)
        laser_sph = laserBeam(
            kvec=[0.0, 0.0, 1.0], pol=np.array([0.0, 0.0, 1.0]), pol_coord="spherical"
        )

        ths = np.linspace(0, np.pi, 21)
        quant_axes = np.array([np.sin(ths), np.zeros(ths.shape), np.cos(ths)])

        p1 = np.array(laser_int.project_pol(quant_axes))
        p2 = np.array(laser_sph.project_pol(quant_axes))

        np.testing.assert_allclose(np.abs(p1) ** 2, np.abs(p2) ** 2, atol=1e-10)

    def test_sigma_plus_along_z_axis_is_pure(self):
        """σ+ light with k along z, quantization axis along z, should be
        purely σ+ (component index 2)."""
        laser = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1)
        proj = np.array(laser.project_pol(np.array([[0.0], [0.0], [1.0]])))
        # σ- (idx 0) and π (idx 1) should be zero, σ+ (idx 2) should be 1
        assert float(np.abs(proj[0, 0]) ** 2) == pytest.approx(0.0, abs=1e-10)
        assert float(np.abs(proj[1, 0]) ** 2) == pytest.approx(0.0, abs=1e-10)
        assert float(np.abs(proj[2, 0]) ** 2) == pytest.approx(1.0, abs=1e-10)

    def test_sigma_minus_along_z_axis_is_pure(self):
        """σ- light with k along z, quantization axis along z, should be
        purely σ- (component index 0)."""
        laser = laserBeam(kvec=[0.0, 0.0, 1.0], pol=-1)
        proj = np.array(laser.project_pol(np.array([[0.0], [0.0], [1.0]])))
        assert float(np.abs(proj[0, 0]) ** 2) == pytest.approx(1.0, abs=1e-10)
        assert float(np.abs(proj[1, 0]) ** 2) == pytest.approx(0.0, abs=1e-10)
        assert float(np.abs(proj[2, 0]) ** 2) == pytest.approx(0.0, abs=1e-10)

    def test_pol_components_sum_to_one(self):
        """The sum of |σ-|² + |π|² + |σ+|² must equal 1 for any quantization axis."""
        laser = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1)
        ths = np.linspace(0, np.pi, 31)
        quant_axes = np.array([np.sin(ths), np.zeros(ths.shape), np.cos(ths)])
        proj = np.array(laser.project_pol(quant_axes))
        total = np.sum(np.abs(proj) ** 2, axis=0)
        np.testing.assert_allclose(total, np.ones(len(ths)), atol=1e-10)

    def test_linear_pol_perpendicular_to_kvec(self):
        """X-polarized light propagating along ẑ, projected onto the ẑ axis,
        should give equal σ⁺ and σ⁻ with no π component.

        Linear polarization along x̂ can be written as:
            ε̂_x = -(ε̂₊₁ - ε̂₋₁)/√2
        so the squared amplitudes for σ⁺ and σ⁻ are each 1/2."""
        laser = laserBeam(
            kvec=[0.0, 0.0, 1.0], pol=np.array([1.0, 0.0, 0.0]), pol_coord="cartesian"
        )
        proj = np.array(laser.project_pol(np.array([[0.0], [0.0], [1.0]])))
        # Linear polarization decomposes into equal σ+ and σ-
        assert float(np.abs(proj[0, 0]) ** 2) == pytest.approx(
            float(np.abs(proj[2, 0]) ** 2), abs=1e-10
        )
        assert float(np.abs(proj[1, 0]) ** 2) == pytest.approx(0.0, abs=1e-10)

    def test_linear_pol_y_kvec_x(self):
        """Y-polarized light propagating along x, projected onto z-axis,
        should give pure π (y is perpendicular to both k and z-axis)."""
        laser = laserBeam(
            kvec=[1.0, 0.0, 0.0], pol=np.array([0.0, 1.0, 0.0]), pol_coord="cartesian"
        )
        proj = np.array(laser.project_pol(np.array([[0.0], [0.0], [1.0]])))
        # Y-polarization with k along x: when quantization axis is z,
        # y-polarization decomposes into equal σ+ and σ-
        assert float(np.abs(proj[0, 0]) ** 2) == pytest.approx(
            float(np.abs(proj[2, 0]) ** 2), abs=1e-10
        )

    def test_pol_projection_rotated_quant_axis(self):
        """Unitarity must hold when the quantization axis is swept continuously.

        As the quantization axis rotates from ẑ through x̂ (or ŷ) and back,
        the individual σ⁺/π/σ⁻ amplitudes redistribute, but their squared
        sum must remain exactly 1.  This tests the Wigner D-matrix
        implementation for non-trivial rotation angles. The atol=1e-6
        accounts for JAX float32 arithmetic."""
        laser = laserBeam(kvec=[0.0, 0.0, 1.0], pol=+1)
        ths = np.linspace(0, np.pi, 51)
        # Sweep quantization axis in the xz-plane
        quant_axes_xz = np.array([np.sin(ths), np.zeros(ths.shape), np.cos(ths)])
        proj_xz = np.array(laser.project_pol(quant_axes_xz))

        # Sweep quantization axis in the yz-plane
        quant_axes_yz = np.array([np.zeros(ths.shape), np.sin(ths), np.cos(ths)])
        proj_yz = np.array(laser.project_pol(quant_axes_yz))

        # Both should conserve total polarization (float32 precision from JAX)
        total_xz = np.sum(np.abs(proj_xz) ** 2, axis=0)
        total_yz = np.sum(np.abs(proj_yz) ** 2, axis=0)
        np.testing.assert_allclose(total_xz, np.ones(len(ths)), atol=1e-6)
        np.testing.assert_allclose(total_yz, np.ones(len(ths)), atol=1e-6)
