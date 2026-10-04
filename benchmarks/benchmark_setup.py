"""
Benchmark: generate_force_profile setup time (set_initial_rho_from_rateeq).

Measures how long it takes to call set_initial_rho_from_rateeq() N times,
which is the setup loop in generate_force_profile.
"""
import time
import numpy as np
import scipy.constants as cts
import pylcp_jax

# ── Build the F2→F3 problem (same as the molasses notebook) ────────────────
atom = pylcp_jax.atom('23Na')
mass = (atom.state[2].gamma * atom.mass) / (cts.hbar * (100 * 2 * np.pi * atom.transition[1].k) ** 2)

det = -2.5
s = 1.0

Hg, Bgq = pylcp_jax.hamiltonians.singleF(F=2, gF=0, muB=1)
He, Beq = pylcp_jax.hamiltonians.singleF(F=3, gF=1/3, muB=1)
dijq = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(2, 3)
hamiltonian = pylcp_jax.hamiltonian(Hg, -det * np.eye(He.shape[0]) + He, Bgq, Beq, dijq, mass=mass)

laserBeams = pylcp_jax.laserBeams([
    {'kvec': np.array([0., 0., 1.]), 'pol': np.array([0., 0., 1.]),
     'pol_coord': 'spherical', 'delta': 0., 's': s},
    {'kvec': np.array([0., 0., -1.]), 'pol': np.array([1., 0., 0.]),
     'pol_coord': 'spherical', 'delta': 0., 's': s},
], beam_type=pylcp_jax.infinitePlaneWaveBeam)

magField = pylcp_jax.constantMagneticField(np.array([0., 0., 0.]))

o = pylcp_jax.obe(laserBeams, magField, hamiltonian,
              include_mag_forces=False, transform_into_re_im=True)

# ── Velocity grid (same size as notebook figure 6: 21 points) ──────────────
v = np.concatenate((np.array([0.0]), np.logspace(-2, np.log10(4), 20))) / np.sqrt(mass)
N = len(v)
print(f"Velocity points: {N}")

r0 = np.zeros((3,))

def setup_rho(vi):
    o.set_initial_position_and_velocity(r0, np.array([0., 0., vi]))
    o.set_initial_rho_from_rateeq()

# ── Warm up: first call compiles JAX kernels ────────────────────────────────
print("\nWarming up (first diag_static_field call + JIT compile) …")
t0 = time.perf_counter()
setup_rho(v[0])
t_warmup = time.perf_counter() - t0
print(f"  Warmup: {t_warmup:.3f} s")

# ── Time N-1 subsequent calls (all with same B=0) ───────────────────────────
print(f"\nTiming {N - 1} subsequent set_initial_rho_from_rateeq calls …")
t0 = time.perf_counter()
for vi in v[1:]:
    setup_rho(vi)
t_loop = time.perf_counter() - t0
print(f"  Total: {t_loop:.3f} s")
print(f"  Per call: {t_loop / (N - 1) * 1000:.1f} ms")

# ── Directly time diag_static_field with and without cache ─────────────────
print("\n── diag_static_field micro-benchmark ──")
ham = o.hamiltonian  # the hamiltonian object used by rateeq inside obe

# Force a cold call (clear cache)
if hasattr(ham, '_last_diag_B'):
    del ham._last_diag_B

t0 = time.perf_counter()
ham.diag_static_field(0.0)
t_cold = time.perf_counter() - t0
print(f"  Cold call (no cache): {t_cold * 1000:.1f} ms")

t0 = time.perf_counter()
for _ in range(N - 1):
    ham.diag_static_field(0.0)
t_cached = time.perf_counter() - t0
print(f"  {N-1} cached calls (same B=0): {t_cached * 1000:.2f} ms total  ({t_cached / (N - 1) * 1e6:.1f} µs each)")

print(f"\n  Speedup per repeated call: {t_cold / (t_cached / (N - 1)):.0f}×")
print(f"  Total setup savings for {N} points: {t_cold * (N - 1) * 1000:.0f} ms → {t_cached * 1000:.2f} ms")
