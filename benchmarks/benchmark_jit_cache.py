"""Benchmark: JIT cache effectiveness for evolve_motion.

Measures time for consecutive evolve_motion calls on the same OBE instance.
With effective JIT caching, the second call should be significantly faster
because it reuses the compiled XLA kernel instead of recompiling.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

import time
import numpy as np
import jax
import jax.numpy as jnp
import pylcp_jax


def setup_obe():
    """Create a simple F=0 -> F'=1 OBE."""
    DET = -2.5
    S = 1.25
    ALPHA = 1e-4

    Hg, Bgq = pylcp_jax.hamiltonians.singleF(F=0, gF=0, muB=1)
    He, Beq = pylcp_jax.hamiltonians.singleF(F=1, gF=1, muB=1)
    dijq = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(0, 1)
    hamiltonian = pylcp_jax.hamiltonian(
        Hg, -DET * np.eye(3) + He, Bgq, Beq, dijq, mass=100
    )
    laserBeams = pylcp_jax.conventional3DMOTBeams(
        s=S, delta=0., beam_type=pylcp_jax.infinitePlaneWaveBeam
    )
    magField = pylcp_jax.quadrupoleMagneticField(ALPHA)
    return pylcp_jax.obe(laserBeams, magField, hamiltonian, transform_into_re_im=True)


def make_batch(obe, n_atoms, seed=42):
    rng = np.random.default_rng(seed)
    r0 = rng.standard_normal((n_atoms, 3)) * 0.1
    v0 = rng.standard_normal((n_atoms, 3)) * 0.01
    # Initialize rho0 via the OBE's own method
    obe.set_initial_rho_equally()
    rho0 = np.tile(np.asarray(obe.rho0), (n_atoms, 1))
    y0 = jnp.array(np.concatenate([rho0, v0, r0], axis=1))
    keys = jax.random.split(jax.random.PRNGKey(seed), n_atoms)
    return y0, keys


def benchmark():
    backend = jax.default_backend()
    print(f"Backend: {backend}")
    print(f"Devices: {jax.devices()}")

    obe = setup_obe()
    N_ATOMS = 8
    MAX_STEPS = 2000
    T_SPAN = [0, 2 * np.pi * 100]
    N_CALLS = 4

    resolved_backend = 'gpu' if backend == 'gpu' else 'cpu'
    kw = dict(
        freeze_axis=[True, True, False],
        random_recoil=True,
        max_scatter_probability=0.1,
        max_steps=MAX_STEPS,
        backend=resolved_backend,
    )

    print(f"\n{'='*60}")
    print(f"  JIT Cache Benchmark")
    print(f"  Atoms: {N_ATOMS}, max_steps: {MAX_STEPS}")
    print(f"  t_span: {T_SPAN}, backend: {resolved_backend}")
    print(f"{'='*60}")

    # --- Test 1: repeated calls with same parameters ---
    print(f"\n--- Test 1: {N_CALLS} consecutive calls (same parameters) ---")
    times = []
    for i in range(N_CALLS):
        y0, keys = make_batch(obe, N_ATOMS, seed=42 + i)
        t0 = time.monotonic()
        obe.evolve_motion(T_SPAN, y0_batch=y0, keys_batch=keys, **kw)
        t1 = time.monotonic()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f"  Call {i+1}: {elapsed:.3f}s")

    speedup_2 = times[0] / times[1] if times[1] > 0 else float('inf')
    print(f"\n  Cold JIT (call 1):     {times[0]:.3f}s")
    print(f"  Warm cache (call 2):   {times[1]:.3f}s")
    print(f"  Speedup call1/call2:   {speedup_2:.1f}x")
    print(f"  JIT cache effective:   {'YES' if speedup_2 > 2.0 else 'NO (recompiles each call)'}")

    # --- Test 2: different freeze_axis (should reuse JIT if using args) ---
    print(f"\n--- Test 2: different freeze_axis ---")
    y0, keys = make_batch(obe, N_ATOMS, seed=100)
    t0 = time.monotonic()
    obe.evolve_motion(T_SPAN, y0_batch=y0, keys_batch=keys,
                      freeze_axis=[False, False, True],
                      random_recoil=True, max_scatter_probability=0.2,
                      max_steps=MAX_STEPS, backend=resolved_backend)
    t1 = time.monotonic()
    t_diff_params = t1 - t0
    print(f"  Time: {t_diff_params:.3f}s")
    reused = t_diff_params < times[0] * 0.5
    print(f"  Reused JIT cache:      {'YES' if reused else 'NO (recompiled)'}")

    # --- Test 3: no random recoil (different code path) ---
    print(f"\n--- Test 3: no random recoil (2 calls) ---")
    kw_no_recoil = dict(kw, random_recoil=False)
    del kw_no_recoil['max_scatter_probability']
    times_nr = []
    for i in range(2):
        y0, keys = make_batch(obe, N_ATOMS, seed=200 + i)
        t0 = time.monotonic()
        obe.evolve_motion(T_SPAN, y0_batch=y0, keys_batch=keys, **kw_no_recoil)
        t1 = time.monotonic()
        times_nr.append(t1 - t0)
        print(f"  Call {i+1}: {times_nr[-1]:.3f}s")
    speedup_nr = times_nr[0] / times_nr[1] if times_nr[1] > 0 else float('inf')
    print(f"  Speedup: {speedup_nr:.1f}x")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Same-params cache hit:     {'YES' if speedup_2 > 2.0 else 'NO'} ({speedup_2:.1f}x)")
    print(f"  Cross-params cache hit:    {'YES' if reused else 'NO'}")
    print(f"  No-recoil cache hit:       {'YES' if speedup_nr > 2.0 else 'NO'} ({speedup_nr:.1f}x)")


if __name__ == '__main__':
    benchmark()