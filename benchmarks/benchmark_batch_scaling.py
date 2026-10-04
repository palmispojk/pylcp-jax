"""
Benchmark: batch size scaling and diminishing returns.

Runs short simulations with increasing atom counts to find:
  1. Whether optimal_batch_size gives a safe (no OOM) estimate
  2. The knee point where per-atom time stops improving

Usage:
    python benchmarks/benchmark_batch_scaling.py
"""
import os
import gc
import time

if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.94'

import numpy as np
import jax
import jax.numpy as jnp

import pylcp_jax
from pylcp_jax.integration_tools_gpu import (
    optimal_batch_size, _bytes_per_atom, _probe_bytes_per_atom,
    _gpu_device_info,
)

# --- Config ---
MAX_STEPS = 2000
INNER_MAX_STEPS = 64
SAVE_EVERY = 20        # 100 output points — enough to exercise the host loop
TMAX = 1e3
SEED = 42

# Atom counts to test — geometric progression to cover a wide range
BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# --- Build a lightweight OBE (J=0 -> J=1, state_dim=22) ---
def build_obe():
    Hg, Bgq = pylcp_jax.hamiltonians.singleF(F=0, gF=0, muB=1)
    He, Beq = pylcp_jax.hamiltonians.singleF(F=1, gF=1, muB=1)
    dijq = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(0, 1)
    hamiltonian = pylcp_jax.hamiltonian(
        Hg, 2.0 * np.eye(3) + He, Bgq, Beq, dijq, mass=100,
    )
    laserBeams = pylcp_jax.conventional3DMOTBeams(
        s=1.0, delta=0., beam_type=pylcp_jax.infinitePlaneWaveBeam,
    )
    magField = pylcp_jax.quadrupoleMagneticField(1e-4)
    return pylcp_jax.obe(laserBeams, magField, hamiltonian,
                     transform_into_re_im=True), hamiltonian


def make_batch(obe, n_atoms):
    obe.set_initial_rho_equally()
    rho0 = jnp.array(obe.rho0)
    rng = np.random.default_rng(SEED)
    r0 = rng.standard_normal((n_atoms, 3)) * 10
    v0 = rng.standard_normal((n_atoms, 3)) * 0.1
    y0_list = [
        jnp.concatenate([rho0, jnp.array(v0[i]), jnp.array(r0[i])])
        for i in range(n_atoms)
    ]
    y0_batch = jnp.stack(y0_list)
    keys = jax.random.split(jax.random.PRNGKey(SEED), n_atoms)
    return y0_batch, keys


def main():
    info = _gpu_device_info()
    if not info or info[0]['bytes_limit'] == 0:
        print("No GPU detected.")
        return
    gpu_info = info[0]
    print(f"GPU: {gpu_info['device']}")
    print(f"  VRAM pool:  {gpu_info['bytes_limit']/2**30:.2f} GiB")
    print(f"  Free:       {gpu_info['bytes_free']/2**20:.0f} MiB")
    print()

    obe, hamiltonian = build_obe()
    state_dim = hamiltonian.n ** 2 + 6
    print(f"state_dim = {state_dim}")

    # Compare analytical vs probe estimate
    analytical_bpa = _bytes_per_atom(state_dim, MAX_STEPS, INNER_MAX_STEPS)
    probe_bpa = _probe_bytes_per_atom(state_dim, INNER_MAX_STEPS)
    opt_n = optimal_batch_size(state_dim, MAX_STEPS, INNER_MAX_STEPS,
                               save_every=SAVE_EVERY)
    print(f"Analytical estimate: {analytical_bpa:,} B/atom ({analytical_bpa/2**20:.4f} MiB)")
    print(f"Probe estimate:      {probe_bpa:,} B/atom ({probe_bpa/2**20:.4f} MiB)")
    print(f"Ratio (probe/analytical): {probe_bpa/analytical_bpa:.1f}x")
    print(f"optimal_batch_size:  {opt_n}")
    print()

    # JIT warmup
    print("JIT warmup...")
    y0_w, keys_w = make_batch(obe, 2)
    obe.evolve_motion(
        [0, 100], y0_batch=y0_w, keys_batch=keys_w,
        max_steps=200, inner_max_steps=INNER_MAX_STEPS, backend='gpu',
    )
    del y0_w, keys_w
    if hasattr(obe, 'sols'):
        del obe.sols
    jax.effects_barrier()
    gc.collect()

    # Run scaling benchmark
    print(f"{'N':>6}  {'Total (s)':>10}  {'s/atom':>10}  {'Speedup':>8}  {'GPU peak (MiB)':>14}  {'Status':>8}")
    print(f"{'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*14}  {'─'*8}")

    results = []
    prev_s_per_atom = None

    for n_atoms in BATCH_SIZES:
        y0_batch, keys = make_batch(obe, n_atoms)

        # Clean up before measuring
        if hasattr(obe, 'sols'):
            del obe.sols
        jax.effects_barrier()
        gc.collect()

        try:
            t0 = time.perf_counter()
            obe.evolve_motion(
                [0, TMAX],
                y0_batch=y0_batch,
                keys_batch=keys,
                random_recoil=True,
                max_scatter_probability=0.5,
                max_step=TMAX / MAX_STEPS,
                max_steps=MAX_STEPS,
                inner_max_steps=INNER_MAX_STEPS,
                save_every=SAVE_EVERY,
                batch_size=n_atoms,  # force exact batch, no auto-chunking
                backend='gpu',
            )
            elapsed = time.perf_counter() - t0
            status = "OK"
        except Exception as e:
            elapsed = None
            status = "OOM" if "out of memory" in str(e).lower() else f"ERR"
            print(f"  {n_atoms:>6}  {'—':>10}  {'—':>10}  {'—':>8}  {'—':>14}  {status:>8}")
            if "out of memory" in str(e).lower():
                print(f"         OOM at N={n_atoms} — this is the GPU limit.")
            else:
                print(f"         Error: {e}")
            break

        gpu_peak = _gpu_device_info()[0]['peak_bytes_in_use']
        s_per_atom = elapsed / n_atoms

        if prev_s_per_atom is not None and prev_s_per_atom > 0:
            speedup = prev_s_per_atom / s_per_atom
        else:
            speedup = None

        results.append({
            'n_atoms': n_atoms,
            'total_s': elapsed,
            's_per_atom': s_per_atom,
            'speedup': speedup,
            'gpu_peak_mib': gpu_peak / 2**20,
        })

        speedup_str = f"{speedup:.2f}x" if speedup else "—"
        print(f"  {n_atoms:>6}  {elapsed:>10.2f}  {s_per_atom:>10.4f}  {speedup_str:>8}  "
              f"{gpu_peak/2**20:>14.0f}  {status:>8}")

        prev_s_per_atom = s_per_atom

        # Clean up
        del y0_batch, keys
        if hasattr(obe, 'sols'):
            del obe.sols
        jax.effects_barrier()
        gc.collect()

    if not results:
        return

    # Find the knee: where doubling N gives <1.3x speedup per atom
    print()
    print("=" * 70)
    s_per_atom_values = [r['s_per_atom'] for r in results]
    best_idx = np.argmin(s_per_atom_values)
    best = results[best_idx]
    print(f"Best throughput: {best['s_per_atom']:.4f} s/atom at N={best['n_atoms']}")

    knee = None
    for i in range(1, len(results)):
        speedup = results[i].get('speedup')
        if speedup is not None and speedup < 1.3:
            knee = results[i - 1]
            break

    if knee:
        print(f"Diminishing returns after N={knee['n_atoms']} "
              f"({knee['s_per_atom']:.4f} s/atom)")
        print(f"  -> Recommended MAX_ATOMS = {knee['n_atoms']}")
    else:
        print(f"No diminishing returns detected in tested range — "
              f"try larger batch sizes.")

    print()
    print(f"optimal_batch_size returned: {opt_n}")
    if opt_n and knee:
        if opt_n > knee['n_atoms'] * 4:
            print(f"  WARNING: optimal_batch_size ({opt_n}) is much larger than "
                  f"the diminishing-returns point ({knee['n_atoms']})")
        else:
            print(f"  OK: optimal_batch_size is in a reasonable range")


if __name__ == '__main__':
    main()
