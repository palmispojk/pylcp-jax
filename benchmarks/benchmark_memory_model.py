"""
Benchmark: memory model accuracy for _bytes_per_atom.

Runs small batches of evolve_motion with different transition sizes and
compares the predicted memory usage from _bytes_per_atom against the
actual GPU memory consumed. This validates that optimal_batch_size
returns a safe atom count that won't OOM.

Transitions tested:
  - F=0 -> F'=1  (state_dim=15)
  - F=1 -> F'=2  (state_dim=55)
  - F=2 -> F'=3  (state_dim=150)
"""
import gc
import os
import time

if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.94'

import numpy as np
import jax
import jax.numpy as jnp

import pylcp_jax
from pylcp_jax.integration_tools_gpu import (
    _bytes_per_atom, _gpu_device_info, optimal_batch_size,
)

MAX_STEPS = 5000
INNER_MAX_STEPS = 64
TMAX = 1e3
SEED = 42
ATOM_COUNTS = [8, 16, 32, 64, 128]

TRANSITIONS = [
    {'name': 'F=0→F\'=1', 'Fg': 0, 'Fe': 1, 'gFg': 0, 'gFe': 1},
    {'name': 'F=1→F\'=2', 'Fg': 1, 'Fe': 2, 'gFg': 0, 'gFe': 0.5},
    {'name': 'F=2→F\'=3', 'Fg': 2, 'Fe': 3, 'gFg': 1.5, 'gFe': 4/3},
]


def gpu_memory_info():
    infos = _gpu_device_info()
    if not infos:
        return None
    info = infos[0]
    if info['bytes_limit'] == 0:
        return None
    return info


def build_obe(Fg, Fe, gFg, gFe):
    Hg, Bgq = pylcp_jax.hamiltonians.singleF(F=Fg, gF=gFg, muB=1)
    He, Beq = pylcp_jax.hamiltonians.singleF(F=Fe, gF=gFe, muB=1)
    dijq = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(Fg, Fe)
    det = -2.5
    hamiltonian = pylcp_jax.hamiltonian(
        Hg, -det * np.eye(2 * Fe + 1) + He, Bgq, Beq, dijq, mass=100
    )
    laserBeams = pylcp_jax.conventional3DMOTBeams(
        s=1.25, delta=0., beam_type=pylcp_jax.infinitePlaneWaveBeam
    )
    magField = pylcp_jax.quadrupoleMagneticField(1e-4)
    return pylcp_jax.obe(laserBeams, magField, hamiltonian,
                     transform_into_re_im=True), hamiltonian


def make_batch(obe, hamiltonian, n_atoms):
    obe.set_initial_rho_equally()
    rho0 = jnp.array(obe.rho0)
    rng = np.random.default_rng(SEED)
    r0 = rng.standard_normal((n_atoms, 3)) * 100
    v0 = rng.standard_normal((n_atoms, 3)) * 0.1
    y0_list = [
        jnp.concatenate([rho0, jnp.array(v0[i]), jnp.array(r0[i])])
        for i in range(n_atoms)
    ]
    y0_batch = jnp.stack(y0_list)
    keys = jax.random.split(jax.random.PRNGKey(SEED), n_atoms)
    return y0_batch, keys


def measure_run(obe, y0_batch, keys):
    """Run evolve_motion and return (actual_bytes_used, wall_time)."""
    # Clean up before measuring baseline
    if hasattr(obe, 'sols'):
        del obe.sols
    jax.effects_barrier()
    gc.collect()

    mem_before = gpu_memory_info()
    if mem_before is None:
        return None, None

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
        backend='gpu',
    )
    elapsed = time.perf_counter() - t0

    mem_after = gpu_memory_info()
    actual_bytes = mem_after['peak_bytes_in_use'] - mem_before['bytes_in_use']
    return actual_bytes, elapsed


def main():
    info = gpu_memory_info()
    if info is None:
        print("No GPU detected.")
        return

    print(f"GPU memory: {info['bytes_limit'] / 2**20:.0f} MiB total, "
          f"{info['bytes_free'] / 2**20:.0f} MiB free")
    print(f"Config: MAX_STEPS={MAX_STEPS}, INNER_MAX_STEPS={INNER_MAX_STEPS}, "
          f"TMAX={TMAX}")
    print()

    all_results = []

    for trans in TRANSITIONS:
        print(f"=== {trans['name']} ===")
        obe, hamiltonian = build_obe(
            trans['Fg'], trans['Fe'], trans['gFg'], trans['gFe']
        )
        state_dim = hamiltonian.n ** 2 + 6
        predicted_bpa = _bytes_per_atom(state_dim, MAX_STEPS, INNER_MAX_STEPS)
        opt_n = optimal_batch_size(state_dim, MAX_STEPS, INNER_MAX_STEPS)
        print(f"  state_dim={state_dim}, predicted={predicted_bpa / 2**20:.3f} MiB/atom, "
              f"optimal_batch={opt_n}")

        # JIT warmup with 2 atoms
        y0_warmup, keys_warmup = make_batch(obe, hamiltonian, 2)
        obe.evolve_motion(
            [0, 100], y0_batch=y0_warmup, keys_batch=keys_warmup,
            max_steps=200, inner_max_steps=INNER_MAX_STEPS, backend='gpu',
        )
        del y0_warmup, keys_warmup
        if hasattr(obe, 'sols'):
            del obe.sols
        jax.effects_barrier()
        gc.collect()

        print(f"\n  {'N':>6}  {'Predicted (MiB)':>15}  {'Actual (MiB)':>13}  "
              f"{'Ratio':>7}  {'Time (s)':>9}")
        print(f"  {'─' * 6}  {'─' * 15}  {'─' * 13}  {'─' * 7}  {'─' * 9}")

        for n_atoms in ATOM_COUNTS:
            y0_batch, keys = make_batch(obe, hamiltonian, n_atoms)

            try:
                actual_bytes, elapsed = measure_run(obe, y0_batch, keys)
            except Exception as e:
                print(f"  {n_atoms:>6}  OOM or error: {e}")
                break

            if actual_bytes is None:
                print(f"  {n_atoms:>6}  GPU memory info unavailable")
                continue

            predicted_bytes = predicted_bpa * n_atoms
            ratio = actual_bytes / predicted_bytes if predicted_bytes > 0 else float('inf')

            print(f"  {n_atoms:>6}  {predicted_bytes / 2**20:>15.2f}  "
                  f"{actual_bytes / 2**20:>13.2f}  {ratio:>7.2f}  "
                  f"{elapsed:>9.2f}")

            all_results.append({
                'transition': trans['name'],
                'state_dim': state_dim,
                'n_atoms': n_atoms,
                'predicted_bytes': predicted_bytes,
                'actual_bytes': actual_bytes,
                'ratio': ratio,
                'wall_time': elapsed,
            })

            # Clean up between runs
            del y0_batch, keys
            if hasattr(obe, 'sols'):
                del obe.sols
            jax.effects_barrier()
            gc.collect()

        print()

    # Summary
    if all_results:
        ratios = [r['ratio'] for r in all_results]
        print("=" * 60)
        print("Summary: actual / predicted ratio")
        print(f"  Min:  {min(ratios):.2f}")
        print(f"  Max:  {max(ratios):.2f}")
        print(f"  Mean: {np.mean(ratios):.2f}")
        print()
        if max(ratios) > 1.0:
            print("WARNING: actual memory exceeded prediction — OOM risk!")
            print("The memory model underestimates real usage.")
        else:
            print("Memory model is conservative — no OOM risk at these sizes.")


if __name__ == '__main__':
    main()