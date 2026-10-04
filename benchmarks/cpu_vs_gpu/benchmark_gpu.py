"""GPU-only benchmark: evolve_motion per-atom times, saved to a pickle.

Sweeps over multiple transitions (state dimensions) and extends atom counts
up to `optimal_batch_size` for each transition.

Output: gpu_results.pkl (next to this file, or --out PATH).
"""
import os
import sys
import time
import pickle
import datetime
import argparse

import numpy as np
import jax
import jax.numpy as jnp

from common import (
    N_POINTS, SEED, SWEEP_GPU_ATOMS, SWEEP_T_FACTORS, TRANSITIONS,
    setup_obe, make_y0_list,
)
from pylcp_jax.integration_tools_gpu import (
    optimal_batch_size, _make_run_group, _probe_bytes_per_atom,
)


def capture_cost(obe, y0_batch, keys_batch, t_span, n_points):
    """XLA cost analysis for one save-group kernel call.

    Lowers the inner JIT kernel (`_run_group`) with the same shapes used in
    the timing run and reports flops + bytes accessed per kernel invocation.
    The kernel is called n_points times per evolve_motion, so program-level
    totals = n_points * these. Arithmetic intensity (flops/bytes) is
    invariant and is the roofline classifier we care about.

    XLA's static cost analysis does not multiply by while-loop trip counts,
    so the reported numbers represent the cost of a single solver group
    body — exactly what we want for arithmetic-intensity comparisons.
    """
    free_axes = jnp.bitwise_not(jnp.asarray([True, True, False], dtype=bool))
    args = {
        "free_axes": free_axes,
        "max_scatter_probability": jnp.asarray(0.1, dtype=jnp.float64),
    }
    run_group = _make_run_group(obe._motion_dydt, obe._no_recoil, "Dopri5")

    N = y0_batch.shape[0]
    t0, tf = float(t_span[0]), float(t_span[1])
    t_save_grid = np.linspace(t0, tf, n_points + 1)
    dt0 = (tf - t0) * 1e-3

    carry = {
        "t": jnp.full(N, t0, dtype=jnp.float64),
        "y": y0_batch,
        "dt": jnp.full(N, dt0, dtype=jnp.float64),
        "key": keys_batch,
        "step_idx": jnp.ones(N, dtype=jnp.int32),
        "nfev": jnp.zeros(N, dtype=jnp.int32),
        "last_t_random": jnp.zeros(N, dtype=jnp.float64),
        "last_n_random": jnp.zeros(N, dtype=jnp.int32),
        "t_save_next": jnp.full(N, t_save_grid[1], dtype=jnp.float64),
    }

    lowered = run_group.lower(carry, float("inf"), 1e-5, 1e-6, args).compile()
    raw = lowered.cost_analysis()
    # JAX returns either a dict or a list-of-dicts depending on version.
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    flops = float(raw.get("flops", 0.0)) if raw else 0.0
    # XLA reports this under either key depending on version.
    bytes_acc = float(raw.get("bytes accessed", raw.get("bytes_accessed", 0.0))) \
        if raw else 0.0
    ai = (flops / bytes_acc) if bytes_acc > 0 else None
    return {"flops": flops, "bytes": bytes_acc, "ai": ai}


def warmup(obe):
    rho0 = jnp.array(obe.rho0)
    y0 = jnp.concatenate([rho0, jnp.zeros(3), jnp.zeros(3)])
    y0_batch = jnp.stack([y0, y0])
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    obe.evolve_motion([0, 100], n_points=8, y0_batch=y0_batch, keys_batch=keys,
                      freeze_axis=[True, True, False], backend='gpu')


def run_gpu_sweep(obe, atom_counts, t_factor):
    kw = dict(freeze_axis=[True, True, False], backend='gpu')
    t_span = [0, 2 * np.pi * t_factor]
    out = {}
    for n in atom_counts:
        y0_list = make_y0_list(obe, n)
        y0_batch = jnp.stack(y0_list)
        keys = jax.random.split(jax.random.PRNGKey(SEED), n)
        try:
            # Prime JIT for this batch shape.
            obe.evolve_motion(t_span, n_points=N_POINTS,
                              y0_batch=y0_batch[:min(n, 2)],
                              keys_batch=keys[:min(n, 2)], **kw)
            try:
                cost = capture_cost(obe, y0_batch, keys, t_span, N_POINTS)
            except Exception as cexc:
                cost = {'flops': None, 'bytes': None, 'ai': None,
                        'error': f"{type(cexc).__name__}: {cexc}"}
            t0 = time.perf_counter()
            obe.evolve_motion(t_span, n_points=N_POINTS,
                              y0_batch=y0_batch, keys_batch=keys, **kw)
            t_per_atom = (time.perf_counter() - t0) / n
            z_gpu = np.array([sol.r[2, -1] for sol in obe.sols])
            out[n] = {'gpu': t_per_atom, 'z_gpu': z_gpu, 'cost': cost}
            ai_str = (f"{cost['ai']:.1f} F/B" if cost.get('ai') is not None
                      else "AI=?")
            print(f"    N={n:>6}: {t_per_atom:.4f} s/atom  ({ai_str})",
                  flush=True)
        except Exception as exc:
            print(f"    N={n:>6}: FAILED ({type(exc).__name__}: {exc})",
                  flush=True)
            out[n] = {'gpu': None, 'z_gpu': None, 'error': str(exc)}
    return out


def run_one_transition(name, Fg, Fe, gFg, gFe):
    """Sweep a single transition. Returns the per-transition result dict."""
    print(f"\n########## Transition {name} "
          f"(Fg={Fg}, Fe={Fe}) ##########", flush=True)
    obe = setup_obe(Fg=Fg, Fe=Fe, gFg=gFg, gFe=gFe)
    state_dim = len(obe.rho0) + 6

    print("Warming up JIT...", flush=True)
    warmup(obe)

    n_batched = optimal_batch_size(state_dim, safety=0.6)
    bytes_per_atom = _probe_bytes_per_atom(state_dim)
    print(f"  state_dim={state_dim}, optimal GPU batch size: {n_batched}, "
          f"bytes/atom={bytes_per_atom}")

    atom_counts = sorted(set(SWEEP_GPU_ATOMS))
    if n_batched:
        atom_counts = [n for n in atom_counts if n <= n_batched]
        if atom_counts and n_batched > 2 * atom_counts[-1]:
            mid = int(np.sqrt(atom_counts[-1] * n_batched))
            atom_counts.append(mid)
        atom_counts.append(n_batched)
        atom_counts = sorted(set(atom_counts))

    runs = {}
    for t_factor in SWEEP_T_FACTORS:
        print(f"\n=== {name}: t = 2pi x {t_factor} ===", flush=True)
        runs[t_factor] = run_gpu_sweep(obe, atom_counts, t_factor)

    return {
        'transition': {'Fg': Fg, 'Fe': Fe, 'gFg': gFg, 'gFe': gFe},
        'state_dim': state_dim,
        'optimal_batch_size': n_batched,
        'bytes_per_atom': bytes_per_atom,
        'atom_counts': atom_counts,
        'runs': runs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None,
                    help="Output pickle path. Default depends on --transition: "
                         "with one named transition, gpu_results_<name>.pkl; "
                         "without, gpu_results.pkl.")
    ap.add_argument('--transition', default=None,
                    help="Name of a single transition to run "
                         "(e.g. F0_F1). Omit to run all in one process.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(here, 'run')
    os.makedirs(run_dir, exist_ok=True)

    if args.transition:
        matches = [t for t in TRANSITIONS if t[0] == args.transition]
        if not matches:
            valid = ', '.join(t[0] for t in TRANSITIONS)
            print(f"ERROR: unknown transition '{args.transition}'. "
                  f"Valid: {valid}", file=sys.stderr)
            sys.exit(2)
        wanted = matches
        default_name = f'gpu_results_{args.transition}.pkl'
    else:
        wanted = list(TRANSITIONS)
        default_name = 'gpu_results.pkl'
    out_path = args.out or os.path.join(run_dir, default_name)

    print(f"Run started: {datetime.datetime.now().isoformat()}")
    gpu_devs = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpu_devs:
        print("ERROR: no GPU devices visible to JAX.", file=sys.stderr)
        sys.exit(1)
    for d in gpu_devs:
        mem = d.memory_stats()
        print(f"  GPU: {d}  memory={mem['bytes_limit']/2**30:.1f} GiB")

    results_by_transition = {
        name: run_one_transition(name, Fg, Fe, gFg, gFe)
        for name, Fg, Fe, gFg, gFe in wanted
    }

    payload = {
        'meta': {
            'timestamp': datetime.datetime.now().isoformat(),
            't_factors': list(SWEEP_T_FACTORS),
            'n_points': N_POINTS,
            'seed': SEED,
            'gpu_devices': [str(d) for d in gpu_devs],
            'gpu_memory_bytes': [d.memory_stats()['bytes_limit']
                                 for d in gpu_devs],
        },
        'transitions': results_by_transition,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
