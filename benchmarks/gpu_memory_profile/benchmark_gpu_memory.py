"""
Benchmark: GPU memory usage scaling with atom count.

Sweeps evolve_motion over increasing atom counts and records GPU memory
at each step: allocated, peak, free, and total.  Produces a log file
and a plot showing how GPU memory is consumed as the batch size grows.

Uses the same F=0→F=1 3D MOT setup as the CPU vs GPU benchmark.
"""
import gc
import os
import sys
import time
import datetime

# Must be set before importing JAX so it preallocates a larger GPU pool.
# Only applies to this process; reverts when the script exits.
if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.94'

import numpy as np
import jax
import jax.numpy as jnp

import pylcp_jax
from pylcp_jax.integration_tools_gpu import (
    optimal_batch_size, optimal_batch_size_per_gpu, _gpu_device_info,
    _bytes_per_atom,
)

# --- Physics parameters (same as cpu_vs_gpu benchmark) ---
DET = -2.5
S = 1.25
ALPHA = 1e-4
SEED = 42
MAX_STEPS = 10000

# Atom counts to sweep.  Built dynamically at runtime: powers of 2 from 1
# up to the optimal GPU batch size (determined by optimal_batch_size after
# JIT warmup).  This ensures we probe right up to the GPU memory limit.
# DEFAULT_MAX_ATOMS caps the sweep to avoid very long runs; pass --no-cap
# on the command line to sweep all the way to the optimal batch size.
DEFAULT_MAX_ATOMS = 16384


class _Tee:
    """Write to both stdout and the log file."""
    def __init__(self, stream, log):
        self._stream, self._log = stream, log
    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
    def flush(self):
        self._stream.flush()
        self._log.flush()
    def __getattr__(self, attr):
        return getattr(self._stream, attr)


def gpu_memory_info():
    """Return dict with GPU memory stats in bytes, or None if no GPU.

    Thin wrapper around _gpu_device_info for the first GPU.
    """
    infos = _gpu_device_info()
    if not infos:
        return None
    info = infos[0]
    if info['bytes_limit'] == 0:
        return None
    return {
        'bytes_limit': info['bytes_limit'],
        'bytes_in_use': info['bytes_in_use'],
        'peak_bytes_in_use': info['peak_bytes_in_use'],
        'bytes_free': info['bytes_free'],
    }


def physical_gpu_memory():
    """Query total physical GPU memory in bytes via nvidia-smi.

    Returns total bytes, or None if nvidia-smi is unavailable.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits', '--id=0'],
            text=True,
        )
        return int(out.strip()) * 2**20  # nvidia-smi reports MiB
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def nvidia_smi_used_memory():
    """Query current GPU memory usage in bytes via nvidia-smi.

    Returns used bytes, or None if nvidia-smi is unavailable.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used',
             '--format=csv,noheader,nounits', '--id=0'],
            text=True,
        )
        return int(out.strip()) * 2**20
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def preallocation_recommendation(phys_bytes, jax_pool_bytes, margin_bytes=300 * 2**20):
    """Compute safe max JAX preallocation fraction.

    Parameters
    ----------
    phys_bytes : int
        Total physical GPU memory.
    jax_pool_bytes : int
        Current JAX memory pool limit.
    margin_bytes : int
        Safety margin to leave free (default 300 MiB).

    Returns
    -------
    dict with current_fraction, safe_max_fraction, non_jax_bytes,
    headroom_bytes, and can_increase (bool).
    """
    nvidia_used = nvidia_smi_used_memory()
    if nvidia_used is None:
        return None

    # Memory used by non-JAX processes (display server, other CUDA apps).
    non_jax = max(nvidia_used - jax_pool_bytes, 0)
    safe_max = (phys_bytes - non_jax - margin_bytes) / phys_bytes
    safe_max = max(0.0, min(safe_max, 0.99))  # clamp to [0, 0.99]
    current_fraction = jax_pool_bytes / phys_bytes
    headroom = (safe_max - current_fraction) * phys_bytes

    return {
        'current_fraction': current_fraction,
        'safe_max_fraction': safe_max,
        'non_jax_bytes': non_jax,
        'margin_bytes': margin_bytes,
        'headroom_bytes': headroom,
        'can_increase': safe_max > current_fraction + 0.02,
    }


def setup():
    """Build the OBE object (F=0→F=1 3D MOT)."""
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
    obe = pylcp_jax.obe(laserBeams, magField, hamiltonian,
                    transform_into_re_im=True)
    return obe


def warmup_evolve(obe):
    """JIT warmup for evolve_motion on GPU."""
    obe.set_initial_rho_equally()
    rho0 = jnp.array(obe.rho0)
    y0 = jnp.concatenate([rho0, jnp.zeros(3), jnp.zeros(3)])
    y0_batch = jnp.stack([y0, y0])
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    obe.evolve_motion([0, 100], y0_batch=y0_batch, keys_batch=keys,
                      freeze_axis=[True, True, False], max_steps=200,
                      backend='gpu')


def make_y0_batch(obe, n_atoms):
    """Create a batch of initial conditions for n_atoms."""
    np.random.seed(SEED)
    rho0 = jnp.array(obe.rho0)
    r_init = np.random.uniform(-2 / ALPHA, 2 / ALPHA, size=(n_atoms, 3))
    r_init[:, :2] = 0.
    y0_list = [
        jnp.concatenate([rho0, jnp.zeros(3), jnp.array(r_init[i])])
        for i in range(n_atoms)
    ]
    return jnp.stack(y0_list)


def run_sweep(obe, atom_counts):
    """Sweep over atom counts, recording memory and timing at each step.

    Returns list of dicts with keys:
        n_atoms, wall_time, time_per_atom,
        mem_before_{in_use,free}, mem_after_{in_use,peak,free}
    All memory values are in bytes.
    """
    t_span = [0, 2 * np.pi * 500]
    kw = dict(freeze_axis=[True, True, False], max_steps=MAX_STEPS, backend='gpu')
    results = []
    y0_batch = keys = None

    for n_atoms in atom_counts:
        # Free all GPU buffers from the prior iteration: solution arrays
        # stored on obe, plus the batch inputs.  Block until async work
        # finishes, then GC so the allocator can reclaim everything.
        del y0_batch, keys
        if hasattr(obe, 'sols'):
            del obe.sols
        if hasattr(obe, 'sol'):
            del obe.sol
        jax.effects_barrier()
        gc.collect()
        jax.clear_caches()

        # Snapshot memory before run.
        mem_before = gpu_memory_info()
        if mem_before is None:
            print("  ERROR: no GPU detected, aborting sweep.")
            break

        print(f"\n  --- {n_atoms} atoms ---")
        print(f"    Before: in_use={mem_before['bytes_in_use']/2**20:.1f} MiB, "
              f"free={mem_before['bytes_free']/2**20:.1f} MiB")

        y0_batch = make_y0_batch(obe, n_atoms)
        keys = jax.random.split(jax.random.PRNGKey(SEED), n_atoms)

        try:
            t0 = time.perf_counter()
            obe.evolve_motion(t_span, y0_batch=y0_batch, keys_batch=keys, **kw)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"    FAILED: {e}")
            print(f"    Stopping sweep (GPU memory likely exhausted).")
            break

        mem_after = gpu_memory_info()
        time_per_atom = elapsed / n_atoms

        row = {
            'n_atoms': n_atoms,
            'wall_time': elapsed,
            'time_per_atom': time_per_atom,
            'mem_before_in_use': mem_before['bytes_in_use'],
            'mem_before_free': mem_before['bytes_free'],
            'mem_after_in_use': mem_after['bytes_in_use'],
            'mem_after_peak': mem_after['peak_bytes_in_use'],
            'mem_after_free': mem_after['bytes_free'],
            'mem_limit': mem_before['bytes_limit'],
        }
        results.append(row)

        print(f"    After:  in_use={mem_after['bytes_in_use']/2**20:.1f} MiB, "
              f"peak={mem_after['peak_bytes_in_use']/2**20:.1f} MiB, "
              f"free={mem_after['bytes_free']/2**20:.1f} MiB")
        print(f"    Time:   {elapsed:.2f}s total, {time_per_atom:.4f}s/atom")

    return results


def print_summary_table(results):
    """Print a formatted summary table."""
    print(f"\n{'N atoms':>8}  {'Wall (s)':>9}  {'s/atom':>8}  "
          f"{'In-use (MiB)':>13}  {'Peak (MiB)':>11}  {'Free (MiB)':>11}  "
          f"{'Usage %':>8}")
    print(f"{'─'*8}  {'─'*9}  {'─'*8}  {'─'*13}  {'─'*11}  {'─'*11}  {'─'*8}")
    for r in results:
        usage_pct = r['mem_after_peak'] / r['mem_limit'] * 100
        print(f"{r['n_atoms']:>8}  {r['wall_time']:>9.2f}  {r['time_per_atom']:>8.4f}  "
              f"{r['mem_after_in_use']/2**20:>13.1f}  "
              f"{r['mem_after_peak']/2**20:>11.1f}  "
              f"{r['mem_after_free']/2**20:>11.1f}  "
              f"{usage_pct:>7.1f}%")


def make_plots(results, out_dir, ts, phys_mem_bytes=None):
    """Generate memory profiling plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not results:
        print("  No data to plot.")
        return

    ns = [r['n_atoms'] for r in results]
    in_use_mib = [r['mem_after_in_use'] / 2**20 for r in results]
    peak_mib = [r['mem_after_peak'] / 2**20 for r in results]
    free_mib = [r['mem_after_free'] / 2**20 for r in results]
    limit_mib = results[0]['mem_limit'] / 2**20
    time_per_atom = [r['time_per_atom'] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    # --- Plot 1: Memory breakdown ---
    ax1 = axes[0]
    ax1.plot(ns, peak_mib, 's-', color='tab:red', label='Peak usage', markersize=6)
    ax1.plot(ns, in_use_mib, 'o-', color='tab:blue', label='Post-run in-use', markersize=5)
    ax1.plot(ns, free_mib, '^-', color='tab:green', label='Post-run free', markersize=5)
    if phys_mem_bytes is not None:
        phys_mib = phys_mem_bytes / 2**20
        ax1.axhline(phys_mib, color='gray', ls=':', lw=1.5,
                     label=f'Physical GPU ({phys_mib:.0f} MiB)')
    ax1.axhline(limit_mib, color='black', ls='--', lw=1.5,
                label=f'JAX pool limit ({limit_mib:.0f} MiB)')
    ax1.set_ylabel('Memory (MiB)')
    ax1.set_title('GPU Memory Usage vs Atom Count')
    ax1.set_xscale('log', base=2)
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', ls='--', alpha=0.4)

    # --- Plot 2: Time per atom ---
    ax2 = axes[1]
    ax2.plot(ns, time_per_atom, 'D-', color='tab:orange', markersize=5)
    ax2.set_xlabel('Number of atoms')
    ax2.set_ylabel('Time per atom (s)')
    ax2.set_title('GPU Compute Efficiency vs Atom Count')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'benchmark_gpu_memory_{ts}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to {out_path}")
    plt.close()


if __name__ == '__main__':
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
    _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _log_path = os.path.join(OUT_DIR, f'benchmark_output_{_ts}.txt')
    _log_file = open(_log_path, 'w')
    sys.stdout = _Tee(sys.stdout, _log_file)

    print(f"Run started: {datetime.datetime.now().isoformat()}")

    # --- Hardware info ---
    print(f"\nHardware:")
    print(f"  CPU cores (logical): {os.cpu_count()}")
    gpu_devs = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpu_devs:
        print(f"  GPU: none detected — this benchmark requires a GPU!")
        sys.exit(1)

    # Detailed per-GPU info.
    gpu_infos = _gpu_device_info(gpu_devs)
    for i, info in enumerate(gpu_infos):
        d = info['device']
        print(f"  GPU {i}: {d}")
        print(f"    Kind:       {info['device_kind']}")
        print(f"    Pool limit: {info['bytes_limit']/2**30:.2f} GiB")
        print(f"    In-use:     {info['bytes_in_use']/2**20:.1f} MiB")
        print(f"    Free:       {info['bytes_free']/2**20:.1f} MiB")

    if len(gpu_infos) > 1:
        pools = [info['bytes_limit'] for info in gpu_infos]
        print(f"\n  Multi-GPU summary: {len(gpu_infos)} devices")
        print(f"    Min pool: {min(pools)/2**30:.2f} GiB, "
              f"Max pool: {max(pools)/2**30:.2f} GiB")
        if min(pools) < max(pools):
            wasted_pct = (1 - min(pools) / max(pools)) * 100
            print(f"    Heterogeneous: largest GPU loses {wasted_pct:.0f}% capacity "
                  f"due to even sharding")

    # --- JAX preallocation info ---
    phys_mem = physical_gpu_memory()
    jax_pool = gpu_infos[0]['bytes_limit']
    mem_fraction_env = os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')
    prealloc_mode = os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')

    print(f"\n  JAX memory preallocation:")
    if phys_mem is not None:
        print(f"    Physical GPU memory:           {phys_mem/2**30:.2f} GiB")
        actual_fraction = jax_pool / phys_mem
        print(f"    JAX pool limit (GPU 0):        {jax_pool/2**30:.2f} GiB "
              f"({actual_fraction*100:.1f}% of physical)")
    else:
        print(f"    Physical GPU memory:           (nvidia-smi unavailable)")
        print(f"    JAX pool limit (GPU 0):        {jax_pool/2**30:.2f} GiB")
    print(f"    XLA_PYTHON_CLIENT_MEM_FRACTION: {mem_fraction_env or 'not set (default 0.75)'}")
    print(f"    XLA_PYTHON_CLIENT_PREALLOCATE:  {prealloc_mode or 'not set (default true)'}")
    if phys_mem is not None:
        wasted = phys_mem - jax_pool
        print(f"    Unavailable to JAX:            {wasted/2**30:.2f} GiB "
              f"({wasted/phys_mem*100:.1f}% of physical)")

    # --- Preallocation headroom check ---
    if phys_mem is not None:
        rec = preallocation_recommendation(phys_mem, jax_pool)
        if rec is not None:
            print(f"\n  Preallocation headroom analysis:")
            print(f"    Non-JAX GPU usage (other procs): {rec['non_jax_bytes']/2**20:.0f} MiB")
            print(f"    Safety margin:                   {rec['margin_bytes']/2**20:.0f} MiB")
            print(f"    Current fraction:                {rec['current_fraction']*100:.1f}%")
            print(f"    Safe max fraction:               {rec['safe_max_fraction']*100:.1f}%")
            if rec['can_increase']:
                print(f"    -> Pool could grow by ~{rec['headroom_bytes']/2**20:.0f} MiB. "
                      f"Set XLA_PYTHON_CLIENT_MEM_FRACTION={rec['safe_max_fraction']:.2f}")
            else:
                print(f"    -> Current fraction is near the safe limit, no increase recommended.")

    print(f"\nParameters: det={DET}, s={S}, alpha={ALPHA}")
    print(f"Max steps: {MAX_STEPS}")

    # --- Setup and baseline memory ---
    obe = setup()
    obe.set_initial_rho_equally()
    state_dim = len(obe.rho0) + 6  # rho + v(3) + r(3)

    bpa = _bytes_per_atom(state_dim, MAX_STEPS, inner_max_steps=64)
    print(f"\n  state_dim={state_dim}")
    print(f"  Estimated bytes/atom (solver buffers): {bpa:,} "
          f"({bpa/2**20:.2f} MiB)")

    mem_baseline = gpu_memory_info()
    print(f"\n  Baseline GPU memory (before warmup):")
    print(f"    In-use: {mem_baseline['bytes_in_use']/2**20:.1f} MiB")
    print(f"    Free:   {mem_baseline['bytes_free']/2**20:.1f} MiB")
    print(f"    Total:  {mem_baseline['bytes_limit']/2**20:.1f} MiB")

    # --- Warmup ---
    # Must happen BEFORE optimal_batch_size so the estimate accounts for
    # GPU memory consumed by JIT caches; otherwise the sweep overshoots
    # and OOMs at the limit.
    print("\n" + "=" * 60)
    print("  JIT Warmup")
    print("=" * 60)
    warmup_evolve(obe)
    mem_post_warmup = gpu_memory_info()
    print(f"\n  Post-warmup GPU memory:")
    print(f"    In-use: {mem_post_warmup['bytes_in_use']/2**20:.1f} MiB")
    print(f"    Peak:   {mem_post_warmup['peak_bytes_in_use']/2**20:.1f} MiB")
    print(f"    Free:   {mem_post_warmup['bytes_free']/2**20:.1f} MiB")

    # --- Compute optimal batch size (post-warmup) ---
    optimal_n = optimal_batch_size(state_dim, MAX_STEPS, inner_max_steps=64, safety=0.6)
    print(f"\n  optimal_batch_size (safety=0.6): {optimal_n}")

    # Per-GPU capacity breakdown (useful for heterogeneous multi-GPU).
    per_gpu = optimal_batch_size_per_gpu(state_dim, MAX_STEPS, inner_max_steps=64, safety=0.6)
    if per_gpu and len(per_gpu) > 1:
        print(f"\n  Per-GPU capacity (heterogeneous breakdown):")
        for dev, cap in per_gpu:
            print(f"    {dev}: {cap} atoms")
        total_prop = sum(c for _, c in per_gpu)
        print(f"    Proportional total: {total_prop} atoms "
              f"(vs even-shard total: {optimal_n})")

    # Build atom counts: powers of 2 from 1 up to the target.
    # By default capped at DEFAULT_MAX_ATOMS; pass --no-cap to go to optimal_n.
    no_cap = '--no-cap' in sys.argv
    max_atoms = optimal_n if no_cap else min(optimal_n, DEFAULT_MAX_ATOMS)
    print(f"  Sweep cap: {'none (--no-cap), using optimal={optimal_n}' if no_cap else f'min({DEFAULT_MAX_ATOMS}, optimal={optimal_n}) = {max_atoms}'}")

    ATOM_COUNTS = []
    n = 1
    while n < max_atoms:
        ATOM_COUNTS.append(n)
        n *= 2
    if not ATOM_COUNTS or ATOM_COUNTS[-1] != max_atoms:
        ATOM_COUNTS.append(max_atoms)
    print(f"  Atom counts to sweep: {ATOM_COUNTS}")

    # --- Sweep ---
    print("\n" + "=" * 60)
    print("  GPU Memory Sweep")
    print("=" * 60)
    results = run_sweep(obe, ATOM_COUNTS)

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print_summary_table(results)

    # --- Plots ---
    print("\n" + "=" * 60)
    print("  Generating plots")
    print("=" * 60)
    make_plots(results, OUT_DIR, _ts, phys_mem_bytes=phys_mem)

    print(f"\nRun finished: {datetime.datetime.now().isoformat()}")
    print(f"Output saved to {_log_path}")
    sys.stdout = sys.stdout._stream
    _log_file.close()
