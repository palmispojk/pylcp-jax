"""Shared config, setup, and helpers for the split CPU/GPU benchmarks."""
import os
import numpy as np

DET = -2.5
S = 1.25
ALPHA = 1e-4
SEED = 42
# Number of output time points saved by evolve_motion. Benchmarks only look at
# final state, so a small number is fine — it does not control solver step size.
N_POINTS = 32

# Time spans to sweep: each is [0, 2*pi*T].
SWEEP_T_FACTORS = [100, 500, 2000]

# Atom counts.
# CPU: extends into the GPU range so the two curves can be compared, but
# only the high-core-count runs go that high (see CPU_PARALLEL_MAX_N).
SWEEP_CPU_ATOMS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Per-core-count cap on N: small core counts would take hours at large N
# without adding useful information, so they stay at the original ceiling.
# Only 16/32/64-core runs sweep the full range.
CPU_PARALLEL_MAX_N = {2: 128, 4: 128, 8: 128, 16: 2048, 32: 2048, 64: 2048}

# Serial per-atom time is N-independent, so there's no reason to pay the
# wall-clock cost at large N.
SERIAL_MAX_N = 128
# GPU: large — start where batching pays off and extend toward optimal_batch_size.
# benchmark_gpu.py appends `optimal_batch_size` onto this list automatically.
SWEEP_GPU_ATOMS = [512, 1024, 2048, 4096, 8192, 16384, 32768]

# Transitions to benchmark (name, F_g, F_e, gF_g, gF_e).
# state_dim = (2Fg+1 + 2Fe+1)^2 + 6.
TRANSITIONS = [
    ('F0_F1',     0,    1,    0,    1   ),   # n=4,  state_dim=22 (Sr blue MOT)
    ('F0p5_F1p5', 0.5,  1.5,  2.0,  2/3 ),   # n=6,  state_dim=42 (alkali D2-like)
    ('F1_F2',     1,    2,    0.5,  0.5 ),   # n=8,  state_dim=70
    ('F2_F3',     2,    3,    0.5,  2/3 ),   # n=12, state_dim=150 (Rb87 cooling)
]

# Parallel (multiprocessing) sweep: number of worker processes to spawn.
PARALLEL_CORE_COUNTS = [2, 4, 8, 16, 32, 64]
N_SERIAL = 2

# Amdahl analysis.
AMDAHL_ATOMS_PER_WORKER = [2, 4]
AMDAHL_CORE_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]

# Roofline constants for the GPU on the target queue (NVIDIA A100 SXM4 40GB,
# qist-gpu partition). FP64 because the integrator carry is float64.
# Override via env vars PYLCP_GPU_PEAK_FP64_TFLOPS / PYLCP_GPU_PEAK_BW_GBS
# in analyze.py if running on a different device.
GPU_PEAK_FP64_TFLOPS = 9.7        # A100 FP64 peak
GPU_PEAK_BW_GBS = 1555.0          # A100 HBM2e peak bandwidth


def setup_obe(Fg=0, Fe=1, gFg=0, gFe=1):
    """Build the OBE object for a given (Fg -> Fe) transition."""
    import pylcp_jax
    Hg, Bgq = pylcp_jax.hamiltonians.singleF(F=Fg, gF=gFg, muB=1)
    He, Beq = pylcp_jax.hamiltonians.singleF(F=Fe, gF=gFe, muB=1)
    dijq = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(Fg, Fe)
    ne = int(round(2 * Fe + 1))
    hamiltonian = pylcp_jax.hamiltonian(
        Hg, -DET * np.eye(ne) + He, Bgq, Beq, dijq, mass=100
    )
    laserBeams = pylcp_jax.conventional3DMOTBeams(
        s=S, delta=0., beam_type=pylcp_jax.infinitePlaneWaveBeam
    )
    magField = pylcp_jax.quadrupoleMagneticField(ALPHA)
    obe = pylcp_jax.obe(laserBeams, magField, hamiltonian, transform_into_re_im=True)
    obe.set_initial_rho_equally()
    return obe


def make_y0_list(obe, n_atoms):
    """Deterministic initial state list (seeded)."""
    import jax.numpy as jnp
    np.random.seed(SEED)
    rho0 = jnp.array(obe.rho0)
    r_init = np.random.uniform(-2 / ALPHA, 2 / ALPHA, size=(n_atoms, 3))
    r_init[:, :2] = 0.
    return [
        jnp.concatenate([rho0, jnp.zeros(3), jnp.array(r_init[i])])
        for i in range(n_atoms)
    ]


def amdahl_speedup(p, n):
    return 1.0 / ((1.0 - p) + p / n)


def fit_amdahl_p(t_per_atom_serial, parallel_results):
    """Least-squares fit of the parallelisable fraction p.

    parallel_results: dict {n_cores: t_per_atom}.
    Returns (p, {n_cores: measured_speedup}) or (None, {}) if <2 points.
    """
    if len(parallel_results) < 2:
        return None, {}
    speedups = {}
    xs, ys = [], []
    for n_cores, t_pa in parallel_results.items():
        s_i = t_per_atom_serial / t_pa
        speedups[n_cores] = s_i
        xs.append(1.0 - 1.0 / n_cores)
        ys.append(1.0 - 1.0 / s_i)
    xs, ys = np.array(xs), np.array(ys)
    p = float(np.clip(np.dot(xs, ys) / np.dot(xs, xs), 0.0, 1.0))
    return p, speedups


# ---------------------------------------------------------------------------
# Worker function — module-level so the spawn pickler can find it.
# Guarded by _PYLCP_BENCH_WORKER=1 in the parent so children pin JAX to CPU.
# ---------------------------------------------------------------------------
def _parallel_worker(y0_batch_np):
    import numpy as _np
    import pylcp_jax as _pylcp
    import jax.numpy as _jnp

    Fg = float(os.environ.get('_PYLCP_BENCH_FG', '0'))
    Fe = float(os.environ.get('_PYLCP_BENCH_FE', '1'))
    gFg = float(os.environ.get('_PYLCP_BENCH_GFG', '0'))
    gFe = float(os.environ.get('_PYLCP_BENCH_GFE', '1'))

    Hg, Bgq = _pylcp.hamiltonians.singleF(F=Fg, gF=gFg, muB=1)
    He, Beq = _pylcp.hamiltonians.singleF(F=Fe, gF=gFe, muB=1)
    dijq = _pylcp.hamiltonians.dqij_two_bare_hyperfine(Fg, Fe)
    ne = int(round(2 * Fe + 1))
    hamiltonian = _pylcp.hamiltonian(
        Hg, -DET * _np.eye(ne) + He, Bgq, Beq, dijq, mass=100
    )
    laserBeams = _pylcp.conventional3DMOTBeams(
        s=S, delta=0., beam_type=_pylcp.infinitePlaneWaveBeam
    )
    obe = _pylcp.obe(laserBeams, _pylcp.quadrupoleMagneticField(ALPHA),
                     hamiltonian, transform_into_re_im=True)
    obe.set_initial_rho_equally()

    t_factor = float(os.environ.get('_PYLCP_BENCH_T_FACTOR', '500'))
    out = []
    for y0_np in y0_batch_np:
        y0 = _jnp.array(y0_np)
        obe.evolve_motion(
            [0, 2 * _np.pi * t_factor],
            n_points=N_POINTS,
            y0_batch=y0[_jnp.newaxis, :],
            freeze_axis=[True, True, False],
            backend='cpu',
        )
        out.append(float(obe.sol.r[2, -1]))
    return out


def run_parallel(y0_list, n_cores, t_factor, transition=(0, 1, 0, 1)):
    """Run atoms across n_cores worker processes (stdlib multiprocessing).

    Returns (time_per_atom, final_z).
    """
    import time
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    Fg, Fe, gFg, gFe = transition
    n_atoms = len(y0_list)
    y0_np_list = [np.array(y0) for y0 in y0_list]
    chunks = [y0_np_list[i::n_cores] for i in range(n_cores)]

    env = {
        '_PYLCP_BENCH_WORKER': '1',
        '_PYLCP_BENCH_T_FACTOR': str(t_factor),
        '_PYLCP_BENCH_FG': str(Fg),
        '_PYLCP_BENCH_FE': str(Fe),
        '_PYLCP_BENCH_GFG': str(gFg),
        '_PYLCP_BENCH_GFE': str(gFe),
    }
    # Spawned children re-exec Python and re-import this module, where the
    # _PYLCP_BENCH_WORKER guard pins JAX to CPU before pylcp_jax is imported.
    os.environ.update(env)
    ctx = mp.get_context('spawn')
    try:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_cores, mp_context=ctx) as pool:
            results_nested = list(pool.map(_parallel_worker, chunks))
        elapsed = time.perf_counter() - t0
    finally:
        for k in env:
            os.environ.pop(k, None)

    final_z = np.empty(n_atoms)
    for i, chunk in enumerate(results_nested):
        for j, z in enumerate(chunk):
            final_z[i + j * n_cores] = z

    return elapsed / n_atoms, final_z
