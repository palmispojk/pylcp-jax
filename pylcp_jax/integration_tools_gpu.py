"""
GPU-accelerated ODE integration tools using JAX and diffrax.

Provides :func:`solve_ivp_random` (batched stochastic trajectories with
disk-backed memmap output) and :func:`solve_ivp_dense` (deterministic batched
dense output on a fixed time grid).  Both are GPU-native, JIT-compiled, and
support multi-GPU sharding.  Helper functions for profiling GPU memory and
throughput (:func:`optimal_batch_size`) are also included.
"""

import inspect
import logging
import os
import tempfile
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress XLA Triton tiling warnings
import jax

jax.config.update("jax_enable_x64", True)
import functools

import jax.numpy as jnp
from diffrax import (
    Bosh3,
    Dopri5,
    Kvaerno5,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
)
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

_log = logging.getLogger(__name__)

# Enable verbose GPU diagnostics with PYLCP_GPU_DEBUG=1
_GPU_DEBUG = os.environ.get("PYLCP_GPU_DEBUG", "0") == "1"


def _gpu_devices():
    """Return all available GPU devices on this node."""
    return [d for d in jax.devices() if d.platform == "gpu"]


def _gpu_device_info(gpu_devs=None):
    """Return a list of per-GPU info dicts.

    Each dict contains:
        device, device_kind, process_index,
        bytes_limit, bytes_in_use, peak_bytes_in_use, bytes_free
    """
    if gpu_devs is None:
        gpu_devs = _gpu_devices()
    info = []
    for d in gpu_devs:
        entry = {
            "device": d,
            "device_kind": getattr(d, "device_kind", "unknown"),
            "process_index": getattr(d, "process_index", 0),
        }
        stats = d.memory_stats()
        if stats is not None:
            entry["bytes_limit"] = stats["bytes_limit"]
            entry["bytes_in_use"] = stats["bytes_in_use"]
            entry["peak_bytes_in_use"] = stats["peak_bytes_in_use"]
            entry["bytes_free"] = stats["bytes_limit"] - stats["bytes_in_use"]
        else:
            entry["bytes_limit"] = 0
            entry["bytes_in_use"] = 0
            entry["peak_bytes_in_use"] = 0
            entry["bytes_free"] = 0
        info.append(entry)
    return info


def _log_gpu_debug(gpu_devs=None, label=""):
    """Print detailed GPU device info when PYLCP_GPU_DEBUG=1."""
    if not _GPU_DEBUG:
        return
    infos = _gpu_device_info(gpu_devs)
    prefix = f"[GPU DEBUG{': ' + label if label else ''}]"
    _log.info(f"{prefix} {len(infos)} GPU(s) detected")
    for i, info in enumerate(infos):
        d = info["device"]
        _log.info(
            f"{prefix}   GPU {i}: {d}"
            f"  kind={info['device_kind']}"
            f"  process={info['process_index']}"
            f"  pool={info['bytes_limit'] / 2**30:.2f} GiB"
            f"  in_use={info['bytes_in_use'] / 2**20:.1f} MiB"
            f"  peak={info['peak_bytes_in_use'] / 2**20:.1f} MiB"
            f"  free={info['bytes_free'] / 2**20:.1f} MiB"
        )


def _shard_batch(arr, gpu_devs):
    """Shard ``arr`` along axis 0 across ``gpu_devs``.

    Pads the batch dimension to be evenly divisible by ``len(gpu_devs)``
    if necessary.  Returns ``(sharded_arr, original_N)``.
    """
    n_gpus = len(gpu_devs)
    if n_gpus <= 1:
        return arr, arr.shape[0]

    N = arr.shape[0]
    remainder = N % n_gpus
    pad_size = 0
    if remainder != 0:
        pad_size = n_gpus - remainder
        pad_shape = (pad_size,) + arr.shape[1:]
        arr = jnp.concatenate([arr, jnp.zeros(pad_shape, dtype=arr.dtype)])

    if _GPU_DEBUG:
        per_gpu = arr.shape[0] // n_gpus
        _log.info(
            f"[GPU DEBUG: _shard_batch] N={N}, n_gpus={n_gpus}, "
            f"padded={pad_size}, per_gpu={per_gpu}, "
            f"arr_shape={arr.shape}"
        )

    mesh = Mesh(gpu_devs, axis_names=("batch",))
    sharding = NamedSharding(mesh, P("batch"))
    return jax.device_put(arr, sharding), N


def _ensure_3arg(func):
    """Wrap ``func`` to match diffrax's ``(t, y, args)`` signature.

    2-arg functions are wrapped so the ``args`` parameter is ignored.
    """
    sig = inspect.signature(func)
    n_required = sum(
        1
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    if n_required < 3:
        return lambda t, y, args: func(t, y)
    return func


class RandomOdeResult:
    """Result of a single ODE trajectory.

    Supports eager (all arrays passed in) or lazy construction from a
    batched state dict.  Lazy mode defers GPU→host transfer until first
    attribute access.

    Callers may attach additional per-trajectory arrays after construction:

    * ``v`` : velocity, shape ``(3, n_steps)``
    * ``r`` : position, shape ``(3, n_steps)``
    * ``rho`` : density matrix, shape ``(n, n, n_steps)`` (OBE only)
    """

    v: np.ndarray | None
    r: np.ndarray | None
    rho: np.ndarray | None

    def __init__(
        self,
        t=None,
        y=None,
        t_random=None,
        n_random=None,
        inds_random=None,
        success=None,
        status=0,
        message="",
        nfev=0,
        _batched_state=None,
        _index=None,
        _tf=None,
    ):
        if _batched_state is not None:
            # Lazy mode: defer slicing until attribute access.
            self._batched_state = _batched_state
            self._index = _index
            self._tf = _tf
            self._cache = {}
        else:
            # Eager mode: store pre-sliced arrays directly.
            self._batched_state = None
            self.t = t
            self.y = y
            self.t_random = t_random
            self.n_random = n_random
            self.inds_random = inds_random
            self.success = success
            self.status = status
            self.message = message
            self.nfev = nfev

    def _materialise(self):
        """Slice this atom's data from the batched state (once)."""
        if "t" in self._cache:
            return
        i = self._index
        bs = self._batched_state
        assert bs is not None  # only called in lazy mode
        assert i is not None
        n_saved = int(bs["save_idx"][i])

        # Initial state is already in slot 0 of ts/ys (written by the
        # while_loop init), so just slice up to n_saved.
        self._cache["t"] = bs["ts"][i, :n_saved]
        self._cache["y"] = np.moveaxis(bs["ys"][i, :n_saved], 0, -1)

        t_rand_raw = bs["t_random"][i, :n_saved]
        n_rand_raw = bs["n_random"][i, :n_saved]
        scatter_mask = n_rand_raw > 0

        self._cache["t_random"] = t_rand_raw[scatter_mask]
        self._cache["n_random"] = n_rand_raw[scatter_mask]
        self._cache["inds_random"] = scatter_mask
        assert self._tf is not None
        self._cache["success"] = bool(bs["t"][i] >= self._tf)
        self._cache["status"] = 0 if self._cache["success"] else -1
        self._cache["message"] = (
            "Success" if self._cache["success"] else "Terminated early (did not reach tf)."
        )
        self._cache["nfev"] = int(bs["nfev"][i])

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails (lazy mode).
        if name.startswith("_"):
            raise AttributeError(name)
        self._materialise()
        try:
            val = self._cache[name]
        except KeyError:
            raise AttributeError(f"RandomOdeResult has no attribute {name!r}")
        # Promote to a real attribute so subsequent accesses skip __getattr__.
        setattr(self, name, val)
        return val


@functools.lru_cache(maxsize=None)
def _make_run_group(func, random_func, solver_type):
    """Return a JIT-compiled, vmapped single-step kernel.

    Keyed on (func, random_func, solver_type) so JAX reuses the compiled
    XLA program across calls with the same physics functions — including
    repeated evolve_motion calls on the same OBE instance.
    """
    if solver_type == "Dopri5":
        solver = Dopri5()
    elif solver_type == "Bosh3":
        solver = Bosh3()
    elif solver_type == "Kvaerno5":
        solver = Kvaerno5()
    else:
        raise ValueError(
            f"Solver '{solver_type}' is not one of the specified "
            f"solvers. Use 'Dopri5', 'Bosh3', or 'Kvaerno5'."
        )

    term = ODETerm(_ensure_3arg(func))

    @functools.partial(jax.jit, static_argnames=("rtol", "atol", "max_step_global"))
    def _run_group(carry_batch, max_step_global, rtol, atol, args):
        def _single_group(carry):
            def cond_fn(s):
                return s["t"] < s["t_save_next"]

            def body_fn(s):
                t_curr = s["t"]
                dt_curr = s["dt"]
                t_next = jnp.minimum(t_curr + dt_curr, s["t_save_next"])
                actual_dt = t_next - t_curr

                sol = diffeqsolve(
                    term,
                    solver,
                    t0=t_curr,
                    t1=t_next,
                    dt0=dt_curr,
                    y0=s["y"],
                    args=args,
                    stepsize_controller=PIDController(rtol=rtol, atol=atol),
                    saveat=SaveAt(t1=True),
                    max_steps=None,
                )
                y_next = sol.ys[-1]  # type: ignore[index]
                nfev_add = sol.stats["num_steps"]

                y_jump, n_scatters, dt_max_suggested, key_new = random_func(
                    t_next, y_next, actual_dt, s["key"], args
                )
                # If the step was clipped to t_save_next, dt_max_suggested
                # collapses to ~0 and would freeze the next save group;
                # reuse dt_curr (already a recoil-safe step) instead.
                dt_next = jnp.where(
                    actual_dt < dt_curr,
                    jnp.minimum(dt_curr, max_step_global),
                    jnp.minimum(dt_max_suggested, max_step_global),
                )

                return {
                    "t": t_next,
                    "y": y_jump,
                    "dt": dt_next,
                    "key": key_new,
                    "step_idx": s["step_idx"] + 1,
                    "nfev": s["nfev"] + nfev_add,
                    "last_t_random": jnp.where(n_scatters > 0, t_next, s["last_t_random"]),
                    "last_n_random": jnp.where(
                        n_scatters > 0,
                        jnp.int32(n_scatters),
                        s["last_n_random"],
                    ),
                    "t_save_next": s["t_save_next"],
                }

            return jax.lax.while_loop(cond_fn, body_fn, carry)

        return jax.vmap(_single_group)(carry_batch)

    return _run_group


def _batched_random_trajectories(
    func,
    random_func,
    t0,
    tf,
    y0_batch,
    keys_batch,
    max_step_global,
    rtol,
    atol,
    dt0,
    n_points,
    solver_type="Dopri5",
    args=None,
    progress=False,
    output_dir=None,
):
    """
    Batched stochastic trajectory solver with disk-backed output.

    Saves state at ``n_points`` evenly-spaced times.  For each save slot
    a JIT-compiled kernel (vmapped across atoms) advances every atom to
    the next save time using adaptive recoil-limited steps inside a
    ``jax.lax.while_loop``.  Output is written to memory-mapped files;
    only the small per-atom carry resides on GPU.
    """
    _run_group = _make_run_group(func, random_func, solver_type)

    N = y0_batch.shape[0]
    state_dim = y0_batch.shape[1]

    # -- Time-based save grid ----------------------------------------------
    # n_points evenly-spaced save boundaries from t0 to tf.
    # The host loop drives one group per save slot; the inner kernel
    # takes as many adaptive steps as needed to reach t_save_next.
    t_save_grid = np.linspace(float(t0), float(tf), n_points + 1)

    # -- Initialise carry (tiny, on GPU) -----------------------------------
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

    # -- Pre-allocate disk-backed output arrays (memmap) --------------------
    n_total = n_points + 1  # +1 for initial state in slot 0
    _tmpdir = output_dir or tempfile.gettempdir()

    def _make_mmap(name, shape, dtype):
        fd, path = tempfile.mkstemp(prefix=f"pylcp_{name}_", suffix=".mmap", dir=_tmpdir)
        os.close(fd)
        mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
        # Unlink immediately: on Linux the file stays accessible via the
        # memmap mapping until the object is GC'd, then disk is reclaimed.
        os.unlink(path)
        return mm

    ts_cpu = _make_mmap("ts", (N, n_total), np.float64)
    ys_cpu = _make_mmap("ys", (N, n_total, state_dim), np.float64)
    t_random_cpu = _make_mmap("trand", (N, n_total), np.float64)
    n_random_cpu = _make_mmap("nrand", (N, n_total), np.int32)

    # Slot 0: initial state
    ts_cpu[:, 0] = np.asarray(carry["t"])
    ys_cpu[:, 0, :] = np.asarray(carry["y"])

    # -- Host loop: one group per save slot --------------------------------
    _log_interval = max(1, n_points // 20)  # ~5 % increments
    _t_first_group = None

    for _gi in range(n_points):
        carry = _run_group(carry, max_step_global, rtol, atol, args)

        # Start timer after first group (which includes JIT compilation).
        if _gi == 0:
            _t_first_group = time.monotonic()

        if progress and _gi > 0 and _gi % _log_interval == 0:
            elapsed = time.monotonic() - _t_first_group  # type: ignore[operator]
            rate = _gi / elapsed
            eta = (n_points - _gi) / rate if rate > 0 else 0
            m, s = divmod(int(eta), 60)
            h, m = divmod(m, 60)
            print(
                f"\r  [{_gi}/{n_points}] {100 * _gi / n_points:.0f}%  ETA {h}h{m:02d}m{s:02d}s",
                end="",
                flush=True,
            )

        # Tiny transfer: scalars + one state vector per atom.
        save_idx = _gi + 1
        ts_cpu[:, save_idx] = np.asarray(carry["t"])
        ys_cpu[:, save_idx, :] = np.asarray(carry["y"])
        t_random_cpu[:, save_idx] = np.asarray(carry["last_t_random"])
        n_random_cpu[:, save_idx] = np.asarray(carry["last_n_random"])

        # Reset scatter accumulators and advance save target for next group.
        if _gi < n_points - 1:
            carry = {
                **carry,
                "last_t_random": jnp.zeros(N, dtype=jnp.float64),
                "last_n_random": jnp.zeros(N, dtype=jnp.int32),
                "t_save_next": jnp.full(N, t_save_grid[_gi + 2], dtype=jnp.float64),
            }

    if progress:
        print(f"\r  [{n_points}/{n_points}] 100%  done.          ")

    # All atoms save exactly n_points + 1 points (including initial state).
    save_idx_arr = np.full(N, n_points + 1, dtype=np.int32)

    return {
        "t": np.asarray(carry["t"]),
        "y": np.asarray(carry["y"]),
        "step_idx": np.asarray(carry["step_idx"]),
        "save_idx": save_idx_arr,
        "ts": ts_cpu,
        "ys": ys_cpu,
        "t_random": t_random_cpu,
        "n_random": n_random_cpu,
        "nfev": np.asarray(carry["nfev"]),
    }


def _bytes_per_atom(state_dim):
    """Estimated peak GPU allocation per atom in the batched solver.

    Output is streamed to disk-backed memmap files, so only the per-atom
    while_loop carry and diffrax RK stage buffers reside on device.
    """
    # Carry (double-buffered by XLA): y, t, dt, key, counters, t_save_next
    carry = 2 * (state_dim * 8 + 100)
    # Diffrax RK stages (Dopri5 + PID)
    diffrax_buf = 7 * state_dim * 8
    return int(carry + diffrax_buf)


def _probe_bytes_per_atom(state_dim):
    """Measure actual GPU bytes-per-atom by running a tiny probe.

    Runs a trivial ODE for N=2 and N=8 atoms, measures peak GPU memory
    for each, and returns the per-atom delta.  Falls back to the
    analytical estimate if no GPU is available.
    """
    gpu_devs = _gpu_devices()
    if not gpu_devs:
        return _bytes_per_atom(state_dim)

    from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve

    solver = Dopri5()

    def _noop_ode(t, y, args):
        return jnp.zeros_like(y)

    term = ODETerm(_noop_ode)

    save_every = 2

    def _single_group(carry):
        def cond_fn(s):
            return (s["t"] < 1.0) & (s["count"] < save_every)

        def body_fn(s):
            sol = diffeqsolve(
                term,
                solver,
                t0=s["t"],
                t1=s["t"] + s["dt"],
                dt0=s["dt"],
                y0=s["y"],
                args=None,
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
                saveat=SaveAt(t1=True),
                max_steps=None,
            )
            return {
                **s,
                "t": s["t"] + s["dt"],
                "y": sol.ys[-1],  # type: ignore[index]
                "count": s["count"] + 1,
            }

        init = {**carry, "count": jnp.int32(0)}
        final = jax.lax.while_loop(cond_fn, body_fn, init)
        return {k: v for k, v in final.items() if k != "count"}

    @jax.jit
    def _run_probe(carry_batch):
        return jax.vmap(_single_group)(carry_batch)

    def _make_carry(n):
        return {
            "t": jnp.zeros(n, dtype=jnp.float64),
            "y": jnp.zeros((n, state_dim), dtype=jnp.float64),
            "dt": jnp.full(n, 0.5, dtype=jnp.float64),
        }

    # Warm up JIT (compilation cost is fixed, not per-atom).
    _run_probe(_make_carry(2))
    jax.effects_barrier()

    # Run with N=2
    carry_small = _make_carry(2)
    _run_probe(carry_small)
    jax.effects_barrier()
    mem_2 = _gpu_device_info(gpu_devs)[0]["peak_bytes_in_use"]

    # Run with N=8 to get a reliable delta
    carry_large = _make_carry(8)
    _run_probe(carry_large)
    jax.effects_barrier()
    mem_8 = _gpu_device_info(gpu_devs)[0]["peak_bytes_in_use"]

    measured_bpa = max(1, (mem_8 - mem_2) // (8 - 2))

    # Fall back to analytical if measurement looks wrong
    analytical_bpa = _bytes_per_atom(state_dim)
    if measured_bpa < analytical_bpa:
        measured_bpa = analytical_bpa

    if _GPU_DEBUG:
        _log.info(
            f"[GPU DEBUG: _probe_bytes_per_atom] state_dim={state_dim}, "
            f"mem_2={mem_2 / 2**20:.1f} MiB, mem_8={mem_8 / 2**20:.1f} MiB, "
            f"measured={measured_bpa} B/atom ({measured_bpa / 2**20:.4f} MiB/atom), "
            f"analytical={analytical_bpa} B/atom"
        )

    # Clean up probe arrays
    del carry_small, carry_large
    jax.effects_barrier()

    return int(measured_bpa)


def _probe_throughput_cap(state_dim, threshold=1.15, max_n=131072, n_groups=5):
    """Find the batch size where GPU compute throughput saturates.

    Doubles N from 32 upward, running ``n_groups`` host-loop iterations
    at each size and measuring wall time.  Stops when doubling N yields
    less than ``threshold`` (default 15%) improvement in per-atom time,
    or when ``max_n`` is reached.

    Uses the same vmap + while_loop + diffrax kernel as the real solver
    but with a trivial ODE so each group completes in microseconds.

    Args:
        state_dim: Length of the per-atom state vector.
        threshold: Minimum speedup ratio (s_per_atom_prev / s_per_atom_cur)
            to keep doubling.  Default 1.15 (15% improvement).
        max_n: Hard upper limit on probe batch size.
        n_groups: Number of host-loop groups per measurement.  More groups
            give more stable timings.  Default 5.

    Returns
    -------
        int: Recommended batch size at the throughput knee, or ``max_n``
        if no saturation was detected.
    """
    gpu_devs = _gpu_devices()
    if not gpu_devs:
        return 256  # sensible CPU default

    from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve

    solver = Dopri5()

    def _noop_ode(t, y, args):
        return jnp.zeros_like(y)

    term = ODETerm(_noop_ode)
    save_every = 2

    def _single_group(carry):
        def cond_fn(s):
            return (s["t"] < 1.0) & (s["count"] < save_every)

        def body_fn(s):
            sol = diffeqsolve(
                term,
                solver,
                t0=s["t"],
                t1=s["t"] + s["dt"],
                dt0=s["dt"],
                y0=s["y"],
                args=None,
                stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
                saveat=SaveAt(t1=True),
                max_steps=None,
            )
            return {
                **s,
                "t": s["t"] + s["dt"],
                "y": sol.ys[-1],  # type: ignore[index]
                "count": s["count"] + 1,
            }

        init = {**carry, "count": jnp.int32(0)}
        final = jax.lax.while_loop(cond_fn, body_fn, init)
        return {k: v for k, v in final.items() if k != "count"}

    @jax.jit
    def _run_group(carry_batch):
        return jax.vmap(_single_group)(carry_batch)

    def _make_carry(n):
        return {
            "t": jnp.zeros(n, dtype=jnp.float64),
            "y": jnp.zeros((n, state_dim), dtype=jnp.float64),
            "dt": jnp.full(n, 0.5, dtype=jnp.float64),
        }

    # JIT warmup (compilation cost excluded from timings)
    _run_group(_make_carry(2))
    jax.effects_barrier()

    def _time_n(n):
        carry = _make_carry(n)
        # Warm run to fill caches
        _run_group(carry)
        jax.effects_barrier()
        # Timed run: multiple groups for stability
        t0 = time.monotonic()
        for _ in range(n_groups):
            carry = _run_group(carry)
            jax.effects_barrier()
        elapsed = time.monotonic() - t0
        del carry
        jax.effects_barrier()
        return elapsed / n  # seconds per atom (across n_groups groups)

    prev_spa = _time_n(32)
    knee_n = 32

    n = 64
    while n <= max_n:
        try:
            spa = _time_n(n)
        except Exception:
            # OOM or other error — previous N was the limit
            break

        speedup = prev_spa / spa if spa > 0 else 1.0

        if _GPU_DEBUG:
            _log.info(
                f"[GPU DEBUG: _probe_throughput_cap] N={n}, "
                f"s/atom={spa:.6f}, speedup={speedup:.2f}x"
            )

        if speedup < threshold:
            # Diminishing returns — previous N was the knee
            break

        knee_n = n
        prev_spa = spa
        n *= 2

    _log.info(f"[_probe_throughput_cap] state_dim={state_dim}, throughput knee at N={knee_n}")
    return knee_n


def optimal_batch_size(state_dim, safety=0.6, **_ignored):
    """Return the optimal batch size considering both GPU memory and throughput.

    Runs two probes at startup (~5-10 s total):

    1. **Memory probe**: measures actual per-atom GPU allocation by running
       a tiny kernel with N=2 and N=8 atoms.  Computes the maximum batch
       that fits in VRAM.
    2. **Throughput probe**: doubles batch size from 32 upward, measuring
       per-atom wall time at each step.  Stops when doubling N yields
       <15% speedup (GPU compute is saturated).

    The returned batch size is ``min(memory_cap, throughput_cap)`` —
    whichever ceiling is hit first.

    Args:
        state_dim (int): Length of the per-atom state vector (``y0.shape[-1]``).
        safety (float, optional): Fraction of free memory to use. Defaults to 0.6
            to leave headroom for XLA workspace and compilation buffers.

    Returns
    -------
        int: Recommended total batch size, or ``None`` if no GPU is present.
    """
    gpu_devs = _gpu_devices()
    if not gpu_devs:
        return None

    _log_gpu_debug(gpu_devs, "optimal_batch_size")

    # --- Memory ceiling ---
    infos = _gpu_device_info(gpu_devs)
    bpa = _probe_bytes_per_atom(state_dim)

    capacities = []
    for info in infos:
        if info["bytes_limit"] == 0:
            return None
        cap = max(1, int(info["bytes_free"] * safety / bpa))
        capacities.append(cap)

    min_cap = min(capacities)
    memory_cap = min_cap * len(gpu_devs)

    # --- Throughput ceiling ---
    throughput_cap = _probe_throughput_cap(state_dim)

    total = min(memory_cap, throughput_cap)

    _log.info(
        f"[optimal_batch_size] memory_cap={memory_cap} "
        f"({bpa / 2**20:.4f} MiB/atom), "
        f"throughput_cap={throughput_cap}, "
        f"result={total}"
    )

    if _GPU_DEBUG:
        for i, (info, cap) in enumerate(zip(infos, capacities)):
            _log.info(
                f"[GPU DEBUG: optimal_batch_size]   GPU {i}: "
                f"free={info['bytes_free'] / 2**20:.1f} MiB, "
                f"memory_cap={cap} atoms "
                f"({cap * bpa / 2**20:.1f} MiB at {bpa / 2**20:.3f} MiB/atom)"
            )

    if _GPU_DEBUG and len(gpu_devs) > 1:
        max_cap = max(capacities)
        wasted = sum(c - min_cap for c in capacities)
        _log.info(
            f"[GPU DEBUG: optimal_batch_size]   Even-shard total: {memory_cap} atoms "
            f"(min_cap={min_cap}, max_cap={max_cap}, "
            f"wasted capacity={wasted} atoms across {len(gpu_devs)} GPUs)"
        )

    return total


def optimal_batch_size_per_gpu(state_dim, safety=0.6, **_ignored):
    """Return per-GPU atom capacities for heterogeneous multi-GPU setups.

    Unlike :func:`optimal_batch_size` which returns a single total capped by
    the smallest GPU, this function returns each GPU's individual capacity so
    callers can assign proportional work.

    Args:
        state_dim (int): Length of the per-atom state vector.
        safety (float, optional): Fraction of free memory to use. Defaults to 0.6.

    Returns
    -------
        list of (jax.Device, int) pairs, or ``None`` if no GPU is present.
        The int is the number of atoms that GPU can handle.
    """
    gpu_devs = _gpu_devices()
    if not gpu_devs:
        return None

    infos = _gpu_device_info(gpu_devs)
    bpa = _probe_bytes_per_atom(state_dim)

    result = []
    for info in infos:
        if info["bytes_limit"] == 0:
            return None
        cap = max(1, int(info["bytes_free"] * safety / bpa))
        result.append((info["device"], cap))
    return result


def solve_ivp_random(
    fun,
    random_func,
    t_span,
    y0_batch,
    keys_batch,
    n_points,
    solver_type="Dopri5",
    max_step=float("inf"),
    rtol=1e-5,
    atol=1e-6,
    batch_size=None,
    args=None,
    progress=False,
    output_dir=None,
    **options,
):
    """
    GPU-batched ODE solver with stochastic jumps.

    Parallel replacement for ``scipy.integrate.solve_ivp``.  Saves at
    ``n_points`` evenly-spaced times (like ``solve_ivp(t_eval=...)``).

    Args:
        fun: RHS of the ODE, signature ``fun(t, y)`` or ``fun(t, y, args)``.
        random_func: Stochastic jump function, returns
            ``(y_jump, n_scatters, dt_max, key_new)``.
        t_span: ``(t0, tf)`` integration interval.
        y0_batch: Initial states, shape ``(N, state_dim)``.
        keys_batch: JAX PRNGKeys, one per trajectory.
        solver_type: ``'Dopri5'``, ``'Bosh3'``, or ``'Kvaerno5'``.
        max_step: Ceiling on step size.  Default: ``inf``.
            Set to reduce GPU warp divergence on large batches.
        rtol, atol: Tolerances for the adaptive step controller.
        batch_size: Max atoms per chunk.  Default: ``None`` (all at once).
        n_points: Number of evenly-spaced output points.
        output_dir: Directory for temporary memmap files.

    Returns
    -------
        list of RandomOdeResult with attributes ``t``, ``y``, ``success``,
        ``t_random``, ``n_random``, ``nfev``.
    """
    # cast to jnp just in case
    y0_batch = jnp.asarray(y0_batch)
    keys_batch = jnp.asarray(keys_batch)
    t0, tf = (
        jnp.asarray(t_span[0], dtype=jnp.float64),
        jnp.asarray(t_span[1], dtype=jnp.float64),
    )
    t_range = float(tf - t0)
    dt0 = jnp.minimum(
        jnp.asarray(t_range * 1e-3, dtype=jnp.float64),
        jnp.asarray(max_step, dtype=jnp.float64),
    )
    N = y0_batch.shape[0]

    if batch_size is None:
        batch_size = N

    gpu_devs = _gpu_devices()
    n_gpus = len(gpu_devs)

    if _GPU_DEBUG:
        _log_gpu_debug(gpu_devs, "solve_ivp_random")
        _log.info(
            f"[GPU DEBUG: solve_ivp_random] N={N}, batch_size={batch_size}, "
            f"n_gpus={n_gpus}, state_dim={y0_batch.shape[-1]}, "
            f"n_points={n_points}"
        )

    def _run_chunk(y0_chunk, keys_chunk, chunk_N):
        """Run a single chunk, sharding across GPUs if multiple are available."""
        if n_gpus > 1:
            y0_chunk, _ = _shard_batch(y0_chunk, gpu_devs)
            keys_chunk, _ = _shard_batch(keys_chunk, gpu_devs)
        batched_state = _batched_random_trajectories(
            fun,
            random_func,
            t0,
            tf,
            y0_chunk,
            keys_chunk,
            max_step,
            rtol,
            atol,
            dt0,
            n_points=n_points,
            solver_type=solver_type,
            args=args,
            progress=progress,
            output_dir=output_dir,
        )
        tf_float = float(tf)
        return [
            RandomOdeResult(_batched_state=batched_state, _index=i, _tf=tf_float)
            for i in range(chunk_N)
        ]

    # When sharding across multiple GPUs, round chunk sizes up to be
    # divisible by n_gpus so that _shard_batch padding is minimal.
    if n_gpus > 1 and batch_size >= n_gpus:
        batch_size = (batch_size // n_gpus) * n_gpus

    if batch_size >= N:
        return _run_chunk(y0_batch, keys_batch, N)

    # Chunked execution to limit peak memory.
    results = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        results.extend(_run_chunk(y0_batch[start:end], keys_batch[start:end], end - start))
    return results


@functools.partial(
    jax.jit,
    static_argnames=(
        "func",
        "n_points",
        "max_steps",
        "rtol",
        "atol",
        "solver_type",
    ),
)
def _batched_dense_trajectories(
    func,
    t0,
    t1,
    y0_batch,
    n_points,
    max_steps=4096,
    rtol=1e-5,
    atol=1e-6,
    solver_type="Dopri5",
    args=None,
):
    """
    JIT-compiled batched ODE solve on a fixed time grid.

    Compiled once per unique (func, n_points, rtol, atol, solver_type);
    reused for different t_span / y0_batch / args values.

    Args:
        func: ODE RHS, ``(t, y)`` or ``(t, y, args)``.  Use a stable
              (e.g. module-level) function for JIT cache reuse.
        t0, t1: Start and end time (JAX float64 scalars).
        y0_batch: Initial conditions, shape (N, state_dim).
        n_points: Number of equally-spaced output time points (static).
        max_steps: Maximum number of adaptive solver steps (static). Default: 4096.
        rtol, atol: Adaptive step-size controller tolerances (static).
        solver_type: 'Dopri5', 'Bosh3', or 'Kvaerno5' (static).
        args: JAX pytree passed to func. Traced dynamically. Default: None.

    Returns
    -------
        ys: shape (N, n_points, state_dim)
        ts: shape (n_points,)
    """
    if solver_type == "Dopri5":
        solver = Dopri5()
    elif solver_type == "Bosh3":
        solver = Bosh3()
    elif solver_type == "Kvaerno5":
        solver = Kvaerno5()
    else:
        raise ValueError(
            f"Solver '{solver_type}' not recognised. Use 'Dopri5', 'Bosh3', or 'Kvaerno5'."
        )

    term = ODETerm(_ensure_3arg(func))

    # t0/t1 can be scalars (shared) or per-atom arrays (shape (N,)).
    # When per-atom, each atom gets its own time grid and integration
    # window — all run in parallel via vmap with a single JIT trace.
    per_atom = jnp.ndim(t0) > 0 or jnp.ndim(t1) > 0

    if not per_atom:
        # Shared t_span: single ts_grid, vmap over y0 only
        ts_grid = jnp.linspace(t0, t1, n_points)

        def solve_one(y0):
            sol = diffeqsolve(
                term,
                solver,
                t0=t0,
                t1=t1,
                dt0=jnp.asarray((t1 - t0) / n_points, dtype=t0.dtype),
                y0=y0,
                args=args,
                stepsize_controller=PIDController(rtol=rtol, atol=atol),
                saveat=SaveAt(ts=ts_grid),
                max_steps=max_steps,
            )
            return sol.ys, ts_grid

        ys, ts_batch = jax.vmap(solve_one)(y0_batch)
        return ys, ts_batch[0]  # ts identical for all atoms
    else:
        # Per-atom t0/t1: each atom has its own time grid
        # Broadcast scalars to match batch dimension
        if jnp.ndim(t0) == 0:
            t0 = jnp.broadcast_to(t0, t1.shape)
        if jnp.ndim(t1) == 0:
            t1 = jnp.broadcast_to(t1, t0.shape)

        def solve_one_per_atom(y0, t0_i, t1_i):
            ts_grid_i = jnp.linspace(t0_i, t1_i, n_points)
            sol = diffeqsolve(
                term,
                solver,
                t0=t0_i,
                t1=t1_i,
                dt0=jnp.asarray((t1_i - t0_i) / n_points, dtype=t0_i.dtype),
                y0=y0,
                args=args,
                stepsize_controller=PIDController(rtol=rtol, atol=atol),
                saveat=SaveAt(ts=ts_grid_i),
                max_steps=max_steps,
            )
            return sol.ys, ts_grid_i

        ys, ts_batch = jax.vmap(solve_one_per_atom)(y0_batch, t0, t1)
        return ys, ts_batch  # (N, n_points)


def solve_ivp_dense(
    func,
    t_span,
    y0_batch,
    n_points=1001,
    max_steps=4096,
    rtol=1e-5,
    atol=1e-6,
    solver_type="Dopri5",
    args=None,
):
    """
    Batched ODE solve returning solution on a fixed time grid (GPU-native).

    Compiled once per (func, n_points, rtol, atol, solver_type); reused
    for different t_span / y0_batch / args.

    Args:
        func: ODE RHS, ``(t, y)`` or ``(t, y, args)``.
        t_span: (t0, t1) integration interval.  t1 may be a scalar
                (shared endpoint for all atoms) or an array of shape (N,)
                giving a per-atom endpoint.  When per-atom, the returned
                ``ts`` has shape (N, n_points) instead of (n_points,).
        y0_batch: Initial conditions, shape (N, state_dim).
        n_points: Number of equally-spaced output time points. Default 1001.
        max_steps: Maximum number of adaptive solver steps. Default: 4096.
        rtol: Relative tolerance. Default 1e-5.
        atol: Absolute tolerance. Default 1e-5.
        solver_type: 'Dopri5' (default), 'Bosh3', or 'Kvaerno5'.
        args: JAX pytree passed through to func as its third argument.
              Default: None.

    Returns
    -------
        ts: jax.Array, shape (n_points,) or (N, n_points) for per-atom t1
        ys: jax.Array, shape (N, n_points, state_dim)
    """
    t0 = jnp.asarray(t_span[0], dtype=jnp.float64)
    t1 = jnp.asarray(t_span[1], dtype=jnp.float64)
    y0_batch = jnp.asarray(y0_batch)

    # Shard across multiple GPUs when available.
    gpu_devs = _gpu_devices()
    if _GPU_DEBUG:
        _log_gpu_debug(gpu_devs, "solve_ivp_dense")
        _log.info(
            f"[GPU DEBUG: solve_ivp_dense] N={y0_batch.shape[0]}, "
            f"n_gpus={len(gpu_devs)}, state_dim={y0_batch.shape[-1]}, "
            f"n_points={n_points}"
        )
    if len(gpu_devs) > 1:
        y0_batch, orig_N = _shard_batch(y0_batch, gpu_devs)
        if jnp.ndim(t1) > 0:
            t1, _ = _shard_batch(t1, gpu_devs)
        if jnp.ndim(t0) > 0:
            t0, _ = _shard_batch(t0, gpu_devs)
    else:
        orig_N = y0_batch.shape[0]

    ys, ts = _batched_dense_trajectories(
        func,
        t0,
        t1,
        y0_batch,
        n_points,
        max_steps,
        rtol,
        atol,
        solver_type,
        args,
    )
    # Strip padding atoms added by _shard_batch.
    if ys.shape[0] > orig_N:
        ys = ys[:orig_N]
        if ts.ndim > 1:
            ts = ts[:orig_N]
    return ts, ys


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    def dydt(t, y):
        """Return the derivative for a simple harmonic oscillator."""
        return jnp.array([-y[1], y[0]])

    def func2(t, y, dt, key, args):
        """Apply a random kick with probability proportional to dt."""
        key, subkey1, subkey2 = jax.random.split(key, 3)
        roll = jax.random.uniform(subkey1)
        did_scatter = roll < 2 * dt
        kick = 5 * jax.random.normal(subkey2)
        y_jump = jnp.where(did_scatter, y.at[1].add(kick), y)
        n_events = jnp.where(did_scatter, 1, 0)
        max_dt = jnp.maximum(0.1, jnp.abs(y_jump[1]))
        return y_jump, n_events, max_dt, key

    # initial conditions
    y0_batch = jnp.array(
        [
            [0.0, 1.0],  # Atom 1
            [0.0, 1.0],  # Atom 2
            [0.0, 1.0],  # Atom 3
        ]
    )

    # 3 unique keys, one for each atom
    initial_key = jax.random.PRNGKey(42)
    keys_batch = jax.random.split(initial_key, 3)

    # Runs the batched solver on 3 atoms
    sols = solve_ivp_random(
        dydt,
        func2,
        [0.0, 10 * jnp.pi],
        y0_batch,
        keys_batch,
        max_step=0.1,
        solver_type="Dopri5",
        n_points=500,
    )

    # only plot one atom instead of all 3
    sol = sols[0]

    plt.figure()
    plt.plot(sol.t, sol.y.T)  # type: ignore[attr-defined]

    if len(sol.t_random) > 0:  # type: ignore[arg-type]  # pyright: ignore[reportOptionalSubscript]
        plt.plot(
            sol.t_random,  # pyright: ignore[reportArgumentType]
            sol.y[:, sol.inds_random].T,  # pyright: ignore[reportOptionalSubscript]
            "o",
            color="black",
            label="Random Kicks",
        )  # type: ignore[attr-defined]
        plt.legend()

    plt.xlabel("Time")
    plt.ylabel("State (Position & Velocity)")
    plt.title("Batched GPU Integration (Atom 1)")
    plt.show()
