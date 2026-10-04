"""
Broadband (BB) red MOT simulation for Sr88 (1S0 -> 3P1, 689 nm).

Third cooling stage: atoms loaded from an upstream blue-transition stage
are held against a 3 MHz frequency sweep at 50 kHz repetition (triangle
waveform) that broadens the narrow-line capture range. The carrier
detuning sits at delta_center; the instantaneous frequency is
delta_center + delta_chirp(t) where delta_chirp is the triangle. The
analytic phase integral of the chirp is passed as `phase(t)` to each MOT
beam; the carrier stays in the Hamiltonian via `-det * I + H_e`, exactly
as in the blue MOT.

The upstream stage is selected with --upstream <pickle>; its sibling
constants.py is used to rescale natural units across the blue -> red
transition boundary. Default: low_power_blue_mot.
"""
import os

if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.94'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import sys
import time
import pickle
import argparse

import numpy as np
import jax
import jax.numpy as jnp

import pylcp_jax
import constants

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from init_atoms import initialize_from_pickle, load_src_constants

# ---------------------------------------------------------------------------
# CLI: choose which upstream stage feeds this one
# ---------------------------------------------------------------------------
_default_upstream = os.path.join(
    os.path.dirname(__file__), '..', 'low_power_blue_mot',
    'low_power_blue_mot_final_state.pkl',
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--upstream', default=_default_upstream,
    help='Upstream final-state pickle (default: %(default)s). The sibling '
         'constants.py in that directory is loaded for unit rescaling.',
)
args = parser.parse_args()
upstream_pickle = os.path.abspath(args.upstream)
upstream_dir = os.path.dirname(upstream_pickle)
upstream_name = os.path.basename(upstream_dir)

# Load the upstream stage's constants module by path (no __init__.py exists,
# and 'constants' is already bound to this stage's module).
import importlib.util
_src_path = os.path.join(upstream_dir, 'constants.py')
_spec = importlib.util.spec_from_file_location('src_constants', _src_path)
src_constants = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src_constants)

# ---------------------------------------------------------------------------
# Triangle-wave FM: analytic phase for the OBE coherent drive.
#
#   delta_chirp(t)  = triangle, amplitude A, period T (average 0)
#   phase(t)        = integral of delta_chirp -> piecewise quadratic,
#                     continuous across halves, zero at every period boundary
# ---------------------------------------------------------------------------
A_nat = 2 * np.pi * (constants.sweep_width_Hz / 2) / constants.gamma_real
T_nat = constants.gamma_real / constants.sweep_rate_Hz

def chirp_phase(t):
    tau  = jnp.mod(t, T_nat)
    half = T_nat / 2
    phi_up   = -A_nat * tau + (2 * A_nat / T_nat) * tau**2
    phi_down =  3 * A_nat * tau - (2 * A_nat / T_nat) * tau**2 - A_nat * T_nat
    return jnp.where(tau < half, phi_up, phi_down)

# ---------------------------------------------------------------------------
# Build the trap
# ---------------------------------------------------------------------------
print("Building BB red MOT setup (triangle FM)...")
trap_time = time.monotonic()

laserBeams = pylcp_jax.conventional3DMOTBeams(
    k=constants.kmag, s=constants.s, delta=0.,
    phase=chirp_phase,
    beam_type=pylcp_jax.infinitePlaneWaveBeam,
)
magField = pylcp_jax.quadrupoleMagneticField(constants.alpha_nat)

# 88Sr 1S0 (J=0) -> 3P1 (J=1), bosonic (I=0):
#   ground   g_J = 0
#   excited  g_J = 3/2  (3P1: L=1, S=1, J=1)
H_g, muq_g = pylcp_jax.hamiltonians.singleF(F=0, gF=0.0, muB=constants.muB)
H_e, muq_e = pylcp_jax.hamiltonians.singleF(F=1, gF=1.5, muB=constants.muB)
d_q = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(0, 1)

hamiltonian = pylcp_jax.hamiltonian(
    H_g, -constants.det * np.eye(3) + H_e, muq_g, muq_e, d_q,
    mass=constants.mass, muB=constants.muB, gamma=constants.gamma, k=constants.kmag,
)

obe = pylcp_jax.obe(laserBeams, magField, hamiltonian, a=constants.a_grav, transform_into_re_im=True)

# ---------------------------------------------------------------------------
# Load atoms from the upstream stage (blue -> red unit conversion)
# ---------------------------------------------------------------------------
rng = np.random.default_rng()
y0_batch, keys_batch = initialize_from_pickle(
    upstream_pickle, obe, dst_constants=constants,
    src_constants=src_constants,        # crosses 461 nm -> 689 nm
    n_atoms=constants.MAX_ATOMS, rng=rng,
)
Natoms = y0_batch.shape[0]

trap_time_total = time.monotonic() - trap_time
m, s = divmod(int(trap_time_total), 60)
h, m = divmod(m, 60)
print(f"Setup time: {h}h{m:02d}m{s:02d}s")

print(f"\n--- Initial conditions (loaded from {upstream_name}) ---")
print(f"  Atoms:            {Natoms}")
print(f"  |v| (natural):    {float(jnp.linalg.norm(y0_batch[:, -6:-3], axis=1).mean()):.1f}")
print(f"  |r| (natural):    {float(jnp.linalg.norm(y0_batch[:, -3:], axis=1).mean()):.1f}")
print(f"  carrier detuning: {constants.det:.2f} gamma")
print(f"  sweep amplitude:  {A_nat:.1f} gamma (half-width)")
print(f"  sweep period:     {T_nat:.3f} gamma^-1")
print(f"  saturation:       {constants.s}")
print(f"  B gradient:       {constants.alpha} T/m")
print()

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
print(f"Starting batched simulation of {Natoms} atoms on {jax.default_backend()}...")
t_total_start = time.monotonic()

sols = obe.evolve_motion(
    [0, constants.tmax],
    y0_batch=y0_batch,
    keys_batch=keys_batch,
    random_recoil=True,
    max_scatter_probability=0.5,
    n_points=1000,
    progress=True,
)

t_total = time.monotonic() - t_total_start
m, s = divmod(int(t_total), 60)
h, m = divmod(m, 60)
n_success = sum(1 for sol in sols if sol.success)
final_ts = np.array([float(sol._batched_state['t'][sol._index]) for sol in sols])
print(f"Simulation complete -- {len(sols)} trajectories in {h}h{m:02d}m{s:02d}s")
print(f"  {t_total/Natoms:.2f} s/atom")
print(f"  Reached tmax: {n_success}/{Natoms} ({100*n_success/Natoms:.0f}%)")
print(f"  Final t: min={final_ts.min():.0f}  median={np.median(final_ts):.0f}  max={final_ts.max():.0f}")

# ---------------------------------------------------------------------------
# Save results + final state
# ---------------------------------------------------------------------------
results = []
for sol in sols:
    results.append({
        't':        np.asarray(sol.t),
        'r':        np.asarray(sol.r),
        'v':        np.asarray(sol.v),
        'success':  sol.success,
        't_random': np.asarray(sol.t_random),
        'n_random': np.asarray(sol.n_random),
    })

with open('bb_red_mot_simulation_data.pkl', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save all atoms (capture thresholds are applied downstream / at analysis time)
r_final = np.array([res['r'][:, -1] for res in results])
v_final = np.array([res['v'][:, -1] for res in results])
final_state = {'r': r_final, 'v': v_final}

with open('bb_red_mot_final_state.pkl', 'wb') as f:
    pickle.dump(final_state, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Data saved to bb_red_mot_simulation_data.pkl")
print(f"Final state saved to bb_red_mot_final_state.pkl ({Natoms} atoms)")
