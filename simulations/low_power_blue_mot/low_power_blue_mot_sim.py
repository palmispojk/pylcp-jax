"""
Low-power blue MOT simulation for Sr88 (1S0 -> 1P1, 461 nm).

Second stage of Kristensen's cooling sequence: after 950 ms of high-power
loading, the MOT-AOM drive is halved for 50 ms. Atoms are loaded from the
upstream blue_mot stage via the shared initialize_from_pickle helper; no
unit conversion is needed since the transition is unchanged.
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
from init_atoms import initialize_from_pickle

# ---------------------------------------------------------------------------
# CLI: choose which upstream stage feeds this one
# ---------------------------------------------------------------------------
_default_upstream = os.path.join(
    os.path.dirname(__file__), '..', 'blue_mot', 'blue_mot_final_state.pkl'
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--upstream', default=_default_upstream,
    help='Upstream final-state pickle (default: %(default)s).',
)
args = parser.parse_args()
upstream_pickle = os.path.abspath(args.upstream)
upstream_name = os.path.basename(os.path.dirname(upstream_pickle))

# ---------------------------------------------------------------------------
# Build the trap
# ---------------------------------------------------------------------------
print("Building low-power blue MOT setup...")
trap_time = time.monotonic()

laserBeams = pylcp_jax.laserBeams()
for kvec in ([1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]):
    laserBeams.add_laser(pylcp_jax.infinitePlaneWaveBeam(
        kvec=constants.kmag * np.array(kvec), pol=-1,
        s=constants.s, delta=0.,
    ))
for kvec in ([0., 0., 1.], [0., 0., -1.]):
    laserBeams.add_laser(pylcp_jax.infinitePlaneWaveBeam(
        kvec=constants.kmag * np.array(kvec), pol=+1,
        s=constants.s_z, delta=0.,
    ))
magField = pylcp_jax.quadrupoleMagneticField(constants.alpha_nat)

H_g, muq_g = pylcp_jax.hamiltonians.singleF(F=0, gF=0, muB=constants.muB)
H_e, muq_e = pylcp_jax.hamiltonians.singleF(F=1, gF=1, muB=constants.muB)
d_q = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(0, 1)

hamiltonian = pylcp_jax.hamiltonian(
    H_g, -constants.det * np.eye(3) + H_e, muq_g, muq_e, d_q,
    mass=constants.mass, muB=constants.muB, gamma=constants.gamma, k=constants.kmag,
)

obe = pylcp_jax.obe(laserBeams, magField, hamiltonian, a=constants.a_grav, transform_into_re_im=True)

# ---------------------------------------------------------------------------
# Load atoms from the upstream stage (same transition, no rescale)
# ---------------------------------------------------------------------------
rng = np.random.default_rng()
y0_batch, keys_batch = initialize_from_pickle(
    upstream_pickle, obe, dst_constants=constants,
    src_constants=None,                 # same transition -> no unit rescale
    n_atoms=constants.MAX_ATOMS, rng=rng,
)
Natoms = y0_batch.shape[0]

trap_time_total = time.monotonic() - trap_time
m, s = divmod(int(trap_time_total), 60)
h, m = divmod(m, 60)
print(f"Setup time: {h}h{m:02d}m{s:02d}s")

print(f"\n--- Initial conditions (loaded from {upstream_name}) ---")
print(f"  Atoms:           {Natoms}")
print(f"  |v| (natural):   {float(jnp.linalg.norm(y0_batch[:, -6:-3], axis=1).mean()):.2f}")
print(f"  |r| (natural):   {float(jnp.linalg.norm(y0_batch[:, -3:], axis=1).mean()):.1f}")
print(f"  detuning:        {constants.det:.2f} gamma")
print(f"  saturation:      xy={constants.s}, z={constants.s_z}")
print(f"  B gradient:      {constants.alpha} T/m")
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

with open('low_power_blue_mot_simulation_data.pkl', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save all atoms (capture thresholds are applied downstream / at analysis time)
r_final = np.array([res['r'][:, -1] for res in results])
v_final = np.array([res['v'][:, -1] for res in results])
final_state = {'r': r_final, 'v': v_final}

with open('low_power_blue_mot_final_state.pkl', 'wb') as f:
    pickle.dump(final_state, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Data saved to low_power_blue_mot_simulation_data.pkl")
print(f"Final state saved to low_power_blue_mot_final_state.pkl ({Natoms} atoms)")
