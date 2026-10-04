"""
Blue MOT simulation for Sr88 (1S0 -> 1P1, 461 nm) using GPU-batched OBE solver.

Initial conditions model atoms arriving from a Zeeman slower beam:
  - Longitudinal velocity (x): peaked near the slower capture velocity
  - Transverse velocity (y, z): small, set by beam divergence
  - Position: concentrated near the beam axis with a tuneable offset

All experimental parameters (detuning, saturation, gradient, beam
distributions) are in constants.py — edit that file to match your setup.
"""
import os

# Pre-allocate GPU memory before importing JAX.
if 'XLA_PYTHON_CLIENT_MEM_FRACTION' not in os.environ:
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.94'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import time
import numpy as np
import jax
import jax.numpy as jnp
import pickle

import pylcp_jax
import constants

# ---------------------------------------------------------------------------
# Build the trap
# ---------------------------------------------------------------------------
print("Building blue MOT setup...")
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

# Sr88 1S0 (J=0) -> 1P1 (J=1):  F=0 ground, F=1 excited
H_g, muq_g = pylcp_jax.hamiltonians.singleF(F=0, gF=0, muB=constants.muB)
H_e, muq_e = pylcp_jax.hamiltonians.singleF(F=1, gF=1, muB=constants.muB)
d_q = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(0, 1)

hamiltonian = pylcp_jax.hamiltonian(
    H_g, -constants.det * np.eye(3) + H_e, muq_g, muq_e, d_q,
    mass=constants.mass, muB=constants.muB, gamma=constants.gamma, k=constants.kmag
)

obe = pylcp_jax.obe(laserBeams, magField, hamiltonian, a=constants.a_grav, transform_into_re_im=True)

# ---------------------------------------------------------------------------
# Build batched initial conditions — Zeeman slower beam
# ---------------------------------------------------------------------------
state_dim = hamiltonian.n**2 + 6
Natoms = constants.MAX_ATOMS
print(f"State dim: {state_dim}, using Natoms={Natoms}")

rng = np.random.default_rng()

# Sample in beam frame (axis 0 = beam direction, 1,2 = transverse), then
# rotate into the lab frame so the beam axis can point off the x-axis.
r0_beam = constants.rscale_beam[None, :] * rng.standard_normal((Natoms, 3)) + constants.roffset_beam[None, :]
v0_beam = constants.vscale_beam[None, :] * rng.standard_normal((Natoms, 3)) + constants.voffset_beam[None, :]

# Clip negative longitudinal velocities — atoms travel along +beam_dir.
v0_beam[:, 0] = np.clip(v0_beam[:, 0], 0, None)

r0_all = r0_beam @ constants.R_beam.T
v0_all = v0_beam @ constants.R_beam.T

# Compute equilibrium rho0 at each atom's starting position/velocity
rho0_all = []
for i in range(Natoms):
    obe.set_initial_position(r0_all[i])
    obe.set_initial_velocity(v0_all[i])
    obe.set_initial_rho_from_rateeq()
    rho0_all.append(obe.rho0)

rho0_all = np.stack(rho0_all)
y0_batch = jnp.array(np.concatenate([rho0_all, v0_all, r0_all], axis=1))
keys_batch = jax.random.split(jax.random.PRNGKey(rng.integers(0, 2**31)), Natoms)

trap_time_total = time.monotonic() - trap_time
m, s = divmod(int(trap_time_total), 60)
h, m = divmod(m, 60)
print(f"Setup time: {h}h{m:02d}m{s:02d}s")

# ---------------------------------------------------------------------------
# Print initial condition summary
# ---------------------------------------------------------------------------
print(f"\n--- Initial conditions (Zeeman slower beam) ---")
print(f"  Atoms:           {Natoms}")
print(f"  beam_dir:        ({constants.beam_dir[0]:.3f}, {constants.beam_dir[1]:.3f}, {constants.beam_dir[2]:.3f})")
print(f"  v_long (nat):    {v0_beam[:, 0].mean():.1f} +/- {v0_beam[:, 0].std():.1f}")
print(f"  v_long (m/s):    {v0_beam[:, 0].mean() / (constants.kmag_real / constants.gamma_real):.1f}"
      f" +/- {v0_beam[:, 0].std() / (constants.kmag_real / constants.gamma_real):.1f}")
print(f"  v_trans (nat):   +/- {v0_beam[:, 1:].std():.1f}")
print(f"  r_trans (nat):   +/- {r0_beam[:, 1:].std():.0f}")
print(f"  r_long (nat):    {r0_beam[:, 0].mean():.0f} +/- {r0_beam[:, 0].std():.0f}")
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
# Save results
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

with open('blue_mot_simulation_data.pkl', 'wb') as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save all atoms (capture thresholds are applied downstream / at analysis time)
r_final = np.array([res['r'][:, -1] for res in results])
v_final = np.array([res['v'][:, -1] for res in results])
final_state = {'r': r_final, 'v': v_final}

with open('blue_mot_final_state.pkl', 'wb') as f:
    pickle.dump(final_state, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Data saved to blue_mot_simulation_data.pkl")
print(f"Final state saved to blue_mot_final_state.pkl ({Natoms} atoms)")
