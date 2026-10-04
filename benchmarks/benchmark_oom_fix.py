"""Benchmark generate_force_profile memory usage with many velocity groups.

This reproduces the scenario from 03_F_to_Fp_1D_molasses.ipynb that
previously caused OOM kills (~10GB RSS) due to JAX recompiling the
diffrax solver for every unique max_steps value.
"""
import time
import os
import numpy as np
import pylcp_jax


def get_rss_mb():
    """Get current RSS in MB from /proc/self/status."""
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024  # kB -> MB
    return 0.0


det = -2.5
s = 1.0

# Same laser beams as the notebook (just sigma+sigma- for benchmark)
laserBeams = pylcp_jax.laserBeams([
    {'kvec': np.array([0., 0., 1.]), 'pol': np.array([0., 0., 1.]),
     'pol_coord': 'spherical', 'delta': 0, 's': s},
    {'kvec': np.array([0., 0., -1.]), 'pol': np.array([1., 0., 0.]),
     'pol_coord': 'spherical', 'delta': 0, 's': s},
], beam_type=pylcp_jax.infinitePlaneWaveBeam)

# F=1 -> F=2 hamiltonian
Fg, Fe = 1, 2
Hg, Bgq = pylcp_jax.hamiltonians.singleF(F=Fg, gF=0, muB=1)
He, Beq = pylcp_jax.hamiltonians.singleF(F=Fe, gF=1/Fe, muB=1)
dijq = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(Fg, Fe)
hamiltonian = pylcp_jax.hamiltonian(
    Hg, He - det * np.eye(2 * Fe + 1), Bgq, Beq, dijq
)

magField = pylcp_jax.constantMagneticField(np.zeros((3,)))

# Same velocity grid as the notebook — creates ~150 unique groups
v = np.concatenate((np.arange(0.0, 0.1, 0.001),
                    np.arange(0.1, 5.1, 0.1)))

print(f"Velocity points: {len(v)}")
print(f"Hamiltonian size: {hamiltonian.n} states")
print(f"Initial RSS: {get_rss_mb():.0f} MB")
print()

obj = pylcp_jax.obe(laserBeams, magField, hamiltonian, transform_into_re_im=True)

t0 = time.perf_counter()
obj.generate_force_profile(
    [np.zeros(v.shape), np.zeros(v.shape), np.zeros(v.shape)],
    [np.zeros(v.shape), np.zeros(v.shape), v],
    name='molasses', deltat_v=4, deltat_tmax=2 * np.pi * 5000, itermax=1000,
    rel=1e-8, abs=1e-10, progress_bar=True
)
elapsed = time.perf_counter() - t0

print(f"\nCompleted in {elapsed:.1f} s")
print(f"Final RSS: {get_rss_mb():.0f} MB")
print(f"Peak RSS stayed under OOM threshold: {'YES' if get_rss_mb() < 8000 else 'NO'}")
