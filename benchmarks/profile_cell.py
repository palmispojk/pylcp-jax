"""Profile the exact cell from 02_F2_to_F3_1D_molasses.ipynb (figure 6)."""
import time
import numpy as np
import scipy.constants as cts
import pylcp_jax

atom = pylcp_jax.atom('23Na')
mass = (atom.state[2].gamma * atom.mass) / (cts.hbar * (100 * 2 * np.pi * atom.transition[1].k) ** 2)

def return_hamiltonian(Fl, Delta):
    Hg, Bgq = pylcp_jax.hamiltonians.singleF(F=Fl, gF=0, muB=1)
    He, Beq = pylcp_jax.hamiltonians.singleF(F=Fl+1, gF=1/(Fl+1), muB=1)
    dijq = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(Fl, Fl+1)
    return pylcp_jax.hamiltonian(Hg, -Delta*np.eye(He.shape[0])+He, Bgq, Beq, dijq, mass=mass)

def return_lasers(delta, s, pol):
    pol_coord = 'spherical' if (pol[0][2]>0 or pol[0][1]>0) else 'cartesian'
    return pylcp_jax.laserBeams([
        {'kvec': np.array([0., 0.,  1.]), 'pol': pol[0], 'pol_coord': pol_coord, 'delta': delta, 's': s},
        {'kvec': np.array([0., 0., -1.]), 'pol': pol[1], 'pol_coord': pol_coord, 'delta': delta, 's': s},
    ], beam_type=pylcp_jax.infinitePlaneWaveBeam)

magField = pylcp_jax.constantMagneticField(np.array([0., 0., 0.]))
pols = {
    'sig+sig-': [np.array([0., 0., 1.]), np.array([1., 0., 0.])],
    'sig+sig+': [np.array([0., 0., 1.]), np.array([0., 0., 1.])],
}

det = -2.73
s = 1.25
v = np.concatenate((np.array([0.0]), np.logspace(-2, np.log10(4), 20))) / np.sqrt(mass)
R = [np.zeros(v.shape)] * 3
V = [np.zeros(v.shape), np.zeros(v.shape), v]

print(f"Velocity points: {len(v)},  Hamiltonian size: ", end='')

for key, pol in list(pols.items())[:1]:  # profile just the first key
    laserBeams = return_lasers(0., s, pol)
    hamiltonian = return_hamiltonian(2, det)
    print(f"{hamiltonian.n} states")

    o = pylcp_jax.obe(laserBeams, magField, hamiltonian, transform_into_re_im=True)

    # Phase 1: rateeq construction (first set_initial_rho_from_rateeq)
    t0 = time.perf_counter()
    o.set_initial_position_and_velocity(np.zeros(3), np.array([0., 0., v[0]]))
    o.set_initial_rho_from_rateeq()
    t1 = time.perf_counter()
    print(f"\n[{key}]")
    print(f"  Phase 1 – rateeq init + first equilibrium_populations: {t1-t0:.3f} s")

    # Phase 2: remaining N-1 set_initial_rho calls
    t0 = time.perf_counter()
    for vi in v[1:]:
        o.set_initial_position_and_velocity(np.zeros(3), np.array([0., 0., vi]))
        o.set_initial_rho_from_rateeq()
    t2 = time.perf_counter()
    print(f"  Phase 2 – {len(v)-1} more equilibrium_populations calls: {t2-t0:.3f} s")

    # Phase 3: first evolve_density call (JIT compile)
    import jax.numpy as jnp
    o.set_initial_position_and_velocity(np.zeros(3), np.array([0., 0., v[0]]))
    o.set_initial_rho_from_rateeq()
    y0 = jnp.array(o.rho0)
    # Manually trigger just the first ODE solve to isolate JIT compile
    t0 = time.perf_counter()
    o.generate_force_profile(R, V, name='test',
                             deltat_tmax=2*np.pi*10, deltat_v=4, itermax=1,
                             progress_bar=False)
    t3 = time.perf_counter()
    print(f"  Phase 3 – first generate_force_profile (1 iteration): {t3-t0:.3f} s")
    print(f"  (includes JIT compile + {len(v)} ODE solves)")
