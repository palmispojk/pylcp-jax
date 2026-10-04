# pylcp_jax

[![Tests](https://github.com/palmispojk/pylcp-jax/actions/workflows/tests.yml/badge.svg)](https://github.com/palmispojk/pylcp-jax/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/palmispojk/pylcp-jax/branch/master/graph/badge.svg)](https://codecov.io/gh/palmispojk/pylcp-jax)

`pylcp_jax` (Python Laser Cooling Physics) simulates laser cooling and trapping of
atoms and molecules. Give it an internal Hamiltonian, a set of laser beams and
a magnetic field, and it builds the governing equations for you: the full
optical Bloch equations (OBE), the rate equations, or a heuristic force model.
You can then compute force profiles, evolve internal states, or follow
Monte Carlo trajectories of many atoms, including random photon recoil.

This version is a GPU-accelerated rewrite of the original
[pylcp](https://github.com/JQIamo/pylcp). It keeps the original physics model
and user-facing API, replaces the NumPy/SciPy backend with
[JAX](https://github.com/jax-ml/jax) and
[Diffrax](https://github.com/patrick-kidger/diffrax), and adds batched
trajectory evolution, multi-GPU sharding, a test suite and CI.
It was developed for the thesis *Accelerating Monte Carlo Trajectory
Simulations of Cold Atoms*. If you use the physics model in research, please
cite the original paper: S. Eckel, D. S. Barker, E. B. Norrgard and J. Scherschligt,
"PyLCP: A Python package for computing laser cooling physics",
Computer Physics Communications 270, 108166 (2022).

## Status

> **Early alpha (v0.0.1).** The code and tests are in good shape, but the API may
> still change and the packaging is new.
>
> **The documentation needs updating.** The Sphinx docs in `docs/` and the example
> notebooks are inherited from the original pylcp. They still use its wording,
> install instructions and version notes, and some text may not match this
> package's JAX-based behaviour (for example `evolve_motion` now needs `n_points`,
> and the import name is `pylcp_jax`). Treat this README and the scripts in
> `simulations/` and `benchmarks/` as the up-to-date reference until the docs are revised.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Simulating many atoms on a GPU](#simulating-many-atoms-on-a-gpu)
- [Governing equations](#governing-equations)
- [Examples and documentation](#examples-and-documentation)
- [Benchmarks](#benchmarks)
- [Simulations: Sr-88 cooling cascade](#simulations-sr-88-cooling-cascade)
- [Repository layout](#repository-layout)
- [Contributing](#contributing)

## Features

- **Three levels of modelling**: full OBE (`pylcp_jax.obe`), rate equations
  (`pylcp_jax.rateeq`) and a heuristic model (`pylcp_jax.heuristiceq`), all sharing
  the same Hamiltonian, laser and field objects.
- **JIT-compiled JAX/Diffrax backend** for forces, density matrices and
  equations of motion.
- **Batched `evolve_motion`**: thousands of atoms are integrated in parallel
  with `jax.vmap` on a GPU, or in a serial / multi-process CPU loop
  (`backend="auto" | "gpu" | "cpu"`). Multi-GPU sharding uses JAX's GSPMD.
- **Random recoil** from scattered photons, with an automatic step-size cap so
  atoms in a batch take similarly sized steps.
- **Reliable force profiles**: a convergence guard for dark-state detection in
  `generate_force_profile` fixes wrong results for type-II optical molasses.
- **Building blocks**: atomic data for common species (`pylcp_jax.atom`),
  Hamiltonians for single and coupled hyperfine manifolds and molecules
  (`pylcp_jax.hamiltonians`), plane-wave, Gaussian and clipped-Gaussian beams,
  conventional 3D MOT beam sets, and quadrupole, constant and Ioffe-Pritchard
  magnetic fields.
- **Tested**: a `pytest` suite, static analysis (`ruff`, `pyright`) and CI.

## Installation

Requires Python >= 3.11 (tested on 3.11 to 3.14).

From PyPI:
```
pip install pylcp-jax
```
The package is installed as `pylcp-jax` and imported as `pylcp_jax`.

For a CUDA-capable GPU (CUDA 12):
```
pip install "pylcp-jax[cuda]"
```

Optional extras: `plot` (matplotlib, for the built-in plotting helpers) and
`gratings` (numba, only needed for `pylcp_jax.gratings`), e.g.
`pip install "pylcp-jax[cuda,plot]"`.

Check that JAX sees your GPU with `python -c "import jax; print(jax.devices())"`.
Without a GPU everything still runs, on the CPU.

### From source

To develop, run the simulations or build the docs, install from a clone with
[uv](https://docs.astral.sh/uv/):
```
git clone https://github.com/palmispojk/pylcp-jax/
cd pylcp-jax
uv sync                          # add --extra cuda for a GPU
uv sync --group dev --group docs # tests, linting, docs and notebooks
```

## Quick start

The workflow is: define the **Hamiltonian**, the **laser beams** and the
**magnetic field**, combine them in a governing equation, then compute
something. Here is a Sr-88 `1S0 -> 1P1` (F=0 -> F'=1) blue MOT in natural
units (`gamma = k = muB = 1`, lengths in 1/k, times in 1/gamma):

```python
import numpy as np
import pylcp_jax

det, s, alpha, mass = -1.3, 0.2, 0.01, 1e5   # detuning, saturation, field gradient, mass

# 1. Hamiltonian: ground F=0, excited F=1, detuned by `det`
H_g, mu_g = pylcp_jax.hamiltonians.singleF(F=0, gF=0, muB=1)
H_e, mu_e = pylcp_jax.hamiltonians.singleF(F=1, gF=1, muB=1)
d_q = pylcp_jax.hamiltonians.dqij_two_bare_hyperfine(0, 1)
hamiltonian = pylcp_jax.hamiltonian(H_g, -det * np.eye(3) + H_e, mu_g, mu_e, d_q,
                                mass=mass, muB=1, gamma=1, k=1)

# 2. Six MOT beams (circular polarisation +/-1, detuning set in the Hamiltonian)
beams = pylcp_jax.laserBeams()
for k, pol in [([1, 0, 0], -1), ([-1, 0, 0], -1), ([0, 1, 0], -1),
               ([0, -1, 0], -1), ([0, 0, 1], +1), ([0, 0, -1], +1)]:
    beams.add_laser(pylcp_jax.infinitePlaneWaveBeam(
        kvec=np.array(k, float), pol=pol, s=s, delta=0.))

# 3. Quadrupole magnetic field (gradient `alpha`)
B = pylcp_jax.quadrupoleMagneticField(alpha)

# 4. Governing equation (full OBE)
obe = pylcp_jax.obe(beams, B, hamiltonian, transform_into_re_im=True)

# 5a. Force profile along x, from an atom at rest to 10 gamma/k
x = np.linspace(-20, 20, 41)
zero = np.zeros_like(x)
obe.generate_force_profile([x, zero, zero], [zero, zero, zero], name="x")
force_x = obe.profile["x"].F[0]

# 5b. ...or follow one atom for 1e4 / gamma, with random photon recoil
obe.set_initial_position(np.array([5., 0., 0.]))
obe.set_initial_velocity(np.zeros(3))
obe.set_initial_rho_from_rateeq()
sols = obe.evolve_motion([0, 1e4], n_points=500, random_recoil=True)
t, r, v = sols[0].t, sols[0].r, sols[0].v
```

Pick the model by swapping `pylcp_jax.obe` for `pylcp_jax.rateeq` (fast, no coherences)
or `pylcp_jax.heuristiceq` (two-level heuristic force). For alkali atoms,
`pylcp_jax.atom("87Rb")` (also 6Li, 7Li, 23Na, 39K, 40K, 41K, 85Rb) provides
reference transition data so you don't have to type in parameters.

## Simulating many atoms on a GPU

`obe.evolve_motion` accepts a batch of initial conditions. Each row of
`y0_batch` is `[rho (flattened), v (3), r (3)]`, one row per atom, and
`keys_batch` holds one JAX random key per atom:

```python
import jax, jax.numpy as jnp

rng = np.random.default_rng()
N = 4096
r0 = 100 * rng.standard_normal((N, 3))                  # positions (1/k)
v0 = 2 * rng.standard_normal((N, 3))                    # velocities (gamma/k)

rho0 = []
for ri, vi in zip(r0, v0):                              # equilibrium start state per atom
    obe.set_initial_position(ri)
    obe.set_initial_velocity(vi)
    obe.set_initial_rho_from_rateeq()
    rho0.append(obe.rho0)

y0_batch = jnp.array(np.concatenate([np.stack(rho0), v0, r0], axis=1))
keys_batch = jax.random.split(jax.random.PRNGKey(0), N)

sols = obe.evolve_motion([0, 1e5], n_points=1000, y0_batch=y0_batch,
                         keys_batch=keys_batch, random_recoil=True,
                         max_scatter_probability=0.5, backend="auto")
```

Useful options: `freeze_axis=[False, False, True]` to fix motion along an axis,
`backend="cpu"` to force the serial path, and solver settings (`rtol`, `atol`,
`solver_type="Dopri5" | "Bosh3" | "Kvaerno5"`). Memory grows with the batch
size and state dimension (`n**2 + 6` per atom), so reduce `N` if you run out of
GPU memory. The scripts in [`simulations/`](simulations) are complete,
working examples of this pattern.

## Governing equations

| Class | Model | Use it for |
|-------|-------|------------|
| `pylcp_jax.obe` | Full optical Bloch equations (density matrix, coherences) | Accurate forces, dark states, coherent effects, narrow lines |
| `pylcp_jax.rateeq` | Rate equations (populations only) | Fast force profiles and trajectories when coherences are negligible |
| `pylcp_jax.heuristiceq` | Heuristic two-level scattering force | Quick estimates, large parameter scans |

All three offer `generate_force_profile`, `find_equilibrium_force` and
`evolve_motion`; `obe` and `rateeq` also have `evolve_density` / `evolve_populations`
for the internal state at fixed position and velocity.

## Examples and documentation

`docs/examples/` holds Jupyter notebooks, in increasing order of complexity:

| Folder | Topics |
|--------|--------|
| `basics/` | Power broadening, Rabi flopping, optical pumping, STIRAP |
| `molasses/` | 1D molasses (two-level, F=0->1, F=2->3, generic), Lambda-enhanced cooling |
| `MOTs/` | 1D/3D MOT forces and capture, temperature, real atoms, two-colour and recoil-limited MOTs, CaF MOT |
| `bichromatic/` | Bichromatic forces |

Build the Sphinx documentation (theory, API reference, notebooks) with
`uv run --group docs make -C docs html`; the output is in `docs/_build/html`.

## Benchmarks

`benchmarks/cpu_vs_gpu/` compares serial and multi-core CPU against batched
GPU `evolve_motion` on four hyperfine transitions (state dimension 22 to 150)
at three integration horizons. Run `benchmark_cpu.sh` and `benchmark_gpu.sh`
(SLURM scripts: edit the resources for your cluster), then `analyze.py` to
regenerate the plots and `run/summary.txt`.

At the longest horizon (t = 2π×2000), where JIT and setup costs are amortised,
the batched GPU path is roughly 11x to 35x faster per atom than the best
parallel CPU configuration (up to about 49x at short horizons, where the
comparison is less representative). The advantage shrinks as the state
dimension grows, and CPU scaling plateaus well below its Amdahl ceiling.

<p align="center"><img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/benchmarks/cpu_vs_gpu/run/gpu_transitions.png" width="560" alt="GPU time per atom versus batch size for four transitions"></p>

## Simulations: Sr-88 cooling cascade

`simulations/` is a four-stage Monte Carlo cooling sequence for <sup>88</sup>Sr
with 65 536 atoms per stage. Each stage starts from the captured atoms of the
previous one and includes gravity along -z:

| Stage | Directory | Final mean temperature |
|-------|-----------|------------------------|
| Blue MOT (461 nm), atoms from a Zeeman-slower beam | `blue_mot/` | about 1430 µK |
| Low-power blue MOT (optional) | `low_power_blue_mot/` | about 1290 µK |
| Broadband red MOT (689 nm, chirped) | `bb_red_mot/` | about 7.6 µK (trapped atoms) |
| Single-frequency red MOT (linear power ramp) | `sf_red_mot/` | about 4.5 µK |

<p align="center"><img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/cooling_sequence_temperature.png" width="640" alt="Temperature versus time across the cooling sequence"></p>

The simulated temperatures are close to experimental values for similar
parameters and reproduce the qualitative behaviour (cloud shapes, velocity
distributions, gravitational sag in the red MOT). Offsets are consistent with
the idealisations of the model; the single-frequency stage is warmer than the
2-3 µK reported experimentally.

### Running it

Each stage directory has `<stage>_sim.py`, `constants.py` (every experimental
parameter lives here; edit it to match your setup, including `MAX_ATOMS`),
`plot.py` and a SLURM script. Run the stages in order, from inside their
directory:

```
cd simulations/blue_mot
mkdir -p logs && sbatch blue_mot_sim.sh       # writes blue_mot_final_state.pkl
python plot.py                                # writes the figures shown below

cd ../bb_red_mot
mkdir -p logs && sbatch bb_red_mot_sim.sh     # reads ../blue_mot/blue_mot_final_state.pkl
```

Without SLURM, run the script directly, e.g.
`uv run python bb_red_mot_sim.py --upstream ../blue_mot/blue_mot_final_state.pkl`.
Point a stage at a different input with `UPSTREAM=path/to/state.pkl sbatch ...`.
To include the low-power stage, set `UPSTREAM` for `bb_red_mot` to
`../low_power_blue_mot/low_power_blue_mot_final_state.pkl`. Edit the `#SBATCH`
lines (partition, memory, walltime) for your cluster. The shared helpers
`analysis.py` and `plotting.py` sit in `simulations/`.

### Results

In every cloud plot below, atoms are histogrammed in position (log colour
scale) and the cyan marker is the trap centre.

<details>
<summary><b>Blue MOT</b></summary>


Atoms arrive from the Zeeman-slower beam and are captured into a roughly
spherical cloud of radius about 0.5 mm, at about 1.4 mK (twice the Doppler limit).

| Cloud, xy plane | Cloud, xz plane |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/blue_mot/blue_mot_cloud_2d_xy.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/blue_mot/blue_mot_cloud_2d_xz.png" width="360"> |

| Position and velocity distributions | Temperature vs time |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/blue_mot/blue_mot_distributions.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/blue_mot/blue_mot_temperature.png" width="360"> |

<p align="center"><img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/blue_mot/blue_mot_3x2_trajectories.png" width="640" alt="Blue MOT example trajectories"></p>

</details>

<details>
<summary><b>Low-power blue MOT</b></summary>


Lower saturation gives a slightly colder cloud (about 1.3 mK), with the z axis
hotter than x and y.

| Cloud, xy plane | Cloud, xz plane |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/low_power_blue_mot/low_power_blue_mot_cloud_2d_xy.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/low_power_blue_mot/low_power_blue_mot_cloud_2d_xz.png" width="360"> |

| Position and velocity distributions | Temperature vs time |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/low_power_blue_mot/low_power_blue_mot_distributions.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/low_power_blue_mot/low_power_blue_mot_temperature.png" width="360"> |

<p align="center"><img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/low_power_blue_mot/low_power_blue_mot_3x2_trajectories.png" width="640" alt="Low-power blue MOT example trajectories"></p>

</details>

<details>
<summary><b>Broadband red MOT</b></summary>


The 689 nm line is narrow, so the laser frequency is chirped over a 3 MHz window
at 50 kHz to capture the hot atoms leaving the blue MOT. Plots are zoomed on the
captured cloud; the temperature is for the trapped subset.

| Cloud, xy plane (zoom) | Cloud, xz plane (zoom) |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/bb_red_mot/bb_red_mot_cloud_2d_xy_zoom.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/bb_red_mot/bb_red_mot_cloud_2d_xz_zoom.png" width="360"> |

| Cloud, xy plane (full) | Cloud, xz plane (full) |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/bb_red_mot/bb_red_mot_cloud_2d_xy.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/bb_red_mot/bb_red_mot_cloud_2d_xz.png" width="360"> |

| Distributions (all atoms) | Distributions (trapped atoms) |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/bb_red_mot/bb_red_mot_distributions.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/bb_red_mot/bb_red_mot_distributions_trapped.png" width="360"> |

| Temperature vs time | Example trajectories |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/bb_red_mot/bb_red_mot_temperature.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/bb_red_mot/bb_red_mot_3x2_trajectories.png" width="360"> |

</details>

<details>
<summary><b>Single-frequency red MOT</b></summary>


A single-frequency beam with a linear power ramp (saturation 1270 down to 127)
compresses and cools the cloud to about 4.5 µK. The cloud sags below the trap
centre under gravity.

| Cloud, xy plane (zoom) | Cloud, xz plane (zoom) |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/sf_red_mot/sf_red_mot_cloud_2d_xy_zoom.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/sf_red_mot/sf_red_mot_cloud_2d_xz_zoom.png" width="360"> |

| Cloud, xy plane (full) | Cloud, xz plane (full) |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/sf_red_mot/sf_red_mot_cloud_2d_xy.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/sf_red_mot/sf_red_mot_cloud_2d_xz.png" width="360"> |

| Position and velocity distributions | Temperature vs time |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/sf_red_mot/sf_red_mot_distributions.png" width="360"> | <img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/sf_red_mot/sf_red_mot_temperature.png" width="360"> |

<p align="center"><img src="https://raw.githubusercontent.com/palmispojk/pylcp-jax/master/simulations/images/sf_red_mot/sf_red_mot_3x2_trajectories.png" width="640" alt="Single-frequency red MOT example trajectories"></p>

</details>

## Repository layout

```
pylcp_jax/      the package (obe, rateeq, heuristiceq, fields, hamiltonians, atom, GPU integrators)
tests/          pytest suite
docs/           Sphinx docs and example notebooks
benchmarks/     CPU-vs-GPU benchmark, memory and batch-scaling studies
simulations/    Sr-88 MOT cascade (SLURM scripts, constants, plotting, result images)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and
linting/formatting instructions. Run the tests with `uv run pytest`.
