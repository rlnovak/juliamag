# JuliaMag roadmap

Where JuliaMag stands and where it can go next, including a comparison with
[MicroMagnetic.jl](https://github.com/MagneticSimulation/MicroMagnetic.jl), the
other Julia micromagnetic simulator.

## Current state (2026-08)

Validated CPU + GPU micromagnetic solver:

- **Physics:** LLG dynamics; exchange, demag (FFT/CUFFT), uniaxial anisotropy,
  interfacial + bulk DMI, Zeeman; Zhang–Li and Slonczewski spin-transfer torques;
  finite-temperature (Langevin) dynamics.
- **Geometry:** shapes + boolean composition + transforms; edge smoothing
  (fractional fill); polycrystalline Voronoi grains; multi-region materials.
- **Solvers:** energy minimizer (Barzilai–Borwein/Cayley), adaptive time
  integration (Tsit5), fixed-step RK/Heun.
- **GPU:** the whole pipeline runs on CUDA via a package extension (dispatch on
  `CuArray`), validated on hardware (labmac04, Quadro P1000).
- **Validation:** µMAG standard problems 2, 4, 5 vs mumax3 and OOMMF; a second
  round adapted from magnum.np demos.
- **Tooling:** data tables, OVF I/O, feature trackers, material library, a
  `Simulation` wrapper, a QML desktop GUI, a Colab benchmark notebook, a paper.

## Comparison with MicroMagnetic.jl

MicroMagnetic.jl (MagneticSimulation group) is a mature, feature-broad Julia
micromagnetic + atomistic simulator. What it has that JuliaMag does not:

| Capability | MicroMagnetic.jl | JuliaMag |
|------------|------------------|----------|
| **Multi-backend GPU** (CUDA, AMD, Intel, Apple) via **KernelAbstractions.jl** | ✅ | CUDA only (hand-written array-programming kernels) |
| **Atomistic spin dynamics** (discrete spins, exchange/DMI/anisotropy on a lattice) | ✅ | micromagnetic (continuum) only |
| **Monte Carlo** (finite-T equilibrium sampling) | ✅ | LLG-Langevin only |
| **NEB / GNEB** (minimum-energy paths, energy barriers, saddle search) | ✅ | ✗ |
| **Eigenmodes** (linearized-LLG normal modes, FMR spectra) | ✅ | ✗ |
| **Transition tooling** (Hessian, minimum-mode, symmetry) | ✅ | ✗ |
| Several Cayley/GPSM integrators | ✅ | Tsit5 + RK4 + BB minimizer |
| Web-based live GUI | ✅ | QML desktop GUI |

Where JuliaMag holds its own or differs in emphasis:

- **Transparency / readability as the primary goal** — each field/torque/solver
  is written directly at the level of the mathematics, with a paper documenting
  it; the target is a reference implementation, not maximum feature coverage.
- **Documented three-way validation** (JuliaMag vs mumax3 vs OOMMF) on the µMAG
  standard problems, with the data tables and figures in the repo, plus a
  hardware GPU-validation record (`gpu_validation/`).
- Both share the same core idea — one generic implementation dispatched to
  CPU/GPU by array type — so the architectures are compatible in spirit.

## What JuliaMag can learn from MicroMagnetic.jl

Ordered roughly by value-to-effort:

1. **KernelAbstractions.jl for multi-backend GPU.** The single highest-value
   change: rewrite the GPU kernels as `@kernel` functions so the same code runs
   on CUDA, AMD (AMDGPU), Intel (oneAPI), and Apple (Metal) — one kernel set
   instead of the current CUDA-only array-programming path. This also tends to be
   faster than fused broadcasts (explicit kernels, fewer launches) and would
   remove the per-term launch overhead we saw limiting the P1000 speedup. Our
   extension structure already matches theirs (weakdeps + per-backend Ext), so
   the migration is contained.
2. **NEB / GNEB** for energy barriers and switching paths — a frequently needed
   capability (thermal stability of bits/skyrmions) that we flagged as future
   work in the stage-2 list. `src/neb/` and `src/transition/` are the port
   references.
3. **Eigenmode / FMR solver** — linearize the LLG about an equilibrium and solve
   for normal modes; covers the FMR/dispersion stage-2 demos we deferred.
   `src/eigen/` is the reference.
4. **Atomistic spin dynamics** — a separate model (discrete spins) for problems
   below the micromagnetic length scale; large addition, clear module boundary
   (`src/atomistic/`).
5. **Monte Carlo** for finite-T equilibrium — complements the LLG-Langevin
   dynamics with faster equilibrium sampling (`src/mc/`).

## Driver ergonomics & run tooling (from real runs on labmac04)

Requested after running the single-disk STT problem on labmac04, where the
hand-written fixed-step RK4 driver made a 40 ns run very slow. High value,
low-to-medium effort — these make the package usable like mumax3 for production runs.

1. **Adaptive time stepping with spin-transfer torque (and thermal) in the built-in
   integrator.** The adaptive `Integrator` (Tsit5) already exists, but STT and the
   thermal field are only wired into fixed-step drivers (`runcurrent!`,
   `runthermal!`, and the hand-written scripts). Extend the `World`/RHS so a
   current (Zhang-Li / Slonczewski) — and, where meaningful, temperature — are part
   of the effective RHS the adaptive integrator sees, so `run!` picks the step
   automatically instead of a tiny fixed `dt`. (A stochastic thermal run still needs
   a fixed step or an SDE solver; adaptive applies to the deterministic STT case.)
2. **Solver selection by name**, mumax3-style. Let a script say
   `run!(sim, T; solver = :tsit5 | :rk4 | :heun | :euler)` (and set a current/temperature
   on the `Simulation`) instead of defining `rhs!`/`rk4!` by hand. The driver builds
   the RHS from the world's active terms + current + temperature and dispatches to
   the chosen integrator. Removes the boilerplate that every custom script now repeats.
3. **OVF autosave during a run** — `autosave_ovf!(sim, interval; prefix)` (mumax3's
   `autosave(m, dt)`), writing `prefix_000123.ovf` snapshots on a schedule. The
   writer (`saveovf`) exists; this is the scheduling hook in the run loop.
4. **Wall-clock start/end time in the log** — print the system time at the start and
   end of a run (and the elapsed) so long runs are timestamped.
5. **Log the device** — print whether the run is on CPU or GPU and, if GPU, which
   device (name + compute capability), at the start of every run.

Items 4 and 5 are trivial and should land first (a small `runinfo`/logging helper
the drivers call); 1–3 are the substantial driver refactor.

## Feature backlog (also from the stage-2 magnum.np review)

- Spin-orbit torque with separate damping-like / field-like terms (η_damp/η_field).
- RKKY interlayer exchange coupling.
- Chirality-dependent DMI boundary condition (for interfacial-DMI edge canting).
- Magnetoelastic coupling; eddy currents / self-induction.
- 3D Voronoi grains + grain-boundary exchange scaling (phase-3 follow-ups).
- Energy functions and `clearempty!` on the GPU (currently CPU-only readback).

## Near-term priorities

The **driver ergonomics group above** (adaptive stepping with STT, named solvers,
OVF autosave, run logging) is now the top near-term priority — it came directly
from a real production run and removes the biggest friction in using JuliaMag for
actual experiments. After that, the **KernelAbstractions migration** (multi-backend
GPU) and **NEB** most broaden JuliaMag's reach for the least architectural
disruption; both have a clean reference in MicroMagnetic.jl.
