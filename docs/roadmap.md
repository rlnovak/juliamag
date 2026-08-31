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
6. **High-level result-logging API in the problem script**, mumax3-style. Every
   custom script currently hand-writes its own `logrow!`/table code (open a file,
   format columns, average, track the core, snapshot). Provide the mumax3 verbs
   directly on the `Simulation`:
   - `save(sim, t)` / `save_ovf(sim; prefix)` — write one OVF snapshot of `m` now.
   - `table_add!(sim, quantity)` and `table_autosave!(sim, interval)` — declare the
     table columns once (mx/my/mz, maxTorque, energies, `ext_corepos`, …) and let the
     run loop append a row every `interval`, mirroring mumax3's `TableAdd` +
     `tableautosave`. Subsumes the OVF-autosave hook in item 3.
   The averaging quantity **must default to `average_region` over the filled geometry**,
   not `sum / (Nx·Ny·Nz)` over the whole box — see the normalization bug below.

Items 4 and 5 are trivial and should land first (a small `runinfo`/logging helper
the drivers call); 1–3 and 6 are the substantial driver refactor (6 is the
table/OVF-autosave layer that removes the per-script boilerplate).

### Validation follow-up: single-disk STT vortex frequency

From comparing the labmac04 single-disk Slonczewski run against the mumax3
reference (`disk_d1=128nm_t=16nm_40mA`), two separate discrepancies were found —
one a script bug (fixed), one a physics item still open:

- **Table normalization (fixed in the driver scripts).** The scripts averaged `⟨m⟩`
  by dividing by `Nx·Ny·Nz` (the whole box) instead of the filled disk cells, diluting
  every component by the disk area fraction (π/4 ≈ 0.785). This alone made the JuliaMag
  curves ~0.76× the mumax ones. Fixed by switching to `average_region(m, rp, 1)`, the
  analogue of mumax3's `TableAdd(m)`. The high-level table API (item 6) must default to
  this so future scripts don't repeat the mistake.
- **Gyrotropic frequency +5.4% vs mumax3 — ROOT-CAUSED and FIXED (exchange edge-fill).**
  The vortex gyration was 975 MHz in JuliaMag vs 925 MHz in mumax3 — an identical ratio
  (1.0541) across mx/my/mz, i.e. a pure frequency offset that accumulated a growing phase
  lag over 40 ns. Ruled out the Slonczewski torque (`slonczewskitorque!` is line-for-line
  identical to `cuda/slonczewski2.cu`, parameters match the mumax3 `log.txt`). Traced to
  the **exchange field's handling of the geometry fill**, by reading the mumax3 source
  (cloned at `../mumax3`). mumax3's rule: the cell fill (`vol`) scales the magnetization
  **only where it is a source of the demag field** (`engine/demag.go` `SetMFull`:
  `M = m·Msat·vol`); the **exchange** divides by the region's **full** Msat
  (`cuda/amul.h` `inv_Msat` reads the Msat LUT, never `vol`) and treats an empty neighbour
  as a free (Neumann) boundary (`cuda/exchange.cu`: `m_ = is0(m_) ? m0 : m_`). JuliaMag had
  two bugs here, on **both CPU and GPU**:
  1. the exchange prefactor divided by `Msat·fill`, inflating `B_exch` by `1/fill` at
     partially-filled boundary cells (stiffening the edge);
  2. an empty neighbour was read as `m = 0` (not Neumann), so a nonzero background-region
     `Aex` leaked a spurious `−Aᶜ·mᶜ/Δ²` term at the geometry edge.
  Fix (`src/exchange.jl`, `ext/JuliaMagCUDAExt.jl`, with `msat_region`/`fill` accessors in
  `src/region_params.jl` + `src/params.jl`): divide by the full region Msat, and null both
  the stiffness and the difference for an empty neighbour. Demag and the energies were
  already correct (they use `msat = Msat·fill`) and were left untouched. Verified on a
  small-disk gyration probe: **1000 → 875 MHz** (−12.5%, larger than the 128 nm disk's
  −5.4% because the smaller disk has a larger edge fraction — the right direction and
  magnitude). Regression tests added in `test/test_region_params.jl`. Still to do: re-run
  the 128 nm disk on labmac04 (GPU) to confirm the frequency lands on mumax3's 925 MHz.

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
