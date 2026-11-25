# JuliaMag roadmap

Where JuliaMag stands and where it can go next, including a comparison with
[MicroMagnetic.jl](https://github.com/MagneticSimulation/MicroMagnetic.jl), the
other Julia micromagnetic simulator.

## Current state (2026-11)

Validated CPU + GPU micromagnetic solver:

- **Physics:** LLG dynamics; exchange, demag (FFT/CUFFT), uniaxial anisotropy,
  interfacial + bulk DMI, Zeeman; Zhang–Li and Slonczewski spin-transfer torques;
  finite-temperature (Langevin) dynamics.
- **Geometry:** shapes + boolean composition + transforms; edge smoothing
  (fractional fill); polycrystalline Voronoi grains; multi-region materials.
- **Solvers:** energy minimizer (Barzilai–Borwein/Cayley); mumax3's explicit
  solvers step for step (Euler, Heun, RK23, RK4, Dormand–Prince RK45 with
  MaxErr/FixDt/Headroom) with spin currents and temperature in the right-hand
  side; OrdinaryDiffEq's Tsit5.
- **GPU:** the whole pipeline runs on CUDA via a package extension (dispatch on
  `CuArray`), one mumax3-style kernel per term, Float64 and Float32, validated on
  hardware (labmac03/04, Quadro P1000); a whole `Simulation` moves with `togpu`.
- **Validation:** µMAG standard problems 2, 4, 5 vs mumax3 and OOMMF; a
  current-driven vortex in a disc vs mumax3 (fields, 7 ns and 40 ns dynamics); a
  second round adapted from magnum.np demos. All re-run after the November 2026
  fixes and GPU rewrite (`gpu_validation/revalidation/`).
- **Tooling:** mumax3-style driver layer (`tableadd!`/`tableautosave!`,
  `autosave!`/`save!` of OVF, output directory, run log with wall-clock times and
  device), feature trackers, material library, a QML desktop GUI, a Colab
  benchmark notebook, a paper.

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
   *Status (2026-11-19):* the CUDA methods are now explicit per-term kernels
   (`ext/cuda_kernels.jl`, one thread per cell, mumax3-style), so the launch
   overhead is gone (14 launches, 0 allocations per RHS; Float32 RHS = mumax3's
   1.11 ms on the P1000) and the KernelAbstractions port is a mechanical
   translation of those kernels.
   *Status (2026-11-23):* deferred to the future features below: only NVIDIA
   hardware is available to test it.
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
   The averaging quantity **must default to `average_magnet`** (fill-weighted over the
   magnet, mumax3's table `m`; `q_m` already uses it), not `sum / (Nx·Ny·Nz)` over the
   whole box — see the normalization bug below. `q_maxtorque` still reports
   max |dm/dt| in rad/s from the LLG torque only; to match mumax3's `maxTorque` it must
   be in T (÷ γLL) and include the active spin torques, as the 40 ns driver does by hand.

**Status (2026-11-20): all six items done.** `src/timestepper.jl` implements
mumax3's Euler, Heun, RK23, RK4 and Dormand–Prince RK45 operation for operation
(stage normalization, error norm, `adaptDt` with Headroom, MinDt/MaxDt, FixDt,
the end-of-run alarm), on the CPU and the GPU; the spin current lives in the
`World` (`setcurrent!`, `SpinCurrent`), so `llgrhs!` — and with it every solver,
adaptive or not, and the `maxTorque` column — includes the Zhang-Li / Slonczewski
torques; `settemperature!` adds the thermal field to the fixed-step solvers;
`setsolver!` / `run!(…; solver)` pick the solver by name (`:tsit5` kept); the table
and OVF autosaves follow mumax3's `needSave` rule, stream to `outdir`, and
`q_maxtorque` reports tesla including the spin torques; every `run!` logs its
wall-clock start/end, device and work counters; `togpu(sim)` moves a whole
simulation to the GPU. Tests: `test/test_timestepper.jl` (orders 1–5, analytic
damped precession, adaptDt, STT via `run!` = hand-written RK4 loops, schedules,
OVF, log) and `gpu_validation/test_gpu.jl` (`stepper_terms`). The 40 ns disc in
the new API: `gpu_validation/disk_stt/juliamag/disk_40mA_sim.jl`.

Items 4 and 5 are trivial and should land first (a small `runinfo`/logging helper
the drivers call); 1–3 and 6 are the substantial driver refactor (6 is the
table/OVF-autosave layer that removes the per-script boilerplate).

### Validation follow-up: single-disk STT vortex frequency

From comparing the labmac04 single-disk Slonczewski run against the mumax3
reference (`disk_d1=128nm_t=16nm_40mA`), several discrepancies were found and are
all resolved (see **Resolution** at the end of this section; the 975/925 MHz figures
below are the historical, pre-resolution measurements):

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
  the stiffness and the difference for an empty neighbour. The demag *field* was already
  correct (it uses `msat = Msat·fill`). Verified on a small-disk gyration probe:
  **1000 → 875 MHz** (−12.5%, larger than the 128 nm disk's −5.4% because the smaller
  disk has a larger edge fraction — the right direction and magnitude). Regression tests
  added in `test/test_region_params.jl`. The GPU exchange carried a second, cancelling
  bug (empty neighbour masked on `fill == 0`, but background cells keep fill 1 with
  region Msat 0); fixed in `0b9110d` by masking on `Msat·fill == 0`.
- **Demag energy ignored the cell fill — FIXED.** `demagenergy` took the field from the
  uniform-Msat path even for `RegionParams`, so on edge-smoothed geometry the source
  ignored the fill (E ∝ fill instead of fill²). It now uses the per-region path, i.e.
  mumax3's `GetDemagEnergy = −½·V·dot(M_full, B_demag)` with `M = m·Msat·vol`.
  Regression test "E ∝ fill²" in `test/test_energy.jl` (fails on the old code).

**Resolution (2026-11-12) — JuliaMag matches mumax3 like-for-like.** Field by field:
the relaxed mumax3 `m` loaded from OVF gives `B_demag` and `B_exch` equal to mumax3's
to a max relative error of 7.7e-7 and 2.3e-7 (EdgeSmooth 0 and 4; geometry/fill
identical cell by cell); all energies agree. The residual frequency gap (963.6 MHz vs
≈930 MHz) had **no JuliaMag bug left in it**; it came from two differences in the
*setup*:

1. **mumax3 adds the Zhang–Li torque whenever `J` is set** (`DisableZhangLiTorque =
   false` by default, `engine/torque.go`), so the reference ran Slonczewski + Zhang–Li
   (with `jz` it acts through ∂z m across the 8-cell thickness, −4.4% in frequency);
   the JuliaMag driver applied Slonczewski only.
2. **EdgeSmooth**: the reference used mumax3's default 0 (staircase); the port used 4
   (≈ −1%).

Like-for-like gyration frequencies (EdgeSmooth 0, 7 ns, ⟨m⟩-based Hilbert phase +
FFT on a uniformly resampled time axis):

| run | mumax3 | JuliaMag |
|---|---|---|
| Slonczewski + Zhang–Li | 930.8 MHz | 930.8 MHz |
| Slonczewski only | 973.5 MHz | 973.5 MHz |
| free gyration (no current) | 969.8 MHz, Γ = 0.0797 ns⁻¹ | 969.8 MHz, Γ = 0.0797 ns⁻¹ |

(f = mean of the ⟨m_x⟩ and ⟨m_y⟩ Hilbert-phase frequencies over 1–7 ns; free gyration
0.5–7 ns. The two codes differ by < 0.05 MHz; the ±1.5 MHz spread between the x and y
estimates is the absolute uncertainty of the measurement, common to both. Reproduce
with `gpu_validation/disk_stt/frequencies.py`; scripts and tables in
`gpu_validation/disk_stt/`.)

The integrator is converged (dt = 5e-14 vs 2.5e-14: 963.56 MHz both, max |Δ⟨m⟩| < 1e-6, the table print precision).
Earlier quoted mumax3 values of 913–925 MHz were a measurement artefact: an FFT that
assumed a uniform time step on mumax3's adaptive-step table (−1.6% bias). The two
vortex-core trackers differed by 2–4% (JuliaMag's `vortexcore` vs mumax3 `ext_corepos`),
so frequencies are measured from ⟨m⟩. Causes found (2026-11-13), both bugs in
`vortexcore`: (1) the sub-cell interpolation denominator was written `f1 - 2f0 + f2`,
and in Julia `2f0` is the Float32 literal 2.0f0, so it computed f1 − 2 + f2 and shrank
every sub-cell offset by ~10× (a core centred on a cell, as in the old tests, hides
it); (2) it searched only the bottom layer k = 1, while `ext_corepos` takes the max
|m_z| over every layer. Both fixed on CPU and GPU, with regression tests (exact
recovery of a parabolic peak; core only in the top layer) that fail on the old code.
On mumax3's own relaxed m the old tracker gave (−0.354, −0.354) nm vs mumax3's (0, 0).
Any earlier core-position figure is biased by up to half a cell. The 40 ns EdgeSmooth-4 series
checks the core positions with the aligned tracker. Also added `average_magnet`
(fill-weighted ⟨m⟩, mumax3's table `m`; `q_m` now uses it instead of the box
average, and it has a GPU method) — it differs from `average_region` only with
EdgeSmooth > 0. Plots: `examples/plot_stt_zhangli_validation.py`,
`examples/plot_free_gyration_compare.py`. Physical model for the paper: Slonczewski only
(the standard CPP vortex-oscillator model); Zhang–Li with a perpendicular current is
kept only as a code-to-code check of mumax3's default.

**40 ns EdgeSmooth-4 series (2026-11-13).** Slonczewski only, both codes on labmac04
(JuliaMag 7.2 h on a Quadro P1000, mumax3 34 min). Frequency 962.28 MHz in both
(every 2 ns window within 0.001 MHz), ringdown Γ = 0.0806 ns⁻¹ both, max |Δ⟨m⟩| 4e-5 in
the first 0.5 ns and < 1e-6 after 10 ns, E_total 2.5e-5, core positions 0.006 nm rms.
With p in-plane the gyration is damped (core settles at (−6.35, −1.06) nm), not an
auto-oscillation. mumax3's `maxTorque` at its default `MaxErr = 1e-5` plateaus near
13 mT after ~13 ns; proved to be RK45 tolerance noise in the stiff exchange modes (same
torques cell by cell on mumax3's m; restarts with `MaxErr = 1e-7` or fixed RK4 drop to the
JuliaMag value within 2 ps; a full `MaxErr = 1e-7` run agrees to 0.2% rms). Details:
`gpu_validation/disk_stt/README.md`; paper Fig. "disc40".

## Feature backlog (also from the stage-2 magnum.np review)

- Spin-orbit torque with separate damping-like / field-like terms (η_damp/η_field).
- RKKY interlayer exchange coupling.
- Magnetoelastic coupling; eddy currents / self-induction.
- 3D Voronoi grains + grain-boundary exchange scaling (phase-3 follow-ups).
- Energy functions and `clearempty!` on the GPU (currently CPU-only readback).

## Work queue (set 2026-11-21)

Runs first; they use the GPUs while the code work below proceeds.

**0a. GPU validation of the driver layer: DONE 2026-11-21**
(`gpu_validation/disk_stt/README.md`, tables in `gpu_validation/disk_stt/tables/item3/`).
- `test_gpu.jl` 542/542, including all five solvers GPU vs CPU.
- The 40 ns disc with adaptive RK45, MaxErr 1e-5:
  - Float32: 1 813 249 evaluations, 2 830 s.
  - Float64: 1 813 243 evaluations, 7 146 s.
  - mumax3: 1 815 341 evaluations, 2 025 s.
  - Physics as with the fixed step; mumax3's maxTorque plateau (12.882 mT) is now reproduced.
- Open: the 1.4× wall-time gap is stepper overhead, about 2 ms per step.
  - Cause: two allocating, host-synchronizing `maxtorque` reductions per step, plus the stage combinations.
  - Fix: fuse the reductions and skip them for fixed steps. This also speeds the thermal (Heun) runs.

**0b. Vortex hysteresis at finite temperature** (paper, Sec. "Vortex
hysteresis in a disc").
- **Protocol.** Same disc, field steps and sweep as the T = 0 loop, but at each
  field step the LLG–Langevin dynamics runs for a fixed time instead of the
  minimizer: `settemperature!`, Heun with a fixed step, on the GPU in Float32.
  - The damping is raised (α = 0.5, quasi-static protocol): α sets the
    relaxation rate, not the equilibrium, and the fluctuation–dissipation
    field uses the same α.
  - Dwell and step fixed by phase 1 (`gpu_validation/thermal_hysteresis/`):
    2 ns per field step (0.5 and 1 ns lag on the reversible branch and delay the
    annihilation); dt = 1e-13 s converged at 50 K and 300 K.
- **Ensemble.** T = 50 K and 300 K, N ≈ 8 realizations each, with seeds set
  through `CUDA.seed!` per realization.
- **Report.** The mean loop and the distribution of the renucleation and
  annihilation fields, next to the T = 0 deterministic and perturbed loops.
- **Caveats to state in the paper.**
  - The simulated sweep rate (~10⁷ T/s) is many orders above experiment, so the
    switching fields are upper bounds on thermally activated switching.
  - Msat and A are kept at their room-temperature values at 50 K.
- **Cost.** About 3.2 h of GPU time per loop, about 50 GPU-hours in all.
- **Status: DONE 2026-11-23** (`gpu_validation/thermal_hysteresis/README.md`).
  Paper: new paragraph and Fig. `vortex_hysteresis_thermal`.
  - Every finite-T switch goes through a vortex, mostly after a C-shaped
    precursor that the symmetric T = 0 run cannot reach.
  - Two vortex classes: annihilation at 60–65 mT or expulsion at 25–35 mT.
  - The 50 K and 300 K switching fields agree within the scatter.
  - Open (optional): re-run one loop of each class with OVF output to show the
    core positions.

**0c. Stepper overhead: DONE 2026-11-22** (`6c53fbb`).
- A device-side error accumulator: one host wait per adaptive step, none per fixed step.
- `lasterr`/`peakerr`/`lasttorque` exported.
- A fixed-step NaN now stops the run instead of retrying forever.
- CPU bit-identical to the previous stepper.
- GPU timing done 2026-11-23: overhead per step RK45 2.44 → 1.78 ms, RK4 0.82 → 0.37 ms, Heun 0.52 → 0.12 ms.
- `test_gpu` 625/630, with the 5 RK45 fixed-step error checks at the rounding floor of a cancelling estimator; the tolerance is fixed, and the re-run is queued.

**Gate.** Items 1–2 start only after 0b has finished and the paper (and the code,
if needed) has been updated with its results (decision of 2026-11-23).
KernelAbstractions was removed from this queue: see "Future features".

**1. NEB / GNEB: energy barriers and minimum-energy paths.** Detailed plan:
[`plan_neb_eigen.md`](plan_neb_eigen.md) §0–1.
- **Status, 2026-11-24.**
  - Step 0 (`energy!`, `isconservative`) and step 1a are DONE: the `src/neb.jl` core
    (Barzilai–Borwein plus Cayley optimizer, climbing image, Hermite saddle) is
    validated on Stoner–Wohlfarth to 6e-15.
  - Step 1b is DONE: ellipse, skyrmion and skyrmion-lattice barriers against
    MicroMagnetic.jl (`gpu_validation/gneb/results/README.md`).
  - GPU checks of the same code (d141053): 670/670.
  - Next: the paper section.
- **Prerequisite.** Energy functions on the GPU, from the feature backlog: the
  NEB evaluates every image's energy and gradient at every iteration.
- **Method.** Geodesic NEB on the product of unit spheres (Bessarab, Uzdin,
  Jónsson, Comput. Phys. Commun. 196, 335 (2015)):
  - geodesic distances and tangents;
  - spring forces;
  - a climbing image;
  - images relaxed with velocity-projection optimization or the existing
    Barzilai–Borwein minimizer;
  - an initial path by geodesic interpolation.
  - Reference implementation: MicroMagnetic.jl `src/neb/`.
- **Validation.**
  - Stoner–Wohlfarth macrospin: ΔE = K V (1 − h)² for a field along the easy
    axis.
  - The magnum.np `energy_barrier_ellipse` demo.
  - A skyrmion-collapse barrier against published values.

**2. Eigenmodes / FMR.** Detailed plan: [`plan_neb_eigen.md`](plan_neb_eigen.md) §2.
- **Method.** Linearize the LLG about an equilibrium in the local tangent frame
  (two components per cell):
  - Small systems: a dense eigenproblem.
  - Large systems: matrix-free Krylov with shift-invert near the frequencies of
    interest; the demag part of the Hessian applied through the existing FFT
    convolution.
  - Reference: MicroMagnetic.jl `src/eigen/`.
- **Validation.**
  - The Kittel formula for a macrospin and a thin film.
  - The FMR standard problem of Baker et al. (J. Magn. Magn. Mater. 421, 428
    (2017)), against a ringdown FFT done with the time integrators.
  - The disc's gyrotropic mode against the 969.8 MHz free gyration already
    validated against mumax3.

Each item: implementation, tests (CPU, then GPU), validation against an analytic
result and a reference code, paper section, commit. Ask before starting each.

**3. A mumax3-style input file.** After items 1–2.
- **Goal.** A simulation is described as simply as in mumax3: a short script of
  plain statements, with no Julia boilerplate (no `sim` argument threaded through
  every call, no array types, no explicit `togpu`). For example:

      gridsize(128, 128, 1); cellsize(4e-9, 4e-9, 4e-9)
      geometry(Ellipse(500e-9, 250e-9))
      Msat = 800e3; Aex = 13e-12; alpha = 0.02
      m = Uniform(1, 1, 0)
      B_ext = (0, 0, 0.01)
      tableadd(E_total); tableautosave(10e-12)
      relax(); run(5e-9)

- **To decide in the design.**
  - Global-state script mode: a `JuliaMag.Script` module with a current
    Simulation, run by `juliamag file.jl`. Or a parsed `.jmag` file translated to
    the existing API.
  - GPU auto-selection.
  - Material assignment by region.
  - Error messages pointing at the input line.
- **Constraint.** It must be a thin layer over the existing driver API: no physics
  in it.
- **Validation.** The µMAG std 4 and the disc examples rewritten in the new form
  reproduce their tables bit for bit.

**4. GUI review** (`gui/`, the QML desktop GUI, left aside since 2026-11-13).
- **Requirements (2026-11-23).** Fast and responsive, and it shows the essentials
  of the running simulation:
  - live ⟨m⟩(t) and energy;
  - a magnetization slice view (colour wheel / mz);
  - time, step, dt, maxTorque, wall time and device;
  - start, pause and stop, and loading a script (item 3).
- **First, an audit.**
  - Threading: the solver must not block the UI.
  - Update rate and data copied per frame: GPU→host only for the displayed slice.
  - Plotting cost.
  - Dependency weight and startup time.
  - Correctness of the displayed quantities.
- **Then the fixes,** each measured: frames per second and the solver slowdown
  with the GUI attached.

**Stage closure.** After items 1–4, this stage of JuliaMag is closed: a
traditional micromagnetic code, implemented, validated and functional. Next come
the future directions below.

## Near-term priorities

The **driver ergonomics group above** is done (2026-11-20). Next, in order (see
"Work queue"):
1. finish the finite-temperature hysteresis and its paper section;
2. **GNEB** and **eigenmodes** ([`plan_neb_eigen.md`](plan_neb_eigen.md));
3. the mumax3-style input file;
4. the GUI review.

That closes the stage.

## Future features

- **KernelAbstractions.jl, multi-backend GPU.** Deferred 2026-11-23: only NVIDIA
  hardware is available to test it.
  - Plan (unchanged from the 2026-11-21 queue):
    - Rewrite the per-term kernels of `ext/cuda_kernels.jl` as `@kernel` functions.
    - Keep only the array type and the FFT plans in thin backend extensions:
      CUFFT, rocFFT, oneMKL; Metal has no FFT.
  - Staged replacement: the CUDA kernels stay until KA-on-CUDA reproduces their
    results to rounding and their cost (1.11 ms per Float32 evaluation on the
    P1000).
  - Taken up when AMD, Intel or Apple hardware is available.
- Atomistic spin dynamics and Monte Carlo (MicroMagnetic.jl comparison, items 4–5).
- The feature backlog above (SOT, RKKY, magnetoelastics).

## Future directions (after the stage closure)

Two projects take JuliaMag further. Each has a PRD in [`docs/prd/`](prd/), meant to be
run in its own Claude Code session.

- **A. Differentiable JuliaMag (fork).**
  - A fork of this code base whose field and energy computations are rewritten
    for automatic differentiation, so that gradients of any simulation output
    with respect to parameters, geometry or the initial state are available.
  - Purpose: research at the interface of micromagnetics and neural networks.
  - This repository stays the traditional, validated reference code; the fork is
    the modern, differentiable one, validated against it.
  - PRD: [`prd/PRD-A-differentiable-juliamag.md`](prd/PRD-A-differentiable-juliamag.md).
- **B. Neural-network project integrated in both versions.**
  - It is meant as the crowning piece of JuliaMag, with the networks developed and
    trained in Julia.
  - PRD: [`prd/PRD-B-neural-project.md`](prd/PRD-B-neural-project.md).
