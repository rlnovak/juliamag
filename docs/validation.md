# JuliaMag — Validation tutorials

These tutorials repeat, step by step, the checks behind the claims of the [manual](manual.md) and of
the paper. Each one uses the scripts that produced the published numbers, as they are in the
repository, and lists what they should print.

Every tutorial has the same parts:
- **What it checks**: the physics and the reference (analytic, OOMMF, mumax3, MicroMagnetic.jl,
  magnum.np or an independent finite-element solution).
- **Files**: the JuliaMag driver, the reference input, and the comparison script.
- **Steps**: the commands, in order, from the root of the repository.
- **Expected result**: the numbers of the validation run.
- **Notes**: what to watch for.

| | problem | reference | hardware |
|---|---|---|---|
| [V1](#v1--standard-problem-2-remanence-and-switching) | standard problem 2 | OOMMF | CPU |
| [V2](#v2--standard-problem-4-every-component-and-energy) | standard problem 4 | mumax3, OOMMF | CPU, GPU |
| [V3](#v3--standard-problem-5-zhangli-torque) | standard problem 5 (Zhang–Li) | mumax3 | CPU |
| [V4](#v4--a-vortex-disc-driven-by-a-spin-current) | vortex disc with spin torques | mumax3 | GPU |
| [V5](#v5--dmi-at-free-edges-and-between-regions-anisotropy-in-partial-cells) | DMI at edges and between regions, anisotropy | mumax3 | CPU (+ mumax3 on a GPU) |
| [V6](#v6--fields-and-currents-that-vary-in-space-and-time) | fields and currents in space and time | mumax3 | GPU |
| [V7](#v7--currents-from-anisotropic-magnetoresistance) | AMR conduction | analytic, finite elements | CPU |
| [V8](#v8--magnumnp-problems-depinning-helix-langevin) | depinning, helix, Langevin ensemble | analytic (magnum.np demos) | CPU |
| [V9](#v9--finite-temperature-nanodot-and-vortex-hysteresis) | thermal nanodot, thermal hysteresis | Boltzmann, T = 0 loop | CPU, GPU |
| [V10](#v10--energy-barriers-gneb-and-saddle-points) | GNEB barriers, saddle order | analytic, MicroMagnetic.jl, magnum.np | CPU |
| [V11](#v11--linear-eigenmodes) | linear eigenmodes | FMR standard problem, MicroMagnetic.jl, time domain | CPU, GPU |
| [V12](#v12--the-gpu-against-the-cpu) | every GPU method | the CPU code | GPU |

**Before you start.**
- The scripts in `examples/` run in the `examples` environment
  (`julia --project=examples -e 'using Pkg; Pkg.instantiate()'` once). The scripts in
  `gpu_validation/` run in the repository environment (`--project=.`), plus CUDA.jl where a GPU is
  used: an environment with JuliaMag and CUDA (`Pkg.add("CUDA")`).
- The mumax3 runs used mumax3 3.11.1 (CUDA 12 build), `mumax3 -gpu 0 -http "" script.mx3`.
- The comparison scripts are Python (numpy, matplotlib; scipy for V7).
- The validation runs were made on a Quadro P1000 (Pascal) and an Intel Core Ultra 5 125H. The
  times given below are those runs: expect less on a newer GPU.
- Long GPU jobs on a shared machine were queued behind other users with
  `gpu_validation/run_when_gpu_idle.sh`.
- Every tutorial writes its own output. The tables already in the repository are those of the
  validation, so a comparison script can be run on them without running the simulation first.

---

## V1 — Standard problem 2: remanence and switching

**What it checks.** µMAG standard problem 2: a thin prism with sides 5 : 1 : 0.1, of size d measured
in exchange lengths ℓex, saturated along [1,1,1]. Demagnetizing field and exchange together,
against the OOMMF table: the remanence over d/ℓex = 0.5–30, the switching at seven sizes.

**Files.**
- `examples/stdproblem2.jl`: remanent ⟨mx⟩, ⟨my⟩ against d/ℓex (energy minimizer).
- `examples/stdproblem2_hysteresis.jl`: coercive field Hc and switching field Hs against d/ℓex.
- `examples/stdprob2_oommf.txt`: the OOMMF reference.
- `gpu_validation/revalidation/compare_reval.py`, `plot_reval.py`: differences and figures (matplotlib).

**Steps.**
1. Remanence: `julia --project=examples examples/stdproblem2.jl`. It writes `examples/stdproblem2.txt`.
2. Hysteresis: `julia --project=examples examples/stdproblem2_hysteresis.jl`. It writes
   `examples/stdproblem2_hysteresis.txt`. This one is slow: a full minimization at every field step
   for every size.
3. Compare with OOMMF: `python gpu_validation/revalidation/compare_reval.py std2 std2hyst`.
4. Figures: `python gpu_validation/revalidation/plot_reval.py` (it draws all the re-validation
   figures, among them `paper/figures/stdproblem2_*`).

**Expected result.**
- Remanence, d/ℓex from 0.5 to 30: |Δ⟨mx⟩|, |Δ⟨my⟩| ≤ 3×10⁻⁴ from OOMMF.
- Hc and Hs at seven sizes: ΔHc ≤ 7×10⁻⁴ Ms, ΔHs ≤ 6×10⁻⁴ Ms, with a field step of 10⁻³ Ms.

**Notes.** The field step limits how well Hc and Hs are resolved: the differences are of its size.

---

## V2 — Standard problem 4: every component and energy

**What it checks.** µMAG standard problem 4: the switching of a 500 × 125 × 3 nm Permalloy film
from the S-state, under field 1. Demagnetizing field, exchange, Zeeman energy and the LLG equation.
Not only ⟨m⟩: every energy term over the whole nanosecond, to find where a difference would start.

**Files.**
- `examples/stdproblem4.jl`: the problem; writes `examples/stdproblem4.txt` (t, ⟨m⟩).
- `examples/stdproblem4_mumax3.txt`, `examples/stdprob4a_oommf.odt`: mumax3 and OOMMF references.
- `gpu_validation/benchmarks/std4.mx3` (mumax3, with `TableAdd` of every energy) and
  `gpu_validation/benchmarks/std4_energies.jl` (the same columns from JuliaMag).
- `examples/plot_benchmark_compare.py`: column-by-column comparison and figure.
- `examples/stdproblem4_gpu.jl`: the whole problem on the GPU, against the CPU table.

**Steps.**
1. ⟨m⟩(t): `julia --project=examples examples/stdproblem4.jl`.
2. Against mumax3 and OOMMF: `python gpu_validation/revalidation/compare_reval.py std4`.
3. Every energy:
   1. `mumax3 gpu_validation/benchmarks/std4.mx3` (on a GPU machine); its table is
      `std4.out/table.txt`.
   2. `JM_OUT=out julia --project=. gpu_validation/benchmarks/std4_energies.jl` (`JM_OUT`: the
      output directory; default, next to the script).
   3. `python examples/plot_benchmark_compare.py std4 out/std4_juliamag_table.txt std4.out/table.txt std4_compare.png`.
      Use the tables in `gpu_validation/benchmarks/` to compare without running.
4. GPU (optional): `julia --project=<env with JuliaMag + CUDA> examples/stdproblem4_gpu.jl`.

**Expected result.**
- Relaxed S-state: ⟨m⟩ = (0.9668, 0.1256, 0.0000), as mumax3.
- ⟨m⟩(t) over 0–1 ns: max |Δ⟨m⟩| 0.0044 from mumax3, 0.0037 from OOMMF.
- rms difference from mumax3 over 1 ns, as a fraction of each column's rms:

  | mx | my | mz | E_demag | E_exch | E_Zeeman | E_total |
  |---|---|---|---|---|---|---|
  | 0.007 % | 0.028 % | 0.027 % | 0.018 % | 0.042 % | 0.007 % | 0.002 % |

- GPU against CPU: ~10⁻⁵ over the whole run (1.6×10⁻⁵ with Tsit5).

**Notes.** The film is a full rectangle: every cell is filled, so this problem does not test the
treatment of partially filled edge cells. V4 and V5 do.

---

## V3 — Standard problem 5: Zhang–Li torque

**What it checks.** µMAG standard problem 5 (Najafi et al., J. Appl. Phys. 105, 113914 (2009)): a
vortex in a 100 × 100 × 10 nm Permalloy square, pushed by an in-plane current, J = 10¹² A/m², P = 1,
ξ = 0.05. The Zhang–Li torque, with demag and exchange.

**Files.**
- `examples/stdproblem5.jl`: the problem.
- `gpu_validation/benchmarks/std5.mx3`, `gpu_validation/benchmarks/std5_energies.jl`: every column
  in both codes.
- `examples/plot_benchmark_compare.py`.

**Steps.**
1. `julia --project=examples examples/stdproblem5.jl`. It prints ⟨m⟩ at 1 ns.
2. Every column:
   1. `mumax3 gpu_validation/benchmarks/std5.mx3`.
   2. `JM_OUT=out julia --project=. gpu_validation/benchmarks/std5_energies.jl`.
   3. `python examples/plot_benchmark_compare.py std5 out/std5_juliamag_table.txt std5.out/table.txt std5_compare.png`.

**Expected result.**
- ⟨m⟩(1 ns) = (−0.23488, −0.09453, 0.02296); mumax3: (−0.23480, −0.09454, 0.02296).
- rms differences over 1 ns: ≤ 0.001 % in every component and energy.

---

## V4 — A vortex disc driven by a spin current

**What it checks.** A curved geometry with partially filled edge cells, the Slonczewski and Zhang–Li
torques and long-time dynamics, against mumax3 3.11.1 run for run.
- Permalloy disc, 128 nm × 16 nm, 64 × 64 × 8 cells of 2 nm; Ms = 800 kA/m, A = 10.8 pJ/m, α = 0.01.
- Vortex (circulation 1, polarity 1), current 40 mA perpendicular to the plane
  (jz = 3.108×10¹² A/m²); fixed layer p = (cos 20°, sin 20°, 0); P = 0.37, Λ = 1, ε′ = 0.01.
- Staircase edge (EdgeSmooth 0) and smoothed edge (EdgeSmooth 4).

**Files** (in `gpu_validation/disk_stt/`, described in its `README.md`).
- `mumax3/*.mx3`: every mumax3 run.
- `juliamag/fieldcmp.jl`: JuliaMag's fields on mumax3's relaxed state, cell by cell.
- `juliamag/gyro_lean.jl`: the 7 ns runs, `julia gyro_lean.jl ES DT TOTAL MODE OUT`, MODE `stt`
  (Slonczewski), `sttzl` (Slonczewski + Zhang–Li, mumax3's default) or `free` (no current).
- `juliamag/disk_40mA_sim.jl`: the 40 ns run in the `Simulation` API, line for line
  `mumax3/stt_es4_noZL_40ns.mx3`.
- `gyrofreq.py`, `frequencies.py`: frequencies from the tables; `examples/plot_disk40_compare.py`:
  the 40 ns comparison.
- `tables/`: all the time series of the validation.

**Steps.**
1. **Static fields.**
   1. `mumax3 mumax3/fields_es0.mx3` and `mumax3 mumax3/fields_es4.mx3`, in a directory DIR. They
      save the relaxed m, B_demag, B_exch, the geometry and the energies.
   2. `julia --project=. gpu_validation/disk_stt/juliamag/fieldcmp.jl DIR`.
2. **Dynamics, 7 ns.**
   1. mumax3: `stt_es0_reference.mx3` (Slonczewski + Zhang–Li), `stt_es0_noZL.mx3` (Slonczewski
      only), `free_es0.mx3` (free gyration: relaxed under Bx = 10 mT, field off).
   2. JuliaMag, the same three:
      `julia --project=<env with JuliaMag + CUDA> gpu_validation/disk_stt/juliamag/gyro_lean.jl 0 5e-14 7e-9 sttzl sttzl_es0.txt`,
      and the same with `stt` and `free`.
   3. Frequencies: `python gpu_validation/disk_stt/frequencies.py` (from the tables in `tables/`).
3. **40 ns, smoothed edge, Slonczewski only** (the physical model).
   1. `mumax3 mumax3/stt_es4_noZL_40ns.mx3`; also `stt_es4_noZL_40ns_maxerr1e-7.mx3` for the
      maxTorque note.
   2. `julia --project=<env with JuliaMag + CUDA> gpu_validation/disk_stt/juliamag/disk_40mA_sim.jl OUTDIR Float32 rk45 1e-5 40e-9`
      (arguments: output directory, precision, solver, MaxErr, total time).
   3. `python examples/plot_disk40_compare.py OUTDIR/<table> tables/mumax_stt_es4_noZL_40ns.txt`. The
      validation tables are `tables/item3/f32_rk45_40ns.txt` and
      `tables/mumax_stt_es4_noZL_40ns.txt`.

**Expected result.**
- Static fields: B_demag to 7.7×10⁻⁷ and B_exch to 2.3×10⁻⁷ relative; the same filling, cell by cell;
  EdgeSmooth 0: E_demag = 7.1783×10⁻¹⁹ J, E_exch = 3.6270×10⁻¹⁸ J, in both codes.
- 7 ns (EdgeSmooth 0):

  | run | mumax3 | JuliaMag |
  |---|---|---|
  | Slonczewski + Zhang–Li | 930.8 MHz | 930.8 MHz |
  | Slonczewski only | 973.5 MHz | 973.5 MHz |
  | free gyration | 969.8 MHz, Γ = 0.0797 ns⁻¹ | 969.8 MHz, Γ = 0.0797 ns⁻¹ |

  Max |Δ⟨m⟩| 4.0–8.9×10⁻⁵; vortex core within 0.003–0.006 nm rms of mumax3's `ext_corepos`.
- 40 ns:
  - 962.318 MHz in both codes; ≤ 0.001 MHz in every 2 ns window;
  - max |Δ⟨m⟩| 3.0×10⁻⁵ (Float32), E_total 1.7×10⁻⁵ relative, core ≤ 0.005 nm rms;
  - 302 206 accepted steps against mumax3's 302 912; wall time on the P1000: 2 830 s (Float32),
    7 146 s (Float64), mumax3 2 025 s.
  - Mean maxTorque over 39–40 ns: 12.882 mT in both codes with MaxErr 10⁻⁵; 1.684 mT
    (mumax3, MaxErr 10⁻⁷) against 1.682 mT (JuliaMag, fixed-step RK4).

**Notes.**
- mumax3's table `m` is the average weighted by the cell filling. Compare it with JuliaMag's
  `average_magnet`, not with the mean over the cells (they differ on a smoothed edge).
- mumax3's tables have adaptive time steps: resample them onto a uniform grid before an FFT
  (`gyrofreq.py` does).
- The maxTorque plateau with MaxErr 10⁻⁵ comes from the tolerance of the integrator, not from the
  physics: with MaxErr 10⁻⁷, or a fixed step, it decays.
- Neither code has the Oersted field here: the frequency validates the code, not a device.

---

## V5 — DMI at free edges and between regions; anisotropy in partial cells

**What it checks.** The exchange + DMI field and the anisotropy field, with their energies, against
mumax3 in 28 cases:
- open and periodic boxes, interfacial and bulk DMI, `OpenBC` false and true;
- discs and ellipsoids with staircase and smoothed edges (DMI and anisotropy in partial cells);
- two regions with different A and Ms; regions with D of the same sign, opposite sign or zero.

Each case evaluates the fields of the random state that mumax3 wrote, with no relaxation, so the
discretizations are compared directly.

**Files** (`gpu_validation/dmi_boundary/`).
- `make_mx3.py`: writes the 28 mumax3 scripts and, for each, the parameters of every region.
- `compare_mumax3.jl`: evaluates the same states in JuliaMag and prints the differences.
- `results.txt`: the output for the present code; `results_cf87368.txt`, `results_old_code.txt`: the
  same comparison with two older versions (negative controls).

**Steps.**
1. `python gpu_validation/dmi_boundary/make_mx3.py DIR` (a new directory).
2. In DIR, on a GPU machine: `mumax3 *.mx3`.
3. `julia --project=. gpu_validation/dmi_boundary/compare_mumax3.jl DIR`.

**Expected result.** Largest field difference over the largest field:

| cases | present code | before the region and partial-cell fixes |
|---|---|---|
| boxes, interfacial and bulk, both `OpenBC` | ≤ 1.5×10⁻⁷ | ≤ 1.5×10⁻⁷ |
| disc, smoothed edge: DMI in partial cells | ≤ 1.3×10⁻⁷ | ≤ 1.3×10⁻⁷ |
| regions of different D (10 cases) | ≤ 9.8×10⁻⁸ | 7.8×10⁻³–2.3×10⁻² at the interface |
| anisotropy, smoothed disc | ≤ 8.4×10⁻⁸ | 9.3 and 14 times the largest field |

Energies to 9.1×10⁻⁸. mumax3 computes in single precision: agreement is to about 10⁻⁷.

**Notes.** mumax3's default (`OpenBC = false`) applies the chiral boundary condition to both z faces
of a bulk-DMI film one cell thick. `openbc = true` is the rule of MicroMagnetic.jl.

---

## V6 — Fields and currents that vary in space and time

**What it checks.** `fieldmap`, `regionmap`, `addfield!` and `addcurrent!` (manual, section 4) against
mumax3's `B_ext.add(mask, f(t))`, `B_ext.SetRegion(r, v(t))` and `J.SetRegion`, on four problems:
1. `pulse`: a Gaussian pulse in space and time on a vortex, 2 ns;
2. `regions`: a field per region, each a function of time, on a film, 1 ns;
3. `zhangli_half`: a Zhang–Li current in one half of the standard problem 5 square, 1 ns;
4. `nanocontact`: a Slonczewski current through a nanocontact in a disc, 2 ns.

Both codes on a GPU, Float32, adaptive RK45, MaxErr 10⁻⁵.

**Files** (`gpu_validation/fields/`).
- `mumax3/<case>.mx3` and `fields_juliamag.jl CASE OUTDIR`: the four problems in each code.
- `run_fields.sh REPO OUT`: runs the eight jobs in sequence on one GPU.
- `compare.py RESULTS FIGURE`: interpolates JuliaMag's rows at mumax3's times and prints the
  differences; `results/compare.txt` and `compare.png` are those of the validation.

**Steps.**
1. Either `bash gpu_validation/fields/run_fields.sh $PWD OUT` (set the paths to mumax3 and Julia at
   the top of the script), or by hand, for each case:
   1. `mumax3 gpu_validation/fields/mumax3/<case>.mx3`, writing to `OUT/mumax3/<case>.out`;
   2. `julia --project=<env with JuliaMag + CUDA> gpu_validation/fields/fields_juliamag.jl <case> OUT/juliamag`.
2. `python gpu_validation/fields/compare.py OUT compare.png`.

**Expected result.** rms differences relative to the largest |value| of each column:

| problem | ⟨m⟩ | E_total |
|---|---|---|
| pulse | 3×10⁻⁴ | 2×10⁻⁶ |
| regions | 1–5×10⁻⁴ | 1.4×10⁻⁴ (from t > 0) |
| zhangli_half | 3–9×10⁻⁶ | 3×10⁻⁷ |
| nanocontact | 2×10⁻⁶–1×10⁻⁵ | 5×10⁻⁷ |

**Notes.**
- In `regions` the row at t = 0 differs: mumax3 still shows the field from before the assignment
  (it evaluates a function of time only once the time changes), JuliaMag the new one.
- The first two problems oscillate fast. Their differences come from rows at different adaptive
  times and from the single-precision relaxation of the start states, each code its own.

---

## V7 — Currents from anisotropic magnetoresistance

**What it checks.** The conduction solver (manual, section 4): ∇·(σ(m)∇φ) = 0 with
σ = (1/ρ⊥)[I − (Δρ/ρ∥) m mᵀ], solved by finite volumes.
1. Against an independent discretization: linear finite elements on triangles (scipy). The sample
   is a 400 × 80 × 4 nm rectangle with ρ∥ = 210 and ρ⊥ = 200 nΩ m, and contacts at 1 V and 0 V on
   the short sides. It is solved for two textures:
   - a domain wall at an angle, where σ_xy changes sign;
   - uniform m at 45°, which gives the largest planar Hall field, shorted at the contacts.
2. On the dynamics: a vortex in a disc between two leads, 2 mA, for Δρ/ρ⊥ = 0, 0.02 and 0.04. The
   effect of AMR must be linear in Δρ/ρ⊥ at small values.

**Files** (`gpu_validation/amr/`).
- `fem_reference.py OUT [wall|uniform45]`: the finite-element reference.
- `juliamag_texture.jl OUT [wall|uniform45]`: JuliaMag on 50 × 10 to 400 × 80 cells.
- `compare_texture.py DIR`: resistances, Richardson extrapolation, potentials.
- `vortex_amr.jl DRHO OUT`: the vortex between leads; `compare_vortex.py DIR OUT.png`.
- `paper_figure.py TEXDIR VORTEXDIR OUT.png`: the paper figure.
- `results/texture.txt`, `results/vortex.txt`: the validation output.

**Steps.**
1. Textures:
   1. `python gpu_validation/amr/fem_reference.py tex wall` and `... tex uniform45`;
   2. `julia --project=. gpu_validation/amr/juliamag_texture.jl tex wall` and `... tex uniform45`;
   3. `python gpu_validation/amr/compare_texture.py tex`.
2. Vortex (5–8 min per run on one CPU thread):
   1. `julia --project=. gpu_validation/amr/vortex_amr.jl 0.0 vortex`, then the same with `0.02`
      and `0.04`;
   2. `python gpu_validation/amr/compare_vortex.py vortex vortex.png`.
3. Figure: `python gpu_validation/amr/paper_figure.py tex vortex amr_conduction.png`.

**Expected result.**
- Resistances, both methods extrapolated to zero cell size:

  | texture | JuliaMag | finite elements | difference | order |
  |---|---|---|---|---|
  | tilted wall | 261.553088 Ω | 261.553087 Ω | 6×10⁻⁹ | 1.9, 2.0 |
  | uniform 45° | 256.233452 Ω | 256.233451 Ω | 4×10⁻⁹ | 1.7, 1.8 |

- The potential of the wall at the cell centres: within 1.2×10⁻⁵ V (100 × 20 cells) and
  2.9×10⁻⁶ V (200 × 40) of the finite elements, out of 1 V.
- Vortex, ratio of the effects at 0.04 and 0.02 (2 if linear):

  | quantity | 0.02 | 0.04 | ratio |
  |---|---|---|---|
  | R(0) − R(0) at Δρ = 0 | 0.298215 Ω | 0.593176 Ω | 1.989 |
  | peak-to-peak of R(t), t > 0.5 ns | 1.22×10⁻³ Ω | 2.40×10⁻³ Ω | 1.97 |
  | largest core shift due to AMR | 0.411 nm | 0.786 nm | 1.91 |

**Notes.**
- The order below 2 for uniform m comes from the corners where the contacts meet the insulating
  sides. The finite elements show it too.
- On a contact face the flux keeps the tangential gradient of the next cell. Dropping it looks
  natural (the face is an equipotential), but it doubles the error. The `README.md` of the
  directory gives the derivation.

---

## V8 — magnum.np problems: depinning, helix, Langevin

**What it checks.** Three demos of magnum.np, each with an analytic answer:
- **Domain-wall depinning** from an anisotropy step: analytic depinning field 1.5683 T.
- **Bulk-DMI helix**: period 4πA/D = 54.5 nm.
- **Langevin**: 4096 uncoupled macrospins in a field at 300 K. The time and ensemble average ⟨mx⟩
  must follow the Langevin function L(ξ) = coth ξ − 1/ξ, ξ = µ0 Ms V H / (kB T). This tests the
  fluctuation–dissipation statistics of the thermal field.

**Files.**
- `examples/stage2_local.jl`: the three problems; writes `examples/stage2_*.txt` and the tables
  behind the figures.
- `gpu_validation/revalidation/langevin_thermalization.jl [DT [TTHERM]]`: the Langevin case
  with a chosen thermalization time.
- `docs/stage2_problems.md`: how each demo was adapted.

**Steps.**
1. `julia --project=examples examples/stage2_local.jl`.
2. `python gpu_validation/revalidation/compare_reval.py stage2`; figures:
   `python gpu_validation/revalidation/plot_reval.py`.
3. Optional: `julia --project=. gpu_validation/revalidation/langevin_thermalization.jl 1e-14 20e-9`.

**Expected result.**
- Depinning at 1.570 T (field step 0.005 T); helix period 55.0 nm.
- Langevin: ⟨mx⟩ = 0.331, 0.535, 0.802, 0.900, 0.949 at ξ = 1, 2, 5, 10, 20; L(ξ) = 0.313, 0.537,
  0.800, 0.900, 0.950. A second realization gives 0.313, 0.539, 0.796, 0.901, 0.951.

**Notes.**
- At ξ = 1 the relaxation time 1/(α γ′ µ0 H) is 3.5 ns. A 2 ns thermalization leaves ⟨mx⟩ about
  0.1 too low there; the script thermalizes for 20 ns.
- The 40 snapshots span 0.4 ps, far less than the correlation time: they count as one sample. The
  sampling error at ξ = 1 is σ ≈ 0.008, so +0.018 is about 2σ.

---

## V9 — Finite temperature: nanodot and vortex hysteresis

**What it checks.**
1. **Nanodot.** A small uniaxial dot along its easy axis, under LLG–Langevin dynamics from 0 K
   upwards. Its time-averaged ⟨mz⟩(T) is compared with the Boltzmann average in one well.
2. **Vortex hysteresis at 50 and 300 K.** A 500 nm × 20 nm Permalloy disc, swept 0 → +150 →
   −150 → +150 mT. The T = 0 loop is set by round-off where the vortex renucleates, because the
   state there is mirror-symmetric. The thermal field breaks that symmetry physically.

**Files.**
- `examples/thermal_demagnetization.jl`: the nanodot; writes `examples/thermal_demagnetization.txt`.
- `examples/disc_hysteresis.jl`: the T = 0 loop (manual, tutorial 2).
- `examples/disc_hysteresis_thermal.jl T SEED [OUT [DWELL [DT [ALPHA [DEVICE [PREC [NSTEPS]]]]]]]`:
  the thermal loop (Heun, fixed step, GPU, Float32 by default).
- `gpu_validation/thermal_hysteresis/`: `phase1/` (dwell and time step), `phase2/` (the ensemble,
  `run_phase2.sh`), `snap/` (snapshot re-runs); its `README.md` has the analysis.

**Steps.**
1. Nanodot: `julia --project=examples examples/thermal_demagnetization.jl`, then
   `python gpu_validation/revalidation/compare_reval.py thermal`.
2. Thermal loop:
   1. A short check first: `julia --project=<env with JuliaMag + CUDA> examples/disc_hysteresis_thermal.jl 50 1 test.txt 2e-9 1e-13 0.5 gpu Float32 2`
      (two field steps).
   2. One full loop: `... disc_hysteresis_thermal.jl 50 1 loop_50K_seed1.txt`. It takes about
      3.2 h on the P1000.
   3. The ensemble of the validation: seeds 1–8 at 50 K, seeds 101–108 at 300 K
      (`gpu_validation/thermal_hysteresis/phase2/run_phase2.sh`, about 25 h on two GPUs).
   4. `python gpu_validation/thermal_hysteresis/phase2/plot_phase2.py`: the table per loop, the
      ensemble statistics and the figure.

**Expected result.**
- Nanodot: within about 2σ of the sampling error of the Boltzmann value (11 independent samples).
- Dwell per field step: 2 ns reproduces the T = 0 reversible branch within the noise and annihilates
  at 60 mT, as T = 0. 0.5 and 1 ns lag it. dt = 10⁻¹³ s and 5×10⁻¹⁴ s agree.
- Ensemble:

  | | 50 K (8 loops) | 300 K (8 loops) | T = 0 |
  |---|---|---|---|
  | virgin annihilation | 61 ± 2 mT | 60 mT | 60 mT |
  | descending switch | −15.0 ± 3.8 mT | −13.1 ± 5.9 mT | −15 mT, direct reversal |
  | switches through a vortex | 16/16 | 16/16 | 1/2 |

- Saturated branch: ⟨mx⟩ is lower than at T = 0 by 0.0013 at 50 K and 0.0076 at 300 K. The ratio is
  6.1: linear in T, as thermal spin waves.

**Notes.** With a given seed the GPU random stream is deterministic: a re-run reproduces the table
bit for bit. The sweep rate, 2.5×10⁶ T/s, is far above an experiment's, so the switching fields are
upper bounds.

---

## V10 — Energy barriers (GNEB) and saddle points

**What it checks.**
1. **Stoner–Wohlfarth macrospin.** Barrier KV(1 − h)² and saddle curvature −2K(1 − h²)/Ms, exact.
2. **Against MicroMagnetic.jl** (its GNEB), with the same end states and the same initial path:
   - an ellipse, along a path through the uniform +y state and along a domain-wall sweep
     (magnum.np demo `energy_barrier_ellipse`);
   - the collapse of one Bloch skyrmion, and of a lattice of four.
3. **Order of each saddle**, from the eigenvalues of the energy Hessian at the climbing image
   (`certifysaddle`).

**Files** (`gpu_validation/gneb/`; outputs and the analysis in `results/` and `results/README.md`).
- `stoner_wohlfarth.jl`.
- `compare_gneb.jl CASE OUT`, CASE `ellipse_uniform`, `ellipse_wall`, `skyrmion`, `lattice4`. It
  needs the environment `gpu_validation/gneb/micromagnetic` (JuliaMag + MicroMagnetic.jl).
- `certify_saddles.jl CASE OUT`: saddle order (dense eigensolver, 5 000–7 200 unknowns).
- `ellipse_magnumnp.py DIR`: magnum.np's string method on the ellipse (its own virtual environment).
- `skyrmion_resolution.jl`, `reject_study.jl`: too few images, a soft spring, the `reject` option.
- `plot_barriers.py`: the paper figure.

**Steps.**
1. `julia --project=. gpu_validation/gneb/stoner_wohlfarth.jl`.
2. For each case:
   1. `julia --project=gpu_validation/gneb/micromagnetic gpu_validation/gneb/compare_gneb.jl skyrmion compare_skyrmion.txt`;
   2. `julia --project=. gpu_validation/gneb/certify_saddles.jl skyrmion certify_skyrmion.txt`.
3. `python gpu_validation/gneb/plot_barriers.py`.

**Expected result.**
- Stoner–Wohlfarth, h = 0 … 0.9: barrier to ≤ 7×10⁻¹⁵ relative, curvature exact, index 1.
- Barriers, 21 images, k = 1 T:

  | case | JuliaMag | MicroMagnetic.jl, relative | saddle index |
  |---|---|---|---|
  | ellipse, through +y | 2.4139554504×10⁻¹⁸ J | +1.97×10⁻⁴ | **2** |
  | ellipse, wall sweep | 1.7100329947×10⁻¹⁸ J | +1.66×10⁻⁴ | 1 |
  | one skyrmion | 7.9422822692×10⁻²⁰ J | −5.5×10⁻¹³ | 1 |
  | four skyrmions | 1.2023840745×10⁻¹⁹ J | −9×10⁻⁹ | **4** |

**Notes.**
- The ellipse differences come from the demagnetizing kernels (JuliaMag uses mumax3's, MicroMagnetic
  Newell's). They are the same when one image is evaluated in both codes.
- An index above 1 means the path is not a transition path:
  - the path through +y keeps the inversion symmetry and stays on a saddle of index 2; the wall
    path is 29 % lower;
  - the four-skyrmion start state is not a minimum.

  Two codes agreeing does not show that a barrier is meaningful: certify the saddle.

---

## V11 — Linear eigenmodes

**What it checks.**
1. **FMR standard problem** (Baker et al., JMMM 421, 428 (2017)): a 120 × 120 × 10 nm Permalloy
   square, 5 nm cells. The 15 frequencies and damping times of Table II.
2. **The gyrotropic mode of the V4 disc**, against free gyration in the time domain, extrapolated to
   zero amplitude.
3. **Against MicroMagnetic.jl**, which linearizes the LLG equation by automatic differentiation:
   operator, frequencies and modes on four systems.

**Files** (`gpu_validation/eigenmodes/`, with a `README.md`; outputs in `results/`).
- `fmr_standard_problem.jl OUT`, `fmr_accuracy.jl`, `fmr_newell.jl` (with `newell.jl`, Newell's
  analytic tensor).
- `disc_gyrotropic.jl OUT [cpu|gpu]`; `gyro_amplitude.jl OUT TOTAL DT B_mT...` (GPU) and
  `gyro_frequency.py`; `damped_shift.jl`.
- `compare_micromagnetic.jl CASE...`, CASE `fmr`, `vortex`, `regions`, `skyrmion`, in an
  environment with JuliaMag and MicroMagnetic.jl.
- `plot_eigenmodes.py`: the paper figure.

**Steps.**
1. FMR:
   1. `julia --project=. gpu_validation/eigenmodes/fmr_standard_problem.jl fmr`;
   2. `julia --project=. gpu_validation/eigenmodes/fmr_newell.jl`;
   3. `julia --project=. gpu_validation/eigenmodes/fmr_accuracy.jl`.
2. Disc:
   1. `julia --project=. gpu_validation/eigenmodes/disc_gyrotropic.jl disc cpu` (or `gpu`);
   2. `julia --project=<env with JuliaMag + CUDA> gpu_validation/eigenmodes/gyro_amplitude.jl gpu_validation/eigenmodes/results/gyro 8e-9 1e-13 10 5 2 1`
      (8 ns of RK4 at 10⁻¹³ s, amplitudes in mT), then
      `python gpu_validation/eigenmodes/gyro_frequency.py 2.0` (it reads `results/gyro`; the
      argument is the start of the fit window, in ns);
   3. `julia --project=. gpu_validation/eigenmodes/damped_shift.jl`.
3. `julia --project=gpu_validation/gneb/micromagnetic gpu_validation/eigenmodes/compare_micromagnetic.jl`
   (any environment with JuliaMag and MicroMagnetic.jl will do; with no CASE it runs all four).
4. `python gpu_validation/eigenmodes/plot_eigenmodes.py`.

**Expected result.**
- FMR, with the frequencies scaled to the paper's γ: from −6.9×10⁻⁴ to −4.4×10⁻⁴ relative with
  JuliaMag's kernel; from −3.4×10⁻⁴ to −2.7×10⁻⁴ with Newell's tensor. With Newell's tensor, one
  common factor of 1.00031 brings all 15 frequencies within the 0.5 MHz rounding of the table.
  MicroMagnetic.jl shows the same factor.
- Disc:
  - eigenfrequency 968.234 MHz, Γ = 0.079401 ns⁻¹;
  - free gyration extrapolated to zero amplitude: 968.088 MHz, Γ = 0.0794 ns⁻¹;
  - the exact damped eigenvalue at α = 0.01 is 968.072 MHz. The extrapolated frequency is
    1.7×10⁻⁵ above it.
- MicroMagnetic.jl: all frequencies within 2.3×10⁻¹³ to 5.2×10⁻¹¹ relative; overlaps of the lowest
  10 modes within 4×10⁻¹⁶ of 1.

**Notes.**
- The 969.8 MHz of V4 is the Hilbert estimate of `gyrofreq.py`, which is biased by +0.9 MHz at this
  decay rate. The comparison between codes in V4 is not affected.
- States with zero modes are refused; partially filled cells only with the dense solver.

---

## V12 — The GPU against the CPU

**What it checks.** Every GPU method against the CPU method, in Float64 and Float32:
- every field, torque and helper, on open and periodic meshes, with scalar and region materials;
- the solvers and the minimizer, the trackers, a whole `Simulation` with its table and OVF output;
- fields and currents in time, the conduction, the eigenmode operators and the saddle certification.

**Files.**
- `gpu_validation/test_gpu.jl`: the suite.
- `examples/gpu_check.jl`, `examples/gpu_demag_check.jl`: quick checks.
- `gpu_validation/README.md`: the results, section by section.

**Steps.**
1. On a machine with a CUDA GPU, from the repository root:
   `julia --project=<env with JuliaMag + CUDA> gpu_validation/test_gpu.jl`.
2. Quick check: `julia --project=. examples/gpu_demag_check.jl`.

**Expected result.**
- The whole suite passes (1143 checks at the close of the first stage).
- Float64: the GPU equals the CPU to ≤ 3×10⁻¹⁵ per method; 200 RK4 steps to 1.4×10⁻¹⁵.
- Float32: ≤ 3.3×10⁻⁷ per method.

**Notes.** Some checks compare quantities the CPU itself does not fix to rounding: the
Barzilai–Borwein steps far from equilibrium, and the error estimate of an adaptive step. Their
tolerance is measured from the CPU's own spread under a one-ulp perturbation, not set by hand.
