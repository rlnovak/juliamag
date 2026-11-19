# JuliaMag GPU validation — labmac04.blumenau.ufsc.br

Remote run of the CUDA extension on real NVIDIA hardware.

- **Machine:** labmac04.blumenau.ufsc.br (Ubuntu 22.04, kernel 5.15)
- **GPU:** NVIDIA Quadro P1000 (GP107GL, Pascal, compute capability 6.1), ×2
- **Julia:** 1.11.9 (`~/opt/julia-1.11.9`)
- **CUDA.jl:** functional (`CUDA.functional() == true`); a driver/NVML version
  mismatch made `nvidia-smi` fail, but the CUDA driver (`libcuda`) itself works,
  so CUDA.jl runs fine.
- **Repo commit:** 5fa93dc

## Result 1 — correctness (PASS, exact)

`examples/gpu_check.jl` compares every GPU field/torque method against the CPU
method on a 64×64×4 mesh (see `gpu_check.txt`):

| method     | max rel diff (CPU vs GPU) |
|------------|---------------------------|
| exchange   | 3.47e-16 |
| anisotropy | 0 (exact) |
| zeeman     | 0 (exact) |
| torque     | 2.01e-16 |
| average    | 1.47e-17 |
| normalize  | 2.22e-16 |

Every implemented GPU method agrees with the CPU method to floating-point
precision. The same source runs on `Array` and `CuArray`, dispatching by array
type alone (path B: additive extension, no change to the tested CPU code). In
particular the Neumann edge handling in the GPU exchange — the term flagged as
most likely to need a fix — is correct.

## Result 2 — demag on the GPU (PASS, exact) + speedup

The demagnetization field — the dominant per-step cost — now runs on the GPU
(FFT convolution via CUFFT). It agrees with the CPU to floating-point precision,
as does the **full effective field** (exchange+anisotropy+demag+zeeman) driven
through `togpu(world)` (see `demag_gpu.txt`):

| quantity | geometry | max rel diff CPU vs GPU |
|----------|----------|--------------------------|
| demag field | std4 / cube3D / film | 6.4e-16 / 6.1e-16 / 5.8e-16 |
| full effective field | std4 / cube3D / film | 3.8e-16 / 4.1e-16 / 4.6e-16 |

**With the demag included, the GPU beats the CPU:**

| cells | CPU (ms) | GPU (ms) | speedup |
|-------|----------|----------|---------|
| 4 096 | 0.52 | 1.06 | 0.49× |
| 16 384 | 2.45 | 2.23 | 1.10× |
| 65 536 | 13.64 | 7.34 | 1.86× |
| 262 144 | 59.65 | 27.55 | 2.17× |
| 524 288 | 269.01 | 87.69 | 3.07× |
| 1 048 576 | 326.16 | 115.32 | 2.83× |

Crossover at ~16k cells, up to ~3× at 512k — on a modest 640-core Quadro P1000.
The demag FFT scales far better on the GPU than the isolated field kernels of
Result 1 (`gpu_scaling_*.txt`), which stayed below the CPU. A last-generation GPU
(planned Colab benchmark) should widen this substantially.

## Result 3 — solvers on the GPU (PASS) + full Standard Problem 4

The energy minimizer (`relax`) and the LLG time integrator (`run`) now run on the
GPU too, so a **whole simulation** — relax to equilibrium, then integrate the
dynamics — runs end-to-end on the device via `togpu` (see `solvers_gpu.txt`):

- **Relax vs CPU:** ⟨m⟩ agrees to 2.7e-7 (the Barzilai–Borwein step count differs
  by a few due to arithmetic order, but both reach the same equilibrium).
- **Run vs CPU:** ⟨m⟩ agrees to 3.4e-13 after an LLG switching segment.
- **µMAG Standard Problem 4, entirely on the GPU:** the relaxed S-state is
  ⟨m⟩=(0.9668, 0.1256, 0) — matching mumax3 — and the 1 ns switching curve agrees
  with the CPU reference to **~9e-6** across the whole run. CPU (lines) and GPU
  (dots) coincide: `std4/stdproblem4_cpu_vs_gpu.png`.

Relax performance: crossover at ~65k cells, up to ~2× at 256k on this GPU (the
minimizer step has more kernel launches than the field alone, so slightly below
the field's ~3×).

## Result 4 — DMI and spin-transfer torques on the GPU (PASS)

The interfacial and bulk DMI fields and the Zhang–Li and Slonczewski
spin-transfer torques now run on the GPU (see `dmi_stt_gpu.txt`):

| term | geometry | max rel diff CPU vs GPU |
|------|----------|--------------------------|
| DMI interfacial | film / periodic-x | 0 (exact) |
| DMI bulk | 3D | 0 (exact) |
| Zhang–Li STT | Jx / Jx+Jy periodic | 4.4e-16 / 3.7e-16 |
| Slonczewski STT | film | 2.4e-16 |

DMI matches to **exactly zero** — the central-derivative kernels reuse the
exchange field's Neumann/periodic edge shift, reproducing the CPU one-sided
boundary exactly. **End-to-end**, a Néel skyrmion (DMI + PMA) relaxed and then
driven by a Zhang–Li current — DMI + STT + demag + minimizer + integrator all on
the GPU — matches the CPU to 2.8e-13 in ⟨m⟩ and to 6.5e-8 nm in position.

The whole physics pipeline now runs on the GPU. On this modest GPU the
derivative-based DMI/STT do not beat the CPU on their own (0.2–0.8×, like the
other isolated kernels); the speedup comes from the demag FFT in the full loop.

## Result 5 — feature trackers on the GPU (PASS)

All four trackers now run on the GPU too (see `trackers_gpu.txt`):

| tracker | CPU vs GPU |
|---------|-----------|
| vortex core (x, y, polarity) | 0 (exact) |
| topological charge Q | 2.6e-15 |
| skyrmion position (open / periodic-x) | ~1e-13 nm |
| domain-wall position | 0 (exact) |

Each locates a feature by a device reduction (`sum` / `findmax`) over per-cell
quantities built by broadcast — no scalar CuArray indexing. Only a handful of
scalars come back to the host (the argmax cell and its 3×3 neighbourhood for the
vortex sub-cell interpolation; the `Nx` column averages for the wall crossing).

## Result 6 — region-wise (multi-material) parameters on the GPU (PASS)

Multi-material simulations (`RegionParams`) run on the GPU too. `togpu(::World)`
materializes the per-region lookup tables into per-cell device arrays (gathered
through the cell→region map), so the field kernels become broadcasts over those
arrays — exchange with harmonic-mean stiffness at interfaces, anisotropy, DMI,
and the region-aware demag (convolution on Msat[cell]·m, prefactor μ0). See
`regionwise_gpu.txt`:

| case | CPU vs GPU |
|------|-----------|
| bilayer effective field (Msat/Aex/Ku per layer) | 3.04e-16 |
| disc + region DMI/PMA + empty background | 3.71e-16 |
| bilayer relax ⟨m⟩ | 3.8e-6 |

Both match to floating-point precision, including the harmonic-mean coupling
across the region interface and the empty (Msat=0) cells.

## Result 7 — extension scope

The **entire** JuliaMag pipeline now runs on the GPU end-to-end via `togpu`:
`exchange!`, `anisotropy!`, `dmi!` (interfacial + bulk), `zeeman!`, `demagfield!`,
the assembled `effectivefield!`, `torque!`, `zhanglitorque!`,
`slonczewskitorque!`, `normalize!`, `average`, the **`Minimizer` (relax) and
`Integrator` (run)**, the **feature trackers** (`vortexcore`, `skyrmionpos`,
`domainwallpos`, `topologicalcharge`), and both material models (scalar
`Material` and multi-region `RegionParams`) — a complete micromagnetic simulation
with its diagnostics, nothing falling back to the CPU.

Core changes were type generalizations only — the `World` scratch buffer and the
`Minimizer`/`Integrator` state fields from `Array` to `AbstractArray`, and three
minimizer inner loops factored into helpers the extension overrides with
broadcast GPU versions. The CPU logic is unchanged; **all 2299 CPU tests pass**.

## Result 8 — mumax3-style kernels, Float32, performance (PASS)

The GPU methods were rewritten after profiling showed the broadcast version to be
launch- and allocation-bound (182 kernel launches and 74 device allocations per
right-hand side). Now, as in mumax3:

- each field and torque term is one CUDA kernel (`ext/cuda_kernels.jl`), one thread
  per cell, repeating the arithmetic of the CPU loop; parameters are scalars or
  per-cell arrays (regions) through the same kernel;
- demag: the Fourier kernel is real (exact reflection symmetry, odd components
  zeroed on their symmetry planes) and stored as such, transformed in Float64 with
  1/N folded in; the three components go through one batched rFFT/brFFT; padding
  written once; the C2R transform skips CUDA.jl's input copy;
- Float32 runs via `Material{Float32}` / `material(...; precision = Float32)`;
  reductions accumulate in Float64.

One RHS now launches 14 kernels and allocates nothing.

**Correctness** — `test_gpu.jl` (446/446, `test_gpu.txt`), every GPU method vs
CPU on open/periodic meshes, scalar/region materials, both DMI forms:

| | Float64 | Float32 |
|---|---|---|
| every field / torque / helper (203 checks each) | ≤ 2.8e-15 | ≤ 3.3e-7 |
| fixed-step RK4, 200 steps | ≤ 1.4e-15 | ≤ 8.1e-7 |
| adaptive Tsit5, 10 ps | ≤ 2.5e-10 | ≤ 1.4e-6 |
| 20 BB minimizer steps | ≤ 1.2e-12 | ≤ 2.6 × own 1-ulp spread |

The BB check uses tol = max(floor, 10× the CPU's own 1-ulp-perturbation spread):
far from equilibrium the BB sequence amplifies rounding ~1e4× in 20 steps (Float32
"mat open 3D Dind": CPU-vs-CPU spread 5.7e-4, GPU 6.5e-4), so a fixed tolerance is
ill-posed there.

**Physics** — 40 ns disc (`disk_stt/`, Slonczewski, EdgeSmooth 4):

- Float64, kernels vs broadcast version, 1 ns: identical to every printed digit
  (⟨m⟩, energies, maxTorque, core position; `tables/jm64_es4_1ns_rewrite.txt`).
- Float32 vs Float64 over 40 ns: max |Δ⟨m⟩| 3.6e-6, total energy 2.9e-6 relative,
  core 0.065 nm (`tables/jm32_es4_40ns.txt`).
- Float32 vs mumax3: orbit-phase frequency 962.318 MHz in both; deviations equal to
  those of the Float64 run within 5e-7.

**Performance** — Quadro P1000, 64×64×8 cells (`disk_stt/bench_rhs_kernels.txt`,
`disk_stt/perf_campaign_labmac03.txt`, `disk_stt/mumax3_bench_labmac03.txt`):

| | ms / RHS | 40 ns wall (s) |
|---|---|---|
| broadcast version, Float64 | 7.5 | 25 902 |
| broadcast version, Float32 | 5.4 | — |
| kernels, Float64 | 3.54 (FFTs 2.8: FP64 at 1/32 rate) | 11 600 (290 s/ns) |
| kernels, Float32 | **1.11** | **4 008** |
| mumax3 (Float32) | 1.11 | 2 025 |

Per evaluation JuliaMag Float32 = mumax3. The remaining 2× in wall time is the
number of evaluations: fixed RK4 at 50 fs (3.2e6) vs mumax3's adaptive RK45
(1.8e6) — adaptive stepping with STT is the next item. Old timings on labmac04,
new on labmac03 (same GPU model): mumax3 takes 2017 s and 2025 s on the two.

## Files

- `test_gpu.txt` — Result 8: every GPU method vs CPU, Float64 and Float32 (`test_gpu.jl`).
- `gpu_check.txt` — field/torque kernels, CPU vs GPU agreement.
- `demag_gpu.txt` — demag + full effective field: accuracy and speedup.
- `solvers_gpu.txt` — relax + run correctness, std4 end-to-end, relax speedup.
- `dmi_stt_gpu.txt` — DMI + spin-transfer torques accuracy, skyrmion end-to-end, speedup.
- `trackers_gpu.txt` — vortex/skyrmion/domain-wall/topological-charge, CPU vs GPU.
- `regionwise_gpu.txt` — multi-material (RegionParams) fields and relax, CPU vs GPU.
- `std4/` — the Standard Problem 4 GPU run: table, GPU figure, CPU-vs-GPU overlay.
- `gpu_scaling_exchange.txt`, `gpu_scaling_effectivefield.txt` — isolated-kernel
  timings (Result 1; no demag), for contrast.
