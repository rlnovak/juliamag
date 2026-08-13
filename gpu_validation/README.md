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

## Result 6 — extension scope

The **entire** JuliaMag pipeline now runs on the GPU end-to-end via `togpu`:
`exchange!`, `anisotropy!`, `dmi!` (interfacial + bulk), `zeeman!`, `demagfield!`,
the assembled `effectivefield!`, `torque!`, `zhanglitorque!`,
`slonczewskitorque!`, `normalize!`, `average`, the **`Minimizer` (relax) and
`Integrator` (run)**, and the **feature trackers** (`vortexcore`, `skyrmionpos`,
`domainwallpos`, `topologicalcharge`) — a complete micromagnetic simulation with
its diagnostics, nothing falling back to the CPU.

Core changes were type generalizations only — the `World` scratch buffer and the
`Minimizer`/`Integrator` state fields from `Array` to `AbstractArray`, and three
minimizer inner loops factored into helpers the extension overrides with
broadcast GPU versions. The CPU logic is unchanged; **all 2299 CPU tests pass**.

## Files

- `gpu_check.txt` — field/torque kernels, CPU vs GPU agreement.
- `demag_gpu.txt` — demag + full effective field: accuracy and speedup.
- `solvers_gpu.txt` — relax + run correctness, std4 end-to-end, relax speedup.
- `dmi_stt_gpu.txt` — DMI + spin-transfer torques accuracy, skyrmion end-to-end, speedup.
- `trackers_gpu.txt` — vortex/skyrmion/domain-wall/topological-charge, CPU vs GPU.
- `std4/` — the Standard Problem 4 GPU run: table, GPU figure, CPU-vs-GPU overlay.
- `gpu_scaling_exchange.txt`, `gpu_scaling_effectivefield.txt` — isolated-kernel
  timings (Result 1; no demag), for contrast.
