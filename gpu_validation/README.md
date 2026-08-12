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

## Result 3 — extension scope

On the GPU end-to-end via `togpu`: `exchange!`, `anisotropy!`, `zeeman!`,
`torque!`, `normalize!`, `average`, `demagfield!`, and the assembled
`effectivefield!` (through `togpu(world)`). All validated above.

Not yet on the GPU: DMI, spin-transfer torques, and the solvers (`Minimizer`,
`Integrator`) — whose Barzilai–Borwein / Cayley steps still use scalar CPU loops.
A full relax/run on the GPU is the next extension target. None of this touches
the tested CPU code; the core change was only to generalize the `World` scratch
buffer type from `Array` to `AbstractArray` (all 2299 CPU tests still pass).

## Files

- `gpu_check.txt` — field/torque kernels, CPU vs GPU agreement.
- `demag_gpu.txt` — demag + full effective field: accuracy and CPU-vs-GPU speedup.
- `gpu_scaling_exchange.txt`, `gpu_scaling_effectivefield.txt` — isolated-kernel
  timings (Result 1; no demag), for contrast.
