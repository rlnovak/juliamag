# Linear eigenmodes: validation

`src/eigenmodes.jl`, 2026-11-25. Float64. Analytic cases: `test/test_eigenmodes.jl`.

## FMR standard problem (Baker et al., JMMM 421, 428 (2017))

`fmr_standard_problem.jl`, `fmr_newell.jl`, `fmr_accuracy.jl`. 120 × 120 × 10 nm, 5 nm cells,
against Table II of the paper (15 modes, eigenvalue method, α = 0.008); frequencies scaled to
the paper's γ (2.210173e5 m/(A s), 0.99960 of JuliaMag's).

| demagnetizing kernel | f, JuliaMag/paper − 1 | damping time, same |
|---|---|---|
| JuliaMag (as mumax3), accuracy 6 | −6.9e-4 … −4.4e-4 | +5e-4 … +1.2e-3 |
| accuracy 12 | −5.2e-4 … −4.1e-4 | +3e-4 … +1.0e-3 |
| accuracy 24 | −4.4e-4 … −3.7e-4 | +2e-4 … +9e-4 |
| accuracy 48 | −3.9e-4 … −3.3e-4 | +2e-4 … +8e-4 |
| Newell (analytic, `newell.jl`) | −3.4e-4 … −2.7e-4 | +1e-4 … +8e-4 |

With Newell's tensor one factor, 1.00030552 to 1.00030797, brings all 15 frequencies within the
rounding of the table (0.5 MHz; largest deviation 0.48 MHz). The same factor brings all 15
damping times within 0.5 ps, if the factor 1/(1 + α²) of the Gilbert form is left out of the
paper's rates. The origin of the factor is not known; it is the same for
all modes.

- JuliaMag's numerical kernel (accuracy 6) differs from Newell's by up to 2.4e-4 (diagonal) and
  7.9e-5 (off-diagonal) in N. This is the kernel of mumax3; not changed.
- Lanczos against dense: f 4e-15, linewidth 1e-11.
- 2.5 nm cells (Lanczos, 18 432 unknowns, 117 s on the CPU): −1e-3 … +1e-2 relative to the paper's 5 nm values.
- `results/oommf_spectrum.txt`: power spectrum of the OOMMF ringdown of the paper's repository.

## Vortex disc

`disc_gyrotropic.jl` (128 × 16 nm, 64 × 64 × 8, staircase, CPU, 51 648 unknowns): 968.234 MHz,
Γ = 0.079401 1/ns, residual 9e-11; next modes 10.53, 11.07, 11.39, 12.39, 12.41 GHz.
`gyro_amplitude.jl` + `gyro_frequency.py` (GPU, Float64, RK4, 10/5/2/1 mT): f = f0 + κ<|z|²>,
f0 = 968.088 MHz (window 2–8 ns; 968.080–968.088 for 0.5–4 ns starts), Γ = 0.0794 1/ns.
`damped_shift.jl`: the exact damped eigenvalue is shifted by −1.69 α² (28 × 28 × 1) and −1.68 α²
(32 × 32 × 2) relative: 968.072 MHz for α = 0.01. f0 / that − 1 = +1.7e-5 (2–8 ns), +0.8e-5 to
+1.7e-5 over the windows.

The estimator of `disk_stt/gyrofreq.py` (moving average + Hilbert) is biased by +0.9 MHz at this
decay rate (a synthetic 968.8 MHz signal gives 969.5–969.8): the 969.8 MHz of the code
comparison is that estimate. The comparison between codes is not affected.

The same on the GPU (P1000, Float64): identical to all printed digits, 136 s against 430 s.

## Code checks

CPU suite 5755/5755 (502 of them in `test_eigenmodes.jl`, at -O1 and -O2); `test_gpu.jl` on
labmac03 1102/1102: operator, dense Hessian, curvatures (dense, LOBPCG), frequencies and linewidths
(dense, Lanczos) and a saddle certified on the GPU, equal to the CPU to 5e-15 (linewidths 1e-10,
saddle curvatures 8e-11) in Float64; operator to 2e-7 in Float32. An independent reviewer
compared the frequencies with a separate integration of `llgrhs!` (vortex, flower state, two
regions, partial cells: 6e-11 to 5e-8, 2.5e-6 for the softest mode) and found the faults listed in
`docs/plan_neb_eigen.md` §2, since fixed.

Limits: states with zero modes are refused; partially filled cells only with the dense solver (up
to 2000 unknowns); no spin-transfer torque.

## Saddle points

`../gneb/certify_saddles.jl`, results in `../gneb/results/`.
