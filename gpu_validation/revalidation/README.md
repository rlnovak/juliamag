# Re-validation after the November 2026 fixes and GPU rewrite

Every older validation problem was re-run with the code at `36a619c`. That commit
followed several fixes:

- exchange edge-fill: `bad1f2c`, `0b9110d`;
- demag energy with the cell fill: `9055899`;
- ⟨m⟩ over the magnet and the vortex-core tracker: `3daa142`, `01a62ff`, `39f2f9e`;
- the GPU path rewritten as mumax3-style per-term kernels, with a real Fourier
  demag kernel, batched FFTs and Float32 (`36a619c`).

The re-runs took place on 2026-11-20 on labmac03, with GPU-free CPU jobs also on
labmac04. The figures from before the re-validation are kept, renamed with their
dates, in [`archive/figures_pre-revalidation/`](../../archive/figures_pre-revalidation/).

## Files

**Campaign**
- `run_reval.sh`: the launcher.
  - `labmac03` role: the CPU jobs, plus two GPU chains through `run_when_gpu_idle.sh`.
  - `labmac04` role: `fieldcmp`, which needs the mumax3 OVF outputs stored there.
  - `SMOKE=1` runs each job briefly as a check.
- `logs/`: every job's log, ending with `done rc=N`.

**Analysis and figures**
- `compare_reval.py`: compares each new result with its reference (OOMMF, mumax3,
  theory) and with the previous JuliaMag result, read from git at `36a619c`.
- `plot_reval.py`: draws the new figures with matplotlib:
  - `paper/figures/stdproblem2_*`, `stdproblem4_compare`, `skyrmion_drive`,
    `vortex_hysteresis`;
  - the matching `examples/` figures, the stage-2 figures, the thermal figure
    and `std4/stdproblem4_cpu_vs_gpu`.
  - The disc figures come from their own scripts in `examples/` (`plot_paper_disk_stt.py`,
    `plot_disk40_*.py`, …).

**Diagnostics**
- `disc_hysteresis/`: the vortex-hysteresis diagnosis (below).
- `langevin_thermalization.jl`: the Langevin thermalization check (below).

## Results

| problem | reference | new result | new vs previous JuliaMag |
|---|---|---|---|
| std 2 remanence, d/ℓex 0.5–30 | OOMMF | \|Δ⟨mx⟩\|, \|Δ⟨my⟩\| ≤ 3×10⁻⁴ | ≤ 1.3×10⁻⁵ |
| std 2 Hc, Hs (7 sizes) | OOMMF | ΔHc ≤ 7×10⁻⁴ Ms, ΔHs ≤ 6×10⁻⁴ Ms (field step 10⁻³) | ≤ 5×10⁻⁶ |
| std 4, 0–1 ns | mumax3 / OOMMF | max \|Δ⟨m⟩\| 0.0044 / 0.0037 (unchanged) | 5×10⁻⁶ |
| std 4 on the GPU | CPU | 1.6×10⁻⁵ (Tsit5, minimizer path) | — |
| std 4 / std 5, components and energies | mumax3 | rms ≤ 0.042 % / ≤ 0.001 % (unchanged) | 5×10⁻⁶ / 0 |
| std 5, ⟨m⟩(1 ns) | mumax3 | (−0.23488, −0.09453, 0.02296) vs (−0.23480, −0.09454, 0.02296) | 0 |
| disc, static fields (ES 0 / 4) | mumax3 OVF | B_demag 7.7×10⁻⁷, B_exch 2.3×10⁻⁷ relative | same |
| disc, Slonczewski, 7 ns | mumax3 | f 973.5 / 973.5 MHz; max \|Δ⟨m⟩\| 4.3×10⁻⁵; core 0.006 nm rms | ⟨m⟩ 0; core moved ≤ 0.7 nm (tracker fix) |
| disc, Slonczewski + Zhang–Li, 7 ns | mumax3 | 930.8 / 930.8 MHz; 4.0×10⁻⁵; core 0.003 nm rms | 1×10⁻⁷ |
| disc, free gyration, 7 ns | mumax3 | 969.8 / 969.8 MHz, Γ 0.0797 ns⁻¹; 8.9×10⁻⁵; core 0.006 nm rms | 3.9×10⁻⁶ |
| disc, dt 5×10⁻¹⁴ vs 2.5×10⁻¹⁴, ES 4, 7 ns | — | \|Δ⟨m⟩\| < 10⁻⁷ (table precision) | — |
| disc, 40 ns, ES 4, Float64 | mumax3 | 962.318 MHz both; max \|Δ⟨m⟩\| 3.7×10⁻⁵; core 0.006 nm rms; wall 11 504 s (was 25 902 s) | identical to all printed digits |
| disc, torques on mumax3's m at 17 ns | mumax3 | 2.8×10⁻⁵ relative, same max cell | same |
| disc, 200 ps restart from mumax3's 17 ns state | mumax3 | maxTorque 10.25 mT (mumax3 MaxErr 10⁻⁷ / RK4: 10.25) | identical |
| skyrmion on a periodic wire, 3 ns | — | v_x = −177.7 m/s, Δy = 62.5 nm | identical |
| magnum.np: domain-wall depinning | analytical 1.5683 T | 1.570 T (field step 0.005 T) | identical |
| magnum.np: bulk-DMI helix | 4πA/D = 54.5 nm | 55.0 nm | identical |
| magnum.np: Langevin ensemble | L(ξ) | ξ ≥ 10: within 0.001; ξ = 1, 2: 0.095 low | statistically the same |
| thermal nanodot ⟨mz⟩(T) | Boltzmann, one well | within ≈ 2σ of the sampling error (11 independent samples) | stochastic |
| vortex hysteresis, 500 nm disc | — | reversible branch and annihilation (55–60 mT) unchanged; return branch differs (below) | max \|Δ⟨mx⟩\| 0.61 |

## The vortex hysteresis loop

After the rewrite, the descending branch reverses at −15 mT directly to the
opposite quasi-uniform state. The old run renucleated a vortex there
(⟨mx⟩ = −0.34). This is not a bug. `disc_hysteresis/symmetry_check.jl` shows that
the state at −10 mT is mirror-symmetric under y → −y to 2.7×10⁻⁶. The new demag
kernel zeroes its odd components exactly on their symmetry planes, and the
calculation now keeps that symmetry.

At −15 mT the quasi-uniform state loses stability, and which state the minimizer
reaches depends on how the symmetry is broken:

| start from −10 mT | state at −15 mT |
|---|---|
| unperturbed | (−0.9488, 0, 0): reversed |
| 10⁻⁶ random perturbation | (−0.3399, 0, 0.0012): vortex, as the old run |
| 10⁻³ random perturbation | (−0.9413, 0.033, 0): reversed |

Full loops vary in the same way:
- 10⁻⁶ random perturbation before every step, `perturbed_loop.jl`, seeds 1 and 2:
  the vortex renucleates at 0 mT (seed 1), or the disc reverses at −10 mT
  (seed 2).
- Field tilted by 0.1° or 1°, `tilted_loop.jl`: reversal at −10 mT or −20 mT.

The virgin branch, the annihilation at 55–60 mT and the saturation value are the
same in every variant. The zero-temperature quasi-static loop therefore does not
determine where the vortex renucleates. The paper now shows the deterministic loop
together with the two perturbed loops and says so.

## Langevin (magnum.np stage 2)

At ξ = 1 and 2, ⟨mx⟩ falls about 0.1 below L(ξ), in the old run as well as in the
new one. The probable cause is the protocol, not the dynamics:
`stage2_local.jl` thermalizes for 2 ns from a random state, but at ξ = 1 the
applied field is only 4.1 mT, and the relaxation time 1/(α γ′ μ0H) is 3.5 ns.
`langevin_thermalization.jl` repeats the ξ = 1, 2 and 5 cases with 20 ns of
thermalization and gives ⟨mx⟩ = 0.3119, 0.5328 and 0.7995, against L(ξ) = 0.3130,
0.5373 and 0.8001 (`langevin_thermalization.txt`). `stage2_local.jl` now
thermalizes for 20 ns with dt = 1e-14 s. Its re-run (`logs/stage2_rerun_langevin20ns.log`,
`examples/stage2_langevin.txt`) gives 0.331, 0.535, 0.802, 0.900 and 0.949 for
ξ = 1, 2, 5, 10, 20, against L(ξ) = 0.313, 0.537, 0.800, 0.900 and 0.950.

The +0.018 at ξ = 1 is about 2σ of the sampling error. The per-spin variance
of mx is about 0.28 there, so the ensemble mean over 4096 spins has σ ≈ 0.008.
The 40 snapshots span 0.4 ps, much less than the 3.4 ns correlation time, so they
count as one sample. Depinning and the helix are unchanged.
