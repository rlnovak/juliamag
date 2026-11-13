# 128 nm disk, spin-torque vortex gyration — JuliaMag vs mumax3

Like-for-like validation of JuliaMag against mumax3 3.11.1 on a curved geometry with
fractional edge cells: a Permalloy disk, 128 nm diameter × 16 nm, 64×64×8 cells of
2 nm (Msat = 800 kA/m, A = 10.8 pJ/m, α = 0.01), vortex (circ 1, pol 1), driven by
a perpendicular current of 40 mA (jz = 3.108×10¹² A/m²) through a fixed layer
p = (cos 20°, sin 20°, 0) with Slonczewski parameters P = 0.37, Λ = 1, ε′ = 0.01.
Both codes ran on the labmac04 GPUs (JuliaMag: fixed-step RK4, dt = 5×10⁻¹⁴ s;
mumax3: its default adaptive RK45). Tables are logged every 5 ps.

## Layout

- `mumax3/stt_es0_reference.mx3` — the original reference run (`sims1`); note it
  sets neither `EdgeSmooth` (default 0, staircase) nor `DisableZhangLiTorque`
  (default false), so it applies **Slonczewski + Zhang–Li**.
- `mumax3/stt_es0_noZL.mx3` — same, `DisableZhangLiTorque = true`, 7 ns.
- `mumax3/stt_es4_mumax.mx3` — same as the reference with `EdgeSmooth = 4`, 7 ns.
- `mumax3/free_es0.mx3` — free gyration: relax under Bx = 10 mT, field off, 7 ns.
- `mumax3/fields_es{0,4}.mx3` — relax and save m, B_demag, B_exch, geom, energies.
- `juliamag/gyro_lean.jl` — GPU driver, `julia gyro_lean.jl ES DT TOTAL MODE OUT`
  with MODE `stt` (Slonczewski), `sttzl` (Slonczewski + Zhang–Li, ξ = 0, as mumax3's
  default) or `free`.
- `juliamag/fieldcmp.jl` — loads mumax3's relaxed m (OVF) and compares B_demag and
  B_exch cell by cell, plus the energies.
- `gyrofreq.py` — frequency estimator (Hilbert phase with a one-period moving-average
  detrend, FFT cross-check). mumax3 tables have a non-uniform time axis (adaptive
  step), so every signal is resampled onto a uniform grid first; an FFT that assumes
  `t[1]−t[0]` is biased by −1.6%.
- `frequencies.py` — reproduces the table below from `tables/`.
- `tables/` — all time series (`mumax_ref_es0.txt` is the reference trimmed to 7 ns;
  JuliaMag tables: `t mx my mz core_x core_y`, ⟨m⟩ averaged over the disk cells).

## Results

**Static fields** (mumax3's relaxed m loaded into JuliaMag, EdgeSmooth 0 and 4):
max relative error 7.7×10⁻⁷ in B_demag and 2.3×10⁻⁷ in B_exch; geometry fill
identical cell by cell; energies equal to mumax3's (EdgeSmooth 0: E_demag =
7.1783×10⁻¹⁹ J, E_exch = 3.6270×10⁻¹⁸ J, E_total = 4.3448×10⁻¹⁸ J).

**Dynamics** (EdgeSmooth 0; f = mean of the ⟨mx⟩, ⟨my⟩ Hilbert estimates over 1–7 ns,
0.5–7 ns for free gyration):

| run | mumax3 | JuliaMag |
|---|---|---|
| Slonczewski + Zhang–Li (mumax3 default) | 930.8 MHz | 930.8 MHz |
| Slonczewski only | 973.5 MHz | 973.5 MHz |
| free gyration (no current) | 969.8 MHz | 969.8 MHz |
| free-gyration decay rate Γ | 0.0797 ns⁻¹ | 0.0797 ns⁻¹ |

The codes differ by < 0.05 MHz; the ±1.5 MHz spread between the x and y estimates is
the absolute uncertainty of the estimator, common to both. ⟨m⟩ agrees to 0.007% (mx,
my) and 0.045% (mz) rms with Zhang–Li. Time-step convergence: dt = 5×10⁻¹⁴ and
2.5×10⁻¹⁴ s give the same frequency (963.6 MHz, EdgeSmooth 4, Slonczewski only) and
max |Δ⟨m⟩| < 10⁻⁶ (the table print precision). Edge smoothing lowers the frequency by
≈1% (mumax3 with Zhang–Li: 930.8 → 922.0 MHz; JuliaMag Slonczewski only: 973.5 → 963.6 MHz).

The vortex-core positions differ by 2–4% because the trackers differ (JuliaMag's
`vortexcore` vs mumax3's `ext_corepos`); frequencies are therefore taken from ⟨m⟩.

## Physical model

The Zhang–Li term with a perpendicular current acts only through ∂z m across the
thickness and shifts the frequency by −4.4%. The standard micromagnetic model of the
current-perpendicular vortex oscillator is the Slonczewski torque plus the Oersted
field (Khvalkovskiy et al., Phys. Rev. B 80, 140401(R) (2009)); the Zhang–Li term was
present in the reference only because it is mumax3's default. The JuliaMag result
we report is therefore **Slonczewski only**; the Zhang–Li run is a code-to-code check.
Neither code includes the Oersted field here, so the absolute frequency is a model
result for validation, not a device prediction.
