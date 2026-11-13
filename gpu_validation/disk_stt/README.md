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
- **40 ns production series** (EdgeSmooth 4, Slonczewski only, the physical model):
  `juliamag/disk_d1=128nm_t=16nm_40mA.jl` (the JuliaMag port, full mumax3-style
  table: ⟨m⟩, maxTorque [T], energies, core) and `mumax3/stt_es4_noZL_40ns.mx3` (the
  `sims1` reference with only `EdgeSmooth = 4` and `DisableZhangLiTorque = true`
  added). `run_queue_disk40.sh` runs both on labmac04, each waiting for its GPU to be
  idle; `examples/plot_disk40_compare.py` compares them, `examples/plot_disk40_paper.py`
  draws the paper figure. `mumax3/stt_es4_noZL_40ns_maxerr1e-7.mx3` is the same
  mumax3 run with `MaxErr = 1e-7` (see the maxTorque note below).
- **maxTorque diagnostics**: `mumax3/stt_es4_noZL_snap.mx3` (snapshots of m, torque,
  LLtorque, STTorque, B_eff at 12 and 17 ns), `juliamag/torque_on_mumax_m.jl`
  (JuliaMag torques on mumax3's m, cell by cell), `mumax3/restart17_{default,
  maxerr1e-7,rk4fixed}.mx3` and `juliamag/restart_from_mumax_m.jl` (200 ps restarts
  from the 17 ns state), `examples/plot_disk40_maxtorque.py`.

**Averaging convention.** mumax3's table `m` is the fill-weighted magnet average,
Σ m·geom / Σ geom (`engine/average.go`, `sAverageMagnet`); `m.Region(r).Average()`
is the unweighted mean over the region's cells. JuliaMag's `average_magnet` and
`average_region` reproduce the two. They coincide on a staircase edge (EdgeSmooth 0:
every code-to-code comparison below), but not with EdgeSmooth > 0, where only
`average_magnet` is the mumax3 table value. The EdgeSmooth-4 `gyro_lean.jl` tables
(`stt_es4_dt25.txt`) use `average_region`: their frequencies (used only for the dt
test and the EdgeSmooth shift) are unaffected, their amplitudes differ slightly from
a fill-weighted average.
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

The vortex-core positions in these tables differ by 2–4% between the codes;
frequencies are therefore taken from ⟨m⟩. The cause was two bugs in JuliaMag's
`vortexcore`, both fixed after these tables were produced: its sub-cell interpolation
read `f1 - 2f0 + f2`, where Julia parses `2f0` as the Float32 literal 2.0 (offsets
shrunk ~10×), and it searched only the bottom layer, while mumax3's `ext_corepos`
takes the max |m_z| over all layers. The 40 ns series uses the fixed tracker.

**40 ns series** (EdgeSmooth 4, Slonczewski only; tables `jm_es4_40ns.txt`,
`mumax_stt_es4_noZL_40ns.txt`, `mumax_stt_es4_noZL_40ns_maxerr1e-7.txt`):

| quantity | JuliaMag vs mumax3 |
|---|---|
| frequency 1–40 ns (Hilbert x/y; orbit phase) | 962.28 / 962.318 MHz, identical |
| frequency, every 2 ns window (961.9–964.1 MHz) | ≤ 0.001 MHz |
| ringdown rate of the ⟨m⟩ amplitude | Γ = 0.0806 ns⁻¹ both |
| max \|Δ⟨m⟩\| (t < 0.5 ns transient) / after 10 ns | 4×10⁻⁵ / < 10⁻⁶ |
| E_total, E_exch / E_demag (max rel.) | 2.5×10⁻⁵ / 2×10⁻⁴ |
| core position (fixed tracker) | 0.006 nm rms |
| maxTorque vs mumax3 MaxErr 1e-7 | 0.2% rms |

With p in the plane the current displaces the core equilibrium to (−6.35, −1.06) nm
and the gyration rings down (orbit radius 6.0 nm at 2 ns → 0.31 nm at 40 ns): a
damped gyration, not an auto-oscillation (damping compensation needs p_z ≠ 0).

**maxTorque and mumax3's tolerance.** With the default `MaxErr = 1e-5`, mumax3's
maxTorque follows JuliaMag until ~13 ns, then plateaus near 13 mT while JuliaMag keeps
decaying. This is integration error in mumax3, shown by:
1. JuliaMag's torques on mumax3's own m (12 and 17 ns) agree cell by cell: B_eff 2×10⁻⁷,
   LL torque 1×10⁻⁵, STT 2×10⁻⁷, total 3×10⁻⁵ relative, same max cell.
2. Near the core both codes have the same max torque (1.02×10⁻² T at 17 ns); mumax3's max
   lies in the vortex body (r ≈ 35 nm, where m ≈ −p).
3. Restarted from mumax3's 17 ns state for 200 ps: default RK45 keeps 13.06 mT; RK45
   `MaxErr = 1e-7` → 10.25 mT; fixed RK4 50 fs → 10.25 mT; JuliaMag → 10.25 mT (body
   torque decays in ~5 ps ≈ 1/(α ω_max)). `MaxErr = 1e-7` uses the same dt as the default,
   so the tolerance, not stability, sets the residual (~10⁻⁴ in m, short-wavelength
   exchange modes).
4. The full 40 ns mumax3 run with `MaxErr = 1e-7` agrees with JuliaMag's maxTorque to 0.2%
   rms (0.5% max before 13 ns, where the default run deviated by up to 24% too).

**Estimator note.** `gyrofreq.freq_phase` defaults to `period = 1.08e-9` for its
moving-average detrend. On the 6 ns windows of the table above, using 1/f_FFT instead
moves the frequencies by ≤ 0.12 MHz, identically in both codes; on 2 ns windows the
Hilbert fit is unusable, hence the orbit-phase method in `plot_disk40_compare.py`.

## Physical model

The Zhang–Li term with a perpendicular current acts only through ∂z m across the
thickness and shifts the frequency by −4.4%. The standard micromagnetic model of the
current-perpendicular vortex oscillator is the Slonczewski torque plus the Oersted
field (Khvalkovskiy et al., Phys. Rev. B 80, 140401(R) (2009)); the Zhang–Li term was
present in the reference only because it is mumax3's default. The JuliaMag result
we report is therefore **Slonczewski only**; the Zhang–Li run is a code-to-code check.
Neither code includes the Oersted field here, so the absolute frequency is a model
result for validation, not a device prediction.
