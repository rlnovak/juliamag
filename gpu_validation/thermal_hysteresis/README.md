# Vortex hysteresis of the 500 nm disc at finite temperature

The zero-temperature loop does not determine where the vortex renucleates on the
return branch: the state at −15 mT is mirror-symmetric, and round-off or a 10⁻⁶
perturbation decides the outcome (`../revalidation/README.md`). Here the symmetry is
broken physically, by the thermal field.

**Driver.** [`examples/disc_hysteresis_thermal.jl`](../../examples/disc_hysteresis_thermal.jl).
- Same disc, mesh and field sweep as `examples/disc_hysteresis.jl`: 0 → +150 → −150 → +150 mT along x, 5 mT steps.
- At each field step, LLG–Langevin dynamics for a fixed dwell replaces the minimizer.
- Integrator: Heun with a fixed step, GPU, Float32.
- α = 0.5 (quasi-static protocol). α sets the relaxation rate, not the equilibrium, and the thermal field uses the same α.
- Ms and A keep their room-temperature values.
- Logged per field step: ⟨m⟩ at the end of the dwell and averaged over its last quarter.

## Phase 1: dwell and time step (`phase1/`)

Virgin branch (0 → 150 mT, 31 steps), 50 K, seed 1, labmac03, commit `309980f`.
`python phase1/plot_phase1.py` prints the table and draws `phase1/phase1_convergence.png`.

| dwell, dt | max \|⟨mx⟩ − T=0\|, B ≤ 50 mT | annihilation | wall (31 steps) |
|---|---|---|---|
| 0.5 ns, 1e-13 s | 0.069 | 65 mT | 553 s |
| 1 ns, 1e-13 s | 0.019 | 65 mT | 1 112 s |
| 1 ns, 5e-14 s | 0.020 | 65 mT | 2 293 s |
| 2 ns, 1e-13 s | 0.008 | 60 mT (as T = 0) | 2 241 s |

- **Dwell.** 0.5 and 1 ns lag the T = 0 branch at low field. The gyrotropic relaxation at α = 0.5 takes a large part of the dwell. They also delay the annihilation by one field step.
- **2 ns** reproduces the T = 0 reversible branch within the thermal noise and annihilates at 60 mT, as T = 0. Phase 2 uses it.
- **Time step.** dt = 1e-13 s and 5e-14 s agree within the noise.
- **Saturated branch.** ⟨mx⟩ lies 0.0013 below T = 0 (spin waves).

## Phase 2: ensemble (`phase2/`, done 2026-11-23)

`run_phase2.sh` on labmac03, launched 2026-11-22 00:12, started 00:21.
- **Runs.** Full loops, dwell 2 ns, dt 1e-13 s.
  - 8 realizations at 50 K: seeds 1–8.
  - 8 realizations at 300 K: seeds 101–108, so they share no noise with the 50 K runs.
- **Chains.** Two, one per GPU, behind `run_when_gpu_idle.sh`.
  - Chain A runs the 300 K dt check first, then the odd seeds.
  - Chain B runs the even seeds.
  - About 3.2 h per loop, about 25 h of wall time.
- **Smoke test.** 2 field steps per run. The 50 K, seed 1 run reproduced phase 1 (dwell 2 ns) to all printed digits, so the GPU random stream is deterministic for a given seed.
- **Note.** The code tree on labmac03 comes from `git archive` run on Windows, so its files have CRLF line ends. Julia reads them, bash does not, so `run_phase2.sh` runs an LF copy of the waiter.

**300 K time-step check** (`phase2/check_dt.py`). Virgin branch, dwell 1 ns, seed 1, dt 1e-13 vs 5e-14 s.
- **Saturated branch (≥ 70 mT).**
  - Mean difference +8×10⁻⁵, mean |difference| 1.9×10⁻⁴, positive in 10 of 17 rows. That is 1.3 standard errors: no discretization bias.
  - The thermal reduction of ⟨mx⟩ relative to T = 0 is 0.0076 at 300 K and 0.0013 at 50 K. The ratio is 6.09, against 6 for a reduction linear in T, as expected for thermal spin waves at these temperatures.
- **Vortex branch (≤ 50 mT).** max |difference| 0.009, of the order of the thermal fluctuation of ⟨my⟩ (~0.02–0.04).
- **60 mT.** Both runs annihilate during this step, at different random times (⟨mx⟩ 0.896 vs 0.853). Both are saturated at 65 mT.

dt = 1e-13 s is therefore used at both temperatures.

**Run time.** All 16 loops ended with rc = 0 (chain B at 06:25, chain A at 07:33 on
2026-11-23). From 2026-11-22 12:53 another project's mumax3 queue shared both GPUs,
so some loops took up to 5.5 h instead of 3.2 h. Only the timing was affected.

### Results

`python phase2/plot_phase2.py` prints the per-loop table and the ensemble
statistics, and draws `phase2/phase2_loops.png` (also
`paper/figures/vortex_hysteresis_thermal.png`).

Definitions used by the script:
- **B_sw:** the field step with the largest jump of ⟨mx⟩ on a branch.
- **State:** "vortex" when ⟨mx⟩ stays in the vortex range for at least two steps.
  The ⟨mz⟩ of the core (~±1–5×10⁻³) confirms it.
- **B_ann:** the first field after the switch where ⟨mx⟩ saturates (|⟨mx⟩| > 0.95).

| | 50 K (N = 8) | 300 K (N = 8) | T = 0 deterministic |
|---|---|---|---|
| virgin annihilation | 61 ± 2 mT | 60 mT (all) | 60 mT |
| descending switch | −15.0 ± 3.8 mT | −13.1 ± 5.9 mT | −15 mT, direct reversal |
| ascending switch | +15.0 ± 3.8 mT | +11.9 ± 4.6 mT | +15 mT, vortex |
| switches through a vortex | 16/16 | 16/16 | 1/2 |

**Saturated branch.** The thermal reduction of ⟨mx⟩ relative to T = 0 is 0.0013
(50 K) and 0.0076 (300 K), a ratio of 6.1: linear in T.

**Symmetry breaking before the switch.** In 28 of the 32 branches, |⟨my⟩| grows to
0.15–0.48 before the switch: a C-shaped state that the mirror-symmetric T = 0
calculation cannot reach. Its remanence is ⟨mx⟩ = 0.70–0.86, against 0.961 at T = 0.
In the other 4, ⟨my⟩ ≈ 0, a symmetric precursor.

**Two classes of nucleated vortex:**
- **Equilibrium vortex** (21/32): ⟨mx⟩ follows the virgin reversible branch
  (0.44 at 20 mT, 0.51 at 25 mT, …), and the vortex is annihilated at 60–65 mT.
- **Short-lived vortex** (11/32): ⟨mx⟩ just after nucleation lies well above that
  branch (0.69–0.73 at 20 mT), and the vortex is expelled at 25–35 mT. All 11 come
  from a C state, but C states also lead to the first class (17 cases); the 4
  symmetric precursors all give the first class.
- The likely explanation is a core that nucleates far from the centre and leaves
  before relaxing. Confirming it needs m snapshots. The GPU random stream is
  deterministic, so any loop can be re-run with OVF output: about 2.5 h to reach
  the switch.

**Temperature dependence.** The 50 K vs 300 K differences in the switching fields
are within the standard error of the mean (1.3–2.1 mT). The finite-T loops differ
from the T = 0 one in kind (a vortex every time, mostly after a C precursor), not
measurably between 50 and 300 K.

## Caveats for the paper

- The sweep rate, 5 mT per 2 ns = 2.5×10⁶ T/s, is many orders of magnitude above an experiment's. The switching fields are therefore upper bounds on thermally activated switching.
- Ms(T) and A(T) are not reduced at 300 K.
