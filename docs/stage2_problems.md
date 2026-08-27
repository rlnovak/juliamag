# Stage-2 test problems (from magnum.np demos)

A second round of validation problems, adapted from the
[magnum.np](https://gitlab.com/magnum.np/magnum.np) `demos/` set, to exercise
JuliaMag beyond the µMAG standard problems. Each entry says whether JuliaMag can
run it today and, if not, what feature it needs.

Run them locally / on labmac04 with `examples/stage2_local.jl`, and on Colab via
the "Stage 2" section of `colab/juliamag_gpu_benchmark.ipynb`.

## How to run

**Locally** (from the repo root; the `examples` environment has CairoMakie):

```
julia --project=examples examples/stage2_local.jl
```

**On labmac04** (over SSH; Julia at `~/opt/julia-1.11.9`, repo at `~/mumag`):

```
ssh rlnovak@labmac04.blumenau.ufsc.br
cd ~/mumag
~/opt/julia-1.11.9/bin/julia --project=. examples/stage2_local.jl
```

(The `examples` env is not set up on the remote; run under `--project=.`, which
already has CairoMakie/DelimitedFiles from the GPU work.)

**On Colab**: run the notebook's "Stage 2" section (it calls the same script).

Each problem writes a tab-separated `.txt` table and a `.png` figure into
`examples/`: `stage2_langevin.*`, `stage2_depinning.*`, `stage2_dmi_spiral.*`.

## Portable now (JuliaMag has the physics)

| magnum.np demo | JuliaMag problem | uses |
|----------------|------------------|------|
| `langevin` | Finite-T hysteresis of a macrospin ensemble: sweep field at several temperatures, record ⟨mx⟩(H,T) | `runthermal!`, `setexternalfield!` |
| `sp_domainwall_pinning` | Domain-wall depinning at a soft/hard interface: two regions, ramp the field, find the depinning field | `RegionParams`, `setregion!`, `relax!` |
| `sp_DMI` | 1D equilibrium with interfacial DMI + PMA: relax a chain, record the m(x) profile | `Dind`, `Ku`, `relax!` |
| `dmi_bulk`, `dmi_interface` | 1D DMI spin-spiral period vs D | `Dbulk`/`Dind`, `relax!` |
| `sp4`, `sp5`, `minimizer` | already covered by the standard-problem section | — |

## Needs a new feature (added to the roadmap)

| magnum.np demo | missing feature | note |
|----------------|-----------------|------|
| `sot` | **Spin-orbit torque (damping-like + field-like)** with `eta_damp`/`eta_field` | JuliaMag has Slonczewski (Λ, ε′) but not the separate SOT coefficients; distinct torque model |
| `sp_FMR`, `dispersion` | **Spin-wave / FMR spectral analysis** (ringdown FFT, dispersion ω(k)) | needs a driver that FFTs m(t) / m(x,t); the dynamics exist, the analysis layer does not |
| `eigensolver` | **Eigenmode solver** (linearized LLG normal modes) | new linear-algebra layer |
| `energy_barrier_ellipse` | **Energy-barrier / string (NEB) method** | minimum-energy-path solver |
| `rkky`, `rkky_biquadratic1/2` | **RKKY interlayer exchange coupling** | inter-region surface exchange term |
| `linear_elasticity` | **Magnetoelastic coupling / elasticity solver** | new physics module |
| `self_induction` | **Eddy currents / self-induced fields** | new physics module |
| `softmagnetic_composite` | RKKY + advanced multiregion | partial; depends on RKKY |
| `sot` diagnostics, `rux`, `timings`, `inverse_*` | miscellaneous / inverse-problem tooling | out of scope for validation |

These are recorded in the project roadmap as future work; the SOT model and the
FMR/dispersion analysis are the most commonly requested and are the natural next
additions.

## Files

- `examples/stage2_local.jl` — runs the portable problems, saving a table and a
  figure for each (local and labmac04).
- `colab/juliamag_gpu_benchmark.ipynb` — "Stage 2" section runs the same on Colab.
