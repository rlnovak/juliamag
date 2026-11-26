# Fields and currents that vary in space and time: validation

`src/excitation.jl` (`fieldmap`, `regionmap`, `addfield!`, `addcurrent!`), 2026-11-26. Manual, section 4.

## Against mumax3 3.11

The same four problems in both codes, on labmac03 (Quadro P1000, Float32, adaptive RK45, MaxErr 1e-5):
`mumax3/*.mx3` and `fields_juliamag.jl`, run by `run_fields.sh`. `compare.py` interpolates
JuliaMag's rows at mumax3's times; differences relative to the largest |value| of the column. Full
output in `results/compare.txt`, figure `compare.png`.

| problem | mumax3 | ⟨m⟩ rms | E_total rms |
|---|---|---|---|
| Gaussian pulse in space and time on a vortex, 2 ns | `B_ext.add(mask, f(t))` | 3e-4 | 2e-6 |
| field per region, functions of time, film, 1 ns | `B_ext.SetRegion(r, v(t))` | 1e-4 – 5e-4 | 1.4e-4 (from t > 0) |
| Zhang-Li current in one half, standard problem 5 geometry, 1 ns | `J.SetRegion` | 3e-6 – 9e-6 | 3e-7 |
| Slonczewski current through a nanocontact in a disc, 2 ns | `J.SetRegion` | 2e-6 – 1e-5 | 5e-7 |

- In "regions" the first row (t = 0) differs: mumax3 still shows the field it had before the
  assignment (its function of time is used only once the time changes), JuliaMag the new one.
  From the second row the applied field agrees to 2e-3 of its largest value (rows at different
  times of an oscillating field), the Zeeman energy to 1e-3.
- The first two problems oscillate fast: their differences come from the rows falling at different
  adaptive times and from the single-precision relaxation of the start states (each code its own).

## Code checks

- CPU: `test/test_excitation.jl`, 35 tests. They cover:
  - a uniform map bit for bit equal to the uniform field;
  - the energies against a direct sum, with regions and partially filled cells;
  - the field against the gradient of the energy;
  - the order of the solvers for a pulse;
  - the current of each cell, and the spin-torque parameters per region against single-material worlds.
- GPU: `gpu_validation/test_gpu.jl` (`excitation_terms`), 24 checks. They cover the effective field,
  the applied field, `energy!`, the Zeeman energy, the torque with currents per cell and 100 steps
  with fields and currents in time. In Float64 the GPU agrees with the CPU to 1e-15, in Float32 to 5e-7.

## Corrected on the way

- An independent review found that a current per region used the spin-torque parameters (Msat,
  pol, xi, lambda, ε′) of one material everywhere: the torque in a region with other parameters was wrong.
  With a current map, or whenever these parameters differ between regions, each cell now takes those of
  its region, as mumax3.
- `runthermal!` and the old `Solver` silently left out the spin torques of a current; they now refuse it.
