# JuliaMag ↔ mumax3 benchmark cross-check (components + energies)

Column-by-column comparison of JuliaMag against mumax3 3.11.1 on the µMAG
standard problems, logging **every output** — `mx, my, mz` **and** the four
energy terms (`E_demag, E_exch, E_Zeeman, E_total`) — over the full 1 ns run, to
localize any divergence rather than only checking the final magnetization.

mumax3 was run on labmac04 (`~/mumax3/mumax3.11.1_linux_cuda12.0`); JuliaMag was
run from this repo at the exchange-edge-fill fix (commit `bad1f2c`).

## Files

- `std4.mx3`, `std5.mx3` — mumax3 input scripts (with `TableAdd` of every energy).
- `std4_energies.jl`, `std5_energies.jl` — matching JuliaMag drivers logging the
  same columns.
- `std4_mumax_table.txt`, `std4_juliamag_table.txt` — Std4 tables.
- `std5_mumax_table.txt`, `std5_juliamag_table.txt` — Std5 tables.
- Figures: `examples/stdproblem4_energy_compare.png`,
  `examples/stdproblem5_energy_compare.png` (matplotlib, per-column rms/scale).

## Results (rms diff over 1 ns, as % of each column's rms scale)

| column   | Std4  | Std5   |
|----------|-------|--------|
| mx       | 0.007%| 0.000% |
| my       | 0.028%| 0.001% |
| mz       | 0.027%| 0.001% |
| E_demag  | 0.018%| 0.000% |
| E_exch   | 0.042%| 0.000% |
| E_Zeeman | 0.007%| 0.000% |
| E_total  | 0.002%| 0.000% |

Both problems agree to well under 0.05% on every component and every energy.

## Why these don't test the edge-fill fix

Std4 (500×125×3 nm film) and Std5 (100×100×10 nm square) are **rectangular,
fully-filled** geometries: every cell has fill = 1, so the exchange edge-fill bug
fixed in `bad1f2c` never activates. Their near-perfect agreement confirms two
things: (1) the core physics — demag, exchange, Zhang-Li STT, and all the energy
functions — matches mumax3; (2) the exchange fix did **not** regress the
fully-filled case.

The discrepancy that motivated the fix is specific to **curved geometry with
fractional edge fill** (the 128 nm disk), where mumax3 scales the moment by the
cell fill only as a demag source and divides the exchange by the full region
Msat. That case was validated separately on labmac04 (2026-11-12): field by
field (B_demag/B_exch to 7.7e-7/2.3e-7 relative) and in the dynamics, where the
like-for-like vortex gyration frequencies (from ⟨m⟩) agree to < 0.05 MHz
(930.8 MHz with Slonczewski + Zhang–Li, 973.5 MHz with Slonczewski only, 969.8 MHz
free gyration, in both codes). Scripts and tables: `gpu_validation/disk_stt/`. See `docs/roadmap.md`, "Validation follow-up:
single-disk STT vortex frequency".
