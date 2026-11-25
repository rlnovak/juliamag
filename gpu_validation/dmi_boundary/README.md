# DMI at free boundaries: JuliaMag against mumax3

2026-11-24. mumax3 3.11.1 on labmac03 (single precision), JuliaMag on the CPU in Float64.

    python make_mx3.py DIR                                     # 16 mumax3 scripts
    (run them: mumax3 -f DIR/*.mx3)
    julia --project=. gpu_validation/dmi_boundary/compare_mumax3.jl DIR

Every case evaluates exchange + DMI on the random state that mumax3 wrote (`m.ovf`), no
relaxation: field `B_exch` and energy `E_exch`. Full output: `results.txt` (present code),
`results_old_code.txt` (commit 161641a, the negative control; it has no openbc option, so
its openbc rows use the clamped rule).

| case group | largest field difference / largest field | same, old code |
|---|---|---|
| open and periodic boxes, interfacial and bulk, OpenBC false/true, Nz = 1 | ≤ 1.5e-7 | 1.3e-2 – 4.0e-2 in the edge cells |
| periodic in x and y, interfacial (no free edge) | ≤ 8.4e-8 | ≤ 8.4e-8 |
| disc and ellipsoid with staircase edge | ≤ 1.3e-7 | 4.7e-3 – 3.3e-2 |
| disc with edge smoothing 4: partial cells | ≤ 1.3e-7 | **0.91 – 0.99** |
| two regions, different A and Msat, same D | ≤ 1.1e-7 | 2.2e-2 |
| two regions, **different D** | **9.0e-3** | 2.2e-2 |

Energies agree to 8.8e-8, except in the last row (−4.4e-5).

GPU: `test_gpu.jl` on labmac03 (2026-11-24), 868/868. The DMI kernels equal the CPU to
≤ 2.8e-16 (Float64) and ≤ 1.5e-7 (Float32) of the largest field, with both boundary rules,
which differ by 7–21 % of the largest field in the cases with free edges that were checked.

- Partial cells: the old code divided the DMI by Msat·fill; mumax3 divides by Msat (as
  exchange! did already). `anisotropy!` and `thermalfield!` still divide by Msat·fill.
- Different D: mumax3 uses a mean of the two cells on their bond, JuliaMag the D of the
  cell. Not changed.
- mumax3's default (OpenBC false) applies the chiral condition to both z faces of a bulk
  DMI film of one cell. `openbc = true` is the rule of MicroMagnetic.jl, which has no
  boundary term.
