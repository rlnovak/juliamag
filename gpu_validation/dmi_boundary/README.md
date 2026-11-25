# DMI (free boundaries, regions) and anisotropy: JuliaMag against mumax3

2026-11-24. mumax3 3.11.1 on labmac03 (single precision), JuliaMag on the CPU in Float64.

    python make_mx3.py DIR                # 28 mumax3 scripts and their .params (new DIR)
    (run them: mumax3 -f DIR/*.mx3)
    julia --project=. gpu_validation/dmi_boundary/compare_mumax3.jl DIR

Every case evaluates the fields of the random state that mumax3 wrote (`m.ovf`), no
relaxation: `B_exch` (exchange + DMI) and `E_exch`; in the `ani_*` cases also `B_anis` and
`E_anis`. Outputs, in order of the code:

- `results_old_code.txt`: 161641a, before the chiral boundary condition (16 cases, older
  output format; no openbc option, so its openbc rows use the clamped rule).
- `results_cf87368.txt`: chiral boundary condition, but the D of the cell between regions
  and the anisotropy divided by Msat·fill. Negative control of the last two steps.
- `results.txt`: present code.

Largest field difference over the largest field:

| case group | present | cf87368 | 161641a |
|---|---|---|---|
| open and periodic boxes, interfacial and bulk, OpenBC false/true, Nz = 1 | ≤ 1.5e-7 | ≤ 1.5e-7 | 1.3e-2 – 4.0e-2 in the edge cells |
| periodic in x and y, interfacial (no free edge) | ≤ 8.4e-8 | ≤ 8.4e-8 | ≤ 8.4e-8 |
| disc and ellipsoid with staircase edge | ≤ 1.3e-7 | ≤ 1.3e-7 | 4.7e-3 – 3.3e-2 |
| disc with edge smoothing 4: DMI in partial cells | ≤ 1.3e-7 | ≤ 1.3e-7 | **0.91 – 0.99** |
| two regions, different A and Msat, same D | ≤ 1.1e-7 | ≤ 1.1e-7 | 2.2e-2 |
| regions of different D (same sign, opposite, zero; 10 cases) | ≤ 9.8e-8 | 7.8e-3 – 2.3e-2 at the interface | – |
| anisotropy, full cells | ≤ 1.0e-7 | ≤ 1.0e-7 | – |
| anisotropy, disc with edge smoothing 4, partial cells | ≤ 8.4e-8 | **9.3 and 14** (times the largest field) | – |

Energies agree to 9.1e-8 (present code). In cf87368 E_anis was off by 12–13 % on the
smoothed discs, and E_exch by up to 1.7e-4 with different D. (`ind_regD`, the only case of
different D in the 161641a file, has 2.2e-2 there.)

GPU (`test_gpu.jl` on labmac03): cf87368, 868/868; present code, 1076/1076. The DMI and
anisotropy kernels equal the CPU to ≤ 2.9e-16 (Float64) and ≤ 1.6e-7 (Float32) of the
largest field, also with regions side by side (D of equal, opposite sign, zero) and
edge-smoothed cells. A first run failed one check, the last adaptive RK45 step in Float32
(bilayer, 2.1 % against 1 %): the CPU run itself moves by up to 2.7 % under one-ulp
perturbations of the initial state (1.9 % in cf87368), so that tolerance is now measured.

- Partial cells: exchange, DMI and anisotropy fields divide by the region's Msat, as in
  mumax3. `thermalfield!` divides by Msat·fill, the moment of the cell
  (fluctuation–dissipation); mumax3 uses Msat. Kept by decision of the author.
- mumax3's default (OpenBC false) applies the chiral condition to both z faces of a bulk
  DMI film of one cell. `openbc = true` is the rule of MicroMagnetic.jl, which has no
  boundary term.
