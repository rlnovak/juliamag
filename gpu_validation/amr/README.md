# Currents from anisotropic magnetoresistance: validation

`src/conduction.jl` (`setconduction!`, `setresistivity!`, `addcontact!`, `solveconduction!`,
`q_contacts`, `q_resistance`, `potentialmap`, `currentmap`), 2026-11-26. Manual, section 4.

The potential solves ∇·(σ(m)∇φ) = 0 with σ = ρ⁻¹, ρ = ρ⊥ I + (ρ∥ − ρ⊥) m mᵀ, by finite volumes on the
cells (BiCGSTAB, preconditioned by the isotropic problem); j = −σ∇φ feeds the spin torques.

## Against an independent finite-element solution

`fem_reference.py` solves the same problem with linear finite elements on triangles (scipy):
- a rectangle 400 × 80 nm², 4 nm thick;
- ρ∥ = 210, ρ⊥ = 200 nΩ m;
- contacts at 1 V and 0 V on the short sides.

`juliamag_texture.jl` solves it with JuliaMag on 50 × 10 to 400 × 80 cells. `compare_texture.py`
compares the two; its output is in `results/texture.txt`.

| texture | JuliaMag, extrapolated | FEM, extrapolated | difference | order (JuliaMag, FEM) |
|---|---|---|---|---|
| domain wall at an angle, m in plane, σ_xy changes sign | 261.553088 Ω | 261.553087 Ω | 6e-9 | 1.9, 2.0 |
| uniform m at 45°, largest planar Hall field, shorted at the contacts | 256.233452 Ω | 256.233451 Ω | 4e-9 | 1.7, 1.8 |

- For the domain wall, the potential at the cell centres agrees with the FEM to 1.2e-5 V on 100 × 20
  cells and 2.9e-6 V on 200 × 40 (of 1 V).
- The order below 2 for uniform m comes from the corners where the contacts meet the insulating
  sides; both methods show it.
- On a contact face the flux uses the tangential gradient of the cell next to it. The face is an
  equipotential, so ∂_eφ = 0 there, and dropping that term looks natural, but it doubles the error
  for uniform m (256.233422 against 256.233438 Ω on 400 × 80 cells).
  - With ∂_eeφ = 0 on the face, the equation gives σ_dd ∂_ddφ = −2σ_de ∂_deφ.
  - The cell-centre gradient, (h/2) ∂_d∂_eφ, therefore cancels the O(h) error of the one-sided
    normal derivative.

## A vortex between two leads: the AMR effect on the dynamics

`vortex_amr.jl` sets up a Permalloy disc, 128 nm × 10 nm, on 64 × 64 cells:
- non-magnetic leads 40 nm wide, with contacts at their ends: 2 mA in, 0 V out;
- the Zhang-Li torque (P = 0.4, ξ = 0.05), no Oersted field;
- 3 ns from the relaxed vortex, for Δρ/ρ⊥ = 0, 0.02, 0.04.

With Δρ/ρ⊥ = ε, the conduction changes j by O(ε), so the effect on R and on the core must be linear
at small ε. `compare_vortex.py` checks this (`results/vortex.txt`, `vortex.png`). The runs took
labmac03, 1 CPU thread, 5 to 8 min each.

| | Δρ/ρ⊥ = 0.02 | 0.04 | ratio (linear: 2) |
|---|---|---|---|
| R(0) − R(0)\|ε=0 | 0.298215 Ω | 0.593176 Ω | 1.989 |
| peak-to-peak of R(t), t > 0.5 ns | 1.22e-3 Ω | 2.40e-3 Ω | 1.97 |
| largest shift of the core by AMR | 0.411 nm | 0.786 nm | 1.91 |

- The remainder of the core shift, max |d(0.04) − 2 d(0.02)| = 0.037 nm, is 5 % of d(0.04): of
  order ε, as expected for the next term.
- With ε = 0 the current is still non-uniform: it follows the shape of the disc and the leads.
- The run took 0–1 BiCGSTAB iterations per solve, warm started.

## Code checks

**CPU: `test/test_conduction.jl`, 61 tests.**
- Strip, analytic:
  - R of the isotropic strip;
  - for uniform m at five angles, E_x/j_x = ρ⊥ + Δρ cos²θ and E_y/j_x = Δρ sinθ cosθ, to 1e-7;
  - the current through every cross-section;
  - m → −m.
- Contacts:
  - potential and current contacts, the ground, functions of time;
  - contacts at nearly the same potential.
- Spin torques: from the computed j, the same torque as the same j given as a current map.
- Float32 world: solved in Float64, no charge lost.
- Islands:
  - a floating 2 × 2 island, and a cell that touches the conductor only at a corner, become
    insulators with a warning;
  - a contact cut off from the potential contacts is an error.
- Contacts fed currents only: currents that do not sum to zero, or a single current contact, are
  errors.
- Changes of the regions after the first solve are taken into account.
- A lead without Msat is isotropic even where m ≠ 0, and contacts carry no current density.
- Partially filled cells: a strip 10.5 cells wide (edge smoothing) gives the exact R, and the
  current density of the material in its edge cells.
- 3D, three cells thick: m out of plane gives ρ⊥; for m = (1,1,1)/√3, E_z/j_x = Δρ/3.
- Simulation: R(t) in the table, the potential and current maps.

**GPU: `gpu_validation/test_gpu.jl` (`conduction_terms`).** The conduction is solved on the host for a
GPU World too. The check uses a three-region film with a time-dependent current contact and compares
the torques and 50 RK4 steps against the CPU:
- Float64: 4e-16 and 8e-15;
- Float32: 2e-7 and 4e-7.

## Corrected on the way

An independent review of the code found these problems, all fixed and covered by the tests above:
- **Float32 worlds solved in Float32.** The residual stalled near 1e-6, which lost 0.2 % of the charge
  and moved R by 4e-5 in a sawtooth. Now the conduction is always solved in Float64 on the host.
- **Parts of the conductor with no contact.** They went unnoticed, or failed in the Cholesky
  factorization. Now a connectivity search at setup catches them.
- **Partially filled cells conducted as full ones.** This gave a 12.5 % error in R for an
  edge-smoothed strip.
- **Changes of the regions after the first solve were ignored.**
- **Convergence scale.** It counted the contact and insulator cells, and the largest |V| instead of
  the spread of the contact potentials.
- **Inconsistent contact currents were silently accepted.**
- **A lead without Msat was anisotropic if m ≠ 0 there.**

The review also proposed dropping the tangential term on the contact faces. That was tested and
reverted (see above).
