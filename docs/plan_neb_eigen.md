# Implementation plan: GNEB energy barriers and linear eigenmodes

Written 2026-11-23. Work starts after the finite-temperature vortex hysteresis
(`gpu_validation/thermal_hysteresis/`) has finished and the paper and code have been
updated with it (see `docs/roadmap.md`, "Work queue"). The two features are
implemented in this order: GNEB first, eigenmodes second. The eigenmode operator
also certifies the GNEB saddle points (one negative curvature).

Common rules, as for every JuliaMag feature:
- Write each piece at the level of the mathematics, CPU first, then GPU through
  the same generic code.
- Validate against an analytic result and against a second code.
- Add a paper section and tests.
- Review campaign scripts before long runs.

---

## 0. Shared prerequisite: energy and linear field on any array

Both features need, for every evaluation:
- the total energy;
- the effective field;
- for the eigenmodes, the field produced by a small deviation δm.

Today the energies are CPU loops (`src/energy.jl`, read back from the GPU for the
table columns).

**Energy from the effective field.** Every term except Zeeman is quadratic in m:
exchange, demag, uniaxial anisotropy, both DMI forms. For a quadratic term,
E_term = −½ V Σ Msat m·B_term. Zeeman is linear: E_Z = −V Σ Msat m·B_ext. Hence

    E_total = −½ V Σ_cells Msat (m · (B_eff + B_ext))

needs only the effective field already computed for the dynamics, plus one
reduction.

Uniaxial anisotropy enters as −Ku (m·u)². This differs from Ku(1 − (m·u)²) by a
constant, which cancels in barriers. It is the same convention as mumax3 and as the
present `anisotropyenergy`. A test pins this down.

**New functions:**
- `energy!(B, m, w) -> E`: computes B_eff (`effectivefield!`) and returns E_total.
  - Generic reduction with Float64 accumulation.
  - CUDA method: one dot-product reduction kernel, the same pattern as
    `_k_accmaxnorm2!` but a sum: a block reduction, then one atomic add per block in
    Float64, or a second pass.
  - Validated against the per-term CPU energies to rounding.
- `linearfield!(B, δm, w)`: the field of the quadratic terms only,
  B_lin(δm) = B_eff(δm) − B_ext. This is the same code path with Bext and the
  thermal field skipped.
  - The deviation δm is not a unit vector; every field term is linear in its
    argument, so no change is needed.
  - Test: B_lin(a δ1 + b δ2) = a B_lin(δ1) + b B_lin(δ2), and
    B_eff(m) = B_lin(m) + B_ext.
- Optional: move `clearempty!` to the GPU (backlog item), needed if images are
  created on the device.

**Tests:**
- CPU: energy! matches totalenergy to 1e-12 relative, on every validation geometry
  (open/periodic, regions, DMI).
- GPU: test_gpu checks energy and linear field against the CPU.

**Status, 2026-11-23: DONE.**
- `energy!`, `linearfield!` and `isconservative` are in place, with
  `test/test_energy_gradient.jl` (42 tests) and checks in `test_gpu.jl`.

**Finding: the gradient check exposes three non-conservative discretizations.** Only
the first is inherited from mumax3 (corrected 2026-11-24; see item 2). The central-difference test, which is exact
for a quadratic E, gives a relative error of about 1e-13 for exchange, anisotropy,
demag (also with fractional cells), Zeeman, and region interfaces. It fails by
2–6 % in cases 1 and 2:
1. **Exchange on edge-smoothed cells.** The field uses the region's Msat, but the
   energy weights it by Msat·fill, so the operator F·L is not symmetric.
   - Checked in mumax3's source: `AddExchange` divides by Msat, while
     `GetExchangeEnergy` uses M_full = Msat·geometry·m.
2. **DMI next to an open boundary** (or, by the same mechanism, an empty cell). The
   cause is JuliaMag's own simplification, not mumax3.
   - `src/dmi.jl` clamps the missing neighbour (Neumann, ∂m/∂n = 0) in the central
     difference. That operator is not symmetric at the edge.
   - mumax3 extrapolates the missing neighbour with the chiral condition
     2A ∂m/∂n = …D. Exchange plus DMI at the edge bond then reduces to
     (D²/2A)(mx, 0, mz) for an x edge (up to the common 1/Msat), which is symmetric.
     Its `OpenBC` option (skip the bond) is symmetric too. mumax3 is therefore
     conservative here.
   - An earlier version of this note blamed mumax3's boundary condition: wrong.
   - The clamp also drops the chiral edge canting, so open DMI samples differ from
     mumax3 in the edge cells.
   - **Fixed 2026-11-24** (user's decision): `dmi!` now has mumax3's chiral boundary
     condition and, with `World(…; openbc=true)`, its OpenBC. Both are conservative;
     `isconservative` no longer flags open DMI samples.
   - Interfacial DMI with x and y periodic is exact.
3. **DMI with a D that differs between regions** (flagged by `isconservative` with item 2). `dmi!` used the
   central cell's D for both directions of a bond. **Fixed 2026-11-24** (user's
   decision): the bond takes mumax3's mean of the two D, which is symmetric, and
   `isconservative` now flags only item 1.

In case 1 the field drives the dynamics correctly. The reported energy, however, is
not the exact potential of that field.

**Consequence for GNEB:**
- GNEB needs an exact (E, ∇E) pair. It runs on `isconservative(world)` systems,
  which include every validation case below (the macrospin, an unsmoothed
  ellipse, a periodic skyrmion film), and warns otherwise.
- A later option is the symmetrized operator ½(L + Lᵀ). For the fill case, Lᵀ is
  the exchange of fill·m, which is cheap.

**Eigenmodes** are not affected: they linearize the dynamics, and the operator need
not be symmetric. The Krylov method must then be a general one, not Lanczos.

---

## 1. GNEB: minimum-energy paths and barriers

### Method

The geodesic nudged elastic band of Bessarab, Uzdin and Jónsson, Comput. Phys.
Commun. 196, 335 (2015), on the product of unit spheres. MicroMagnetic.jl
`src/neb/` is the reference port: clone it and read it before coding.

- **Chain.** Images m⁽⁰⁾ … m⁽ᴺ⁾. The end images are fixed and relaxed minima, made
  with the existing minimizer.
- **Distance.** Per cell, θ = atan2(|m₁×m₂|, m₁·m₂). The image distance is
  L = sqrt(Σ_cells θ²), summed over magnetic cells only (Msat > 0).
- **Tangent.** The energy-weighted upwind tangent of Henkelman and Jónsson (2000),
  built from the differences to the neighbouring images. Each cell's component is
  projected onto the tangent plane of m⁽ⁱ⁾, then normalized over the whole image.
- **Forces.**
  - G = projection on the tangent planes of −∂E/∂m = V Msat B_eff, in J per unit
    rotation.
  - NEB force on an interior image: F = G − (G·τ)τ + k (L₊ − L₋) τ.
  - Climbing image (the highest-energy one, switched on after the band has
    roughly converged): F = G − 2 (G·τ)τ, with no spring.
- **Optimizer.**
  - Velocity-projection optimization (VPO), as in Bessarab 2015: per-image
    velocity, projected on F, with a fixed time step.
  - The update is a rotation: the existing Cayley step on each cell, which keeps
    |m| = 1 exactly.
  - A second option, using the existing machinery: the Barzilai–Borwein step of
    the minimizer, applied to the whole chain with F as the descent direction. Its
    convergence is to be compared against VPO on the Stoner–Wohlfarth case.
  - Keep whichever is more robust; mumax3-style simplicity is preferred.
- **Initial path.**
  - Geodesic interpolation per cell (Rodrigues rotation about m₀×m₁, with a
    random axis perpendicular to m₀ when the two are antiparallel).
  - Optionally, user-supplied intermediate states, such as a skyrmion of
    decreasing radius.
- **Convergence.** max over images and cells of |F⊥|/(V Msat) < tol, in T,
  default 1e-4 T. A step cap and stagnation detection end the run.
- **Output.**
  - Energies E⁽ⁱ⁾ and the path coordinate.
  - Barrier ΔE = max E − E⁽⁰⁾, also in kT at a given T.
  - Saddle estimate by cubic Hermite interpolation, using E and the tangential
    force at each image.
  - Images saved as OVF (`m_image###.ovf`) and a table (`neb.txt`).

### Code

**`src/neb.jl`:**
- `struct GNEB{T,W,A}` holds the world, the vector of images (CPU or GPU arrays),
  the tangents, the forces, the velocities and the energies.
- `GNEB(world, mA, mB; nimages=16, spring=…, climbing=true)`.
- `relax!(neb; stopforce=1e-4, maxsteps=…)`.
- Accessors: `energies(neb)`, `barrier(neb)`, `pathcoordinate(neb)`,
  `saveimages(neb, dir)`.
- Simulation-level convenience: `neb(sim, mB; …)`.

**Kernels.** Every per-image operation (geodesic distance, tangent projection,
spring, Cayley update) is written as broadcasts and reductions, so it runs on
CuArrays through the existing extension. Dedicated kernels come only if profiling
asks for them.

**Memory.** N images × (m, B, τ, F, v) ≈ 5N fields. For the 64×64×8 disc with
N = 16 that is about 100 MB in Float64, which fits the 4 GB P1000.

### Validation

1. **Stoner–Wohlfarth macrospin** (one cell, uniaxial K, field along the easy
   axis). Exact ΔE = K V (1 − h)², with h = Msat B / (2K). Check over h = 0 … 0.8:
   relative error < 1e-6. The saddle must be m ⟂ u at h = 0.
2. **Elongated thin-film element**, magnum.np demo `energy_barrier_ellipse`.
   - Reproduce its geometry and compare the barrier with magnum.np: install it on
     labmac with the torch CUDA wheels, or use the value quoted by the demo.
   - Cross-check with MicroMagnetic.jl's NEB on the same mesh.
   - Report the barrier against the number of images (8, 16, 32) and the spring
     constant: the barrier must not depend on k.
3. **Skyrmion collapse** in an interfacial-DMI film. Compare the barrier with
   MicroMagnetic.jl on the same discretization and with published continuum
   values; cite and document any discretization caveat (collapse through a
   singular point).
4. **Saddle certification** (once item 2 exists): the Hessian at the climbing
   image has exactly one negative eigenvalue.
5. **GPU = CPU** on items 1–3 to rounding (test_gpu).

**Status, 2026-11-24: step 1a DONE** (`src/neb.jl`, `test/test_neb.jl`, 49 tests).
- API: `GNEB(world, mA, mB)` or `GNEB(world, [states…])`, `relax!(neb)`,
  `setclimbing!`, `energies`, `barrier`, `pathcoordinate`, `saddle` (Hermite),
  `images`, `saveimages`. Forces reduced by V·Msref, so `stopforce` and `spring`
  are in tesla.
- **Optimizer: Barzilai–Borwein on the whole chain plus the Cayley rotation** (the
  minimizer's scheme). VPO (Bessarab 2015) was implemented and compared, then
  removed. With the corrected projection it converges in every case and gives the
  same barrier (≤ 1.5e-13), in 2–3× more steps on Stoner–Wohlfarth, 6–21× on the
  flat cell and 10–11× on the wire. (A first comparison, made before the
  projection fix, had it stall at h = 0.8; that was the bug.) A force-increase rejection
  test (thresholds 10, 2, 1.25) cost 2–5× more steps on these problems and did not
  rescue any of them. Only non-finite steps are rejected by default (`reject=Inf`);
  see step 1b for harder bands, where a threshold helps some.
- **Bug found and fixed while validating.** The projection X − (m·X)m is orthogonal
  to m only if |m| = 1 exactly. The Cayley step then amplified the rounding error of
  |m| by ×1.6 per step, until the band blew up. The fix is to write it as
  |m|²X − (m·X)m (= −m×(m×X)), as the minimizer's torque already does; ||m|−1| now
  stays below 1e-12.
- Results (Float64, stopforce 1e-7 T):
  - Stoner–Wohlfarth, h = 0 … 0.8: ΔE = KV(1−h)² to ≤ 6e-15 relative; saddle
    m_z = h to 1e-6.
  - Flat demagnetizing cell, from an out-of-plane initial path: the band relaxes
    into the plane, and ΔE matches the 1D in-plane maximum to ≤ 3e-13 for k = 0.1,
    1 and 10 T.
    - A waypoint on the symmetry plane xz keeps the band in that invariant plane, so
      it converges to a higher-order saddle path. This is correct NEB behaviour, and
      the initial path must break the symmetry.
  - Without climbing (9 images), the Hermite saddle estimate is within 4.4e-4,
    against 1.4e-2 for the highest image.
  - Domain-wall nucleation in a 64 nm wire:
    - ΔE is identical to 11 digits for k = 0.3, 1 and 3, and for N = 13 and 21.
    - At 0.3 h it is 1/3 of the coherent barrier.
    - At h = 0 the saddle is a centred wall. ΔE/(4√(AK)·S) − 1 = −2.83, −0.66 and
      −0.16 % for 4, 2 and 1 nm cells: second-order convergence (√(A/K) = 5.1 nm).
    - Float32 gives the same barrier to 3e-7.
- GPU = CPU, `gpu_validation/test_gpu.jl` on labmac03 (2026-11-24, code c3a1fee):
  662/662 checks pass.
  - Wire and region disc, Float64: initial E and F agree to ≤ 6e-16; the relaxed
    barrier to ≤ 7e-13.
  - Float32: the relaxed barrier agrees to ≤ 7e-6.
  - `energy!` and `linearfield!` on the GPU: ≤ 8e-16 in Float64.

**Status, 2026-11-24: step 1b DONE** (`gpu_validation/gneb/`, results and
tables in `gpu_validation/gneb/results/README.md`).
- Same end states, initial path, 21 images, MicroMagnetic.jl velocity projection
  against JuliaMag: one skyrmion 7.9422822692e-20 J (−5.5e-13 relative), lattice of four
  skyrmions 1.2023840745e-19 J (−9e-9), ellipse +1.97e-4 (uniform path) and +1.66e-4
  (wall path). The ellipse differences are those of the demagnetizing kernel, present in
  the energy of the same image.
- Independent of N = 11, 21, 31 and k = 0.3, 1, 3 for the ellipses; for the films only
  with enough images (N = 21 or more).
- Correction. A "skyrmion collapse" first reported was the lattice of four: the tutorial
  seed has charge 0 and its relaxed state depends on the minimizer. The scripts now check
  the Berg–Lüscher charge of the start state.
- Weak points found. With 11 images the lattice band converged to barrier 0 (flat
  translation modes, the transition in one jump between two images): `relax!` now warns
  from the energies halfway between images (`midpointenergies`). Barzilai–Borwein is less
  robust than velocity projection on two under-resolved bands; a finite `reject` rescues
  some, with no reliable value, so the default stays `Inf`.
- GPU = CPU, `gpu_validation/test_gpu.jl` on labmac03 (code d141053): 670/670 checks.
  GNEB on the wire and the region disc, Float64: energies, forces and midpoint energies
  to <= 6e-16, relaxed barrier to <= 3e-12; Float32: barrier to <= 1.3e-6. The film and
  ellipse cases above were run on the CPU only.
- Not done: the DMI-film barrier against published continuum values.
- **Step 1c, 2026-11-25: saddle certification and paper section DONE.** Order from the lowest
  eigenvalues of the energy Hessian (`certifysaddle`): skyrmion 1, ellipse wall path 1,
  **ellipse uniform path 2**, **four-skyrmion lattice 4, with a start state that is not a
  minimum** (two curvatures of −1.6 mT). Retraction: the lattice barrier reported in step 1b is
  not that of a metastable state; the agreement between codes did not show it.

**Paper.** A section "Energy barriers", with the Stoner–Wohlfarth figure
(barrier vs h, analytic curve in red, JuliaMag in blue) and the ellipse or
skyrmion path.

---

## 2. Linear eigenmodes (FMR spectra, mode profiles)

**Status, 2026-11-25: DONE** (`src/eigenmodes.jl`, `test/test_eigenmodes.jl`,
`gpu_validation/eigenmodes/README.md`), with the differences from the plan below: no KrylovKit (own
antisymmetric Lanczos in the A-product with conjugate gradients, dense Hermitian problem,
LOBPCG for the lowest curvatures). An independent review found a serious fault of the first
version: the Lanczos basis lost its orthogonality in degenerate or nearly complete spectra and
returned wrong modes without a warning; fixed by recomputing A·v for every new vector, and
tested with many starting vectors.

### Method

Linearize the LLG about an equilibrium m₀, relaxed with the minimizer to a tight
stopdm, with the torque checked.
- **Deviation.** δm = a e₁ + b e₂ in a local orthonormal frame (e₁, e₂, m₀) per
  cell; two unknowns per magnetic cell.
- **Hessian.** In the tangent space,
  K δm = −V Msat [ P B_lin(δm) − (m₀·B₀) δm ],
  where P projects onto the tangent planes and B₀ = B_eff(m₀). It is symmetric,
  and positive definite at a stable minimum.
- **Undamped dynamics.** dδm/dt = γ J K δm, with J the 90° rotation in each
  tangent plane, scaled by 1/(V Msat). The eigenvalues come in pairs ±iω.
- **Damping (optional).** (J − αI)-type factor; the imaginary parts then give the
  linewidths.

**Operator.** A matrix-free product: one `linearfield!` plus a few broadcasts per
product. It works unchanged on the GPU, so it reuses the FFT demag.

**Solvers:**
- **Small systems** (2N ≲ 6000): build K column by column with 2N operator
  products, then a dense generalized eigenproblem. This is the exact reference
  for the iterative solver.
- **Large systems:** the lowest |ω| are interior eigenvalues of J K, so use shift
  and invert around a target frequency σ.
  - Outer: KrylovKit `eigsolve`. Inner: `linsolve` (GMRES) on (J K − iσ)
    x = y, matrix-free. KrylovKit works with CuArrays.
  - A second route: for real symmetric K > 0, J K is similar to the real
    antisymmetric K^{1/2} J K^{1/2}, so its spectrum is ±iω. A structure-preserving
    (Hamiltonian) Lanczos in the K-inner product finds the ω without forming
    K^{1/2}. Evaluate both routes on the Kittel case and keep the robust one.
  - Fallback if shift-and-invert is too slow on the disc: compute the
    highest-frequency modes, or use spectral transformation by polynomial
    filtering. Document the cost.

**Output:**
- Frequencies f = ω/2π and, with damping, linewidths.
- Mode profiles as OVF: real and imaginary δm, mapped back to Cartesian.
- Mode participation (fraction of energy per region).
- A spectrum table.

### Code

**`src/eigenmodes.jl`:**
- `EigenProblem(world, m0)` (frames, operator).
- `eigenmodes(world, m0; nmodes=10, target=0, method=:auto, damping=false)`.
  Returns frequencies and modes.
- `savemode(dir, mode, k)`.
- Simulation-level: `eigenmodes(sim; …)`.

**Dependencies.**
- KrylovKit.jl as a dependency, small and pure Julia; or a package extension if
  the core must stay lean (decide at implementation).
- LinearAlgebra for the dense path.

### Validation

1. **Macrospin Kittel** (one cell, uniaxial anisotropy and shape, field along the
   easy axis): f = γ/2π · sqrt((B + B_K + μ0 Msat (N_y − N_z))(B + B_K +
   μ0 Msat (N_x − N_z))), exact to 1e-8.
2. **Periodic thin film:**
   - Uniform mode f = γ/2π sqrt(B (B + μ0 Msat)).
   - Exchange spin waves at k = 2πn/L: the analytic dispersion of the
     discretized exchange (finite-difference factor), to rounding.
3. **µMAG FMR standard problem** (Baker et al., J. Magn. Magn. Mater. 421, 428
   (2017)): the 120×120×10 nm Py square with the specified tilted field.
   - Compare the eigenfrequencies with the published spectrum peaks.
   - Compare with a ringdown done with JuliaMag's own integrators (excitation,
     ⟨m⟩(t), FFT), so the two JuliaMag routes agree with each other.
4. **Vortex disc gyrotropic mode.** The lowest eigenfrequency of the 128 nm disc
   must match the 969.8 MHz free gyration validated against mumax3. With damping,
   the linewidth must match Γ = 0.0797 ns⁻¹.
5. **GNEB saddle certification** (see §1): exactly one negative eigenvalue at the
   climbing image.
6. **GPU = CPU** for the operator to rounding, and for the frequencies on cases 1–4.

**Paper.** A section "Eigenmodes", with the FMR standard-problem spectrum
(published or ringdown in red, eigen in blue) and the gyrotropic mode profile.

---

## 3. Order of work and gates

| step | content | gate to the next step |
|---|---|---|
| 0 | `energy!`, `linearfield!` (CPU + GPU) | tests pass; GPU check queued on an idle GPU |
| 1a | GNEB core, CPU, Stoner–Wohlfarth | ΔE to 1e-6 — DONE 2026-11-24 (≤ 6e-15) |
| 1b | Ellipse and skyrmion barriers, image and spring convergence | cross-code agreement documented — DONE 2026-11-24 |
| 1c | GPU runs, paper section, commit | — DONE 2026-11-25 |
| 2a | Eigen operator and dense solver, Kittel and film cases | to 1e-8 — DONE 2026-11-25 (film: only without demag) |
| 2b | Iterative solver, FMR standard problem, disc gyrotropic mode | agreement with the ringdown and with mumax3's 969.8 MHz — DONE 2026-11-25 (968.234 MHz; 969.8 was an estimator biased by +0.9 MHz) |
| 2c | GNEB saddle certification, GPU, paper section, commit | — |

GPU runs go through `run_when_gpu_idle.sh`; the machines are shared. Each long run
is reviewed by a subagent first.
