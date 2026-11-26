# Plan: excitations that vary in space and time, and currents from anisotropic magnetoresistance

2026-11-25. Two features, in this order, before the GUI review. The second needs the first.

## Where JuliaMag is

- The applied field is one vector for the whole mesh (`World.Bext::NTuple{3}`), constant or a
  function of time only (`setdrive!`).
- The current is one vector for the whole mesh (`SpinCurrent.J`): the Zhang-Li and Slonczewski
  kernels take `J` as a scalar triple.
- Material parameters are per region. Excitations are not: that is what is missing. mumax3 has both
  `B_ext.SetRegion(r, f(t))` and `B_ext.Add(mask, f(t))`, the latter in its hard-disk example (a
  Gaussian write field times sin(2πft)).

## A. Field and current that depend on position and time

**Model (as mumax3).** The field is a sum of terms, each a fixed spatial profile times a function of
time:

    B(r, t) = B0(t) + Σ_k  M_k(r) f_k(t),

where `B0(t)` is the uniform part (as now), `M_k` a vector map on the mesh (tesla) and `f_k(t)` a
scalar function. A region is a mask that is the vector inside the region and 0 elsewhere. The same
for the current density `J(r, t)`.

The maps live on the device (GPU) once. Each stage of the solvers then multiplies them by the scalars
`f_k(t)` and adds them, in one kernel, with no copy from the host. A field that is not separable,
for instance a pulse that moves, is a sum of terms. Otherwise, as a slower option, it is a function
`(x, y, z, t)` evaluated on the host at each stage.

**Interface (Julia).**

```julia
pulse = fieldmap(mesh) do x, y, z               # the profile, in tesla
    (0, 0, exp(-((x - 100e-9)^2 + y^2) / (2 * (30e-9)^2)))
end
addfield!(sim, pulse, t -> 0.05 * exp(-((t - 1e-9) / 50e-12)^2))   # Gaussian in space and time
addfield!(sim, regionmap(sim, 2, (0, 0, 1)), t -> 0.01sin(2π * 3e9 * t))  # one region
setexternalfield!(sim, (0.1, 0, 0))             # the uniform part, as now
addcurrent!(sim, currentmap(...), t -> 1.0)     # same for J
clearfields!(sim)
```

**Everything that uses the field sees it.** That covers the effective field, the Zeeman energy and
`totalenergy`, the minimizer, GNEB, the eigenmodes (static maps), the table columns, and the CPU and
GPU paths. The spin torques take J per cell.

**Validation.**
- Against mumax3 3.11 on labmac03, the same problem in both codes:
  - a Gaussian pulse in a disc (`B_ext.Add`);
  - a field per region (`B_ext.SetRegion`, with a function of time);
  - Zhang-Li with a current per region.
  - Tables and fields are compared.
- Energy: the Zeeman energy of a known map against its integral.
- GPU against CPU (`test_gpu.jl`), in Float64 and Float32.
- A uniform map reproduces today's uniform field bit for bit.

## B. Currents from anisotropic magnetoresistance (AMR)

### What the legacy Fortran code does (a thesis appendix, modules `ini_amr`, `evo_amr`)

- **Electric potential:** an electric potential U on one layer of cells.
- **Relaxation:** Gauss–Seidel relaxation of Kirchhoff's law at every cell:

      U_i = (Σ_n s_n U_n − C_i) / Σ_n s_n,

  - `s_n = 1/ρ_n` is the conductance of the bond to the neighbour n, with
    `ρ_n = ρ⊥ + (ρ∥ − ρ⊥) m_d²` and `m_d` the component of m of **the neighbour** along the bond;
  - `C_i` injects ±380 µA in the end columns (the contacts);
  - empty cells do not conduct.
- **Currents:** the currents `(U_i − U_{i+1}) s` give the Zhang-Li drift velocity.
- **Coupling:** quasi-static. The potential is relaxed again after every time step, starting from the
  last one.

### Can JuliaMag have it? Yes

It is a linear elliptic problem for the potential at each instant. The rest of the code supports it:

- it runs on the same mesh, regions and geometry;
- its stencils have the structure of the exchange;
- the solver is the conjugate gradient already in `src/eigenmodes.jl`.

It needs feature A for the current per cell in the spin torques.

The physics should not be ported as written in the Fortran. Six points, read from the listing:

1. **The bond conductance uses one cell only.** Each bond takes the m of one cell: the conductance
   from i to i+1 differs from that from i+1 to i. The network is not symmetric, so current is not
   conserved from cell to cell.
2. **Only the diagonal of the resistivity is kept.** The resistivity of AMR is a tensor,
   `ρ = ρ⊥ I + (ρ∥ − ρ⊥) m mᵀ`. The terms `m_x m_y` (the planar Hall effect) are dropped, and with
   them the transverse currents of a texture.
3. **One layer.** The potential is solved in one layer.
4. **Stopping criterion.** The relaxation stops when 1000 cells change by less than 1e-6 relative.
   That is not a residual, and it is ill-defined where U ≈ 0.
5. **Hard-coded cell size.** `ρ/5e-9` fixes the cell size inside the conductance.
6. **Units of the drift velocity.** The velocity divides the current by the cell volume, not by the
   cell's cross-section.

Two more points concern the micromagnetics, not the AMR: the LLG update is done cell by cell, in
place, and the dipolar field is cut off at ±10 cells.

### Model in JuliaMag

- **Conduction:** `∇·(σ(m) ∇φ) = 0` in the conducting cells, `j = −σ ∇φ`, with
  `σ = ρ⁻¹ = (1/ρ⊥) [I − (Δρ/ρ∥) m mᵀ]` (exact inverse; Δρ = ρ∥ − ρ⊥).
  - `ρ∥`, `ρ⊥` are given per region. A region without Msat can conduct (a lead).
- **Discretization:** finite volumes on the cells, in 3D.
  - The flux through a face has its normal and its transverse terms, the latter through the
    gradient averaged over the two cells. A 19-point stencil in 3D (9 in 2D).
  - The face conductance is the mean of the two cells. The matrix is then symmetric.
- **Contacts:** regions with
  - a fixed potential, or
  - a fixed total current: one unknown potential per contact, with a constraint of total current.

  All other boundaries are insulating, `n·j = 0`.
- **Solver:** conjugate gradient, matrix-free.
  - It starts from the last potential.
  - Its residual is in the norm of the current (stop below 1e-8 relative), with a Jacobi
    preconditioner.
  - CPU and GPU, Float64 sums.
- **Coupling:** quasi-static. The potential is solved at every stage of the solvers, or, as an
  option, once per step (cheaper, first order in Δt; measured against the former).
- **Outputs:**
  - the fields φ and j (OVF);
  - the voltage and the current of each contact;
  - the resistance R(t), for magnetoresistance curves against field.
- **Spin torques:** Zhang-Li with j(r) per cell (feature A); Slonczewski per cell if j_z.

### Validation

- **Uniform m at angle θ in a long strip:**
  - `R = (L/(W t)) (ρ⊥ + Δρ cos²θ)`;
  - the planar Hall voltage across the width is `(Δρ I/t) sinθ cosθ`;
  - second-order convergence with the cell.
- **Conservation and symmetry:**
  - the same current through every cross-section;
  - reciprocity `R(m) = R(−m)`;
  - with Δρ = 0, the uniform current of today's code to rounding.
- **Independent reference:** the same problem on a nonuniform texture (a Néel wall in a strip, a vortex
  in a disc with two contacts) with a separate finite-element solve in Python (scipy, triangular
  mesh). Potential and resistance against mesh refinement.
- **Dynamics:**
  - a vortex in a disc driven by a current through two contacts, with and without AMR;
  - the difference against Δρ/ρ: it must vanish linearly.
- GPU against CPU.

### Not in the plan, can follow

- The Oersted field of j(r): an FFT convolution like the demag, Biot–Savart.
- Joule heating.
- The anomalous Hall effect.
