<!-- All code blocks in this manual are executable JuliaMag API, verified to run. -->
# JuliaMag — User Manual

JuliaMag is a micromagnetic simulator written entirely in Julia. It solves the
Landau–Lifshitz–Gilbert (LLG) equation on a finite-difference mesh, with the
exchange, demagnetizing, anisotropy, Dzyaloshinskii–Moriya, and Zeeman fields,
and the Zhang–Li and Slonczewski spin-transfer torques. It reproduces the µMAG
standard problems against mumax3 and OOMMF.

This manual covers installation, the core concepts, and three worked tutorials:

1. [µMAG Standard Problem 4](#tutorial-1--standard-problem-4) — dynamic reversal.
2. [A disc with a vortex](#tutorial-2--a-disc-with-a-vortex) — non-rectangular geometry.
3. [A stripe with a skyrmion driven by a current](#tutorial-3--a-stripe-with-a-skyrmion-and-a-current) — DMI + spin-transfer torque.

---

## 1. Installation

JuliaMag needs Julia 1.9+. Install Julia from [julialang.org](https://julialang.org)
(or via [Juliaup](https://github.com/JuliaLang/juliaup)), then add the package:

```julia
using Pkg
Pkg.add(url = "https://github.com/rlnovak/juliamag")
```

Load it in every session with:

```julia
using JuliaMag
```

The optional GPU and GUI features have their own dependencies; see
[Going further](#7-going-further).

---

## 2. Core concepts

A simulation is built from a few objects:

| Object | What it is |
|--------|-----------|
| `Mesh` | the finite-difference grid: cell counts `(Nx,Ny,Nz)` and cell size `(cx,cy,cz)` in metres |
| `Material` | uniform material parameters: `Msat`, `Aex`, `alpha`, and optionally `Ku`, `Dind`, `Dbulk`, STT constants |
| `RegionParams` | per-region parameters, for multi-material or shaped geometry |
| magnetization | an `Array{T,4}` of shape `(3, Nx, Ny, Nz)`: `m[:, i, j, k]` is the unit vector in cell `(i,j,k)` |
| `Simulation` | bundles a mesh, parameters, magnetization, and an output table |

**Coordinate convention.** The origin is at the geometric centre of the sample.
Cell centres run from `-L/2` to `+L/2` along each axis. Shapes and initial-state
configurations use these centred coordinates, and `translate` shifts them.

**Units.** SI throughout: lengths in metres, `Msat` in A/m, `Aex` in J/m, fields
in tesla, time in seconds. Anisotropy `Ku` is J/m³, interfacial DMI `Dind` is
J/m², bulk DMI `Dbulk` is J/m³.

**The two ways to reach equilibrium.**

- `relax!(sim)` runs an energy minimizer — the fast way to a static equilibrium.
- `run!(sim, duration)` integrates the LLG in time — for real dynamics.

---

## 3. A minimal example

Relax a Permalloy square to equilibrium and read its average magnetization:

```julia
using JuliaMag

mesh = Mesh((64, 64, 1), (5e-9, 5e-9, 5e-9))     # 320 × 320 nm, 5 nm cells
sim  = Simulation(mesh, material("Permalloy"))    # Msat, Aex, α from the library
setmag!(sim, UniformConfig(1, 0, 0))              # saturate along +x
relax!(sim)

println(average(sim))                             # ⟨mx, my, mz⟩
```

`material("Permalloy")` returns a `Material` with standard parameters. The known
names are `Permalloy`/`Py`, `Fe`, `Co`, `Ni`, `CoFeB`, `YIG`
(`materialnames()`); override the damping with `material("Co"; alpha = 0.01)`.

You can also build a `Material` directly:

```julia
mat = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)
```

---

## Tutorial 1 — Standard Problem 4

The µMAG standard problem 4 is the canonical dynamic test: a Permalloy film is
relaxed to its S-state, then a field is applied and the magnetization reverses.

**Geometry and material.** 500 × 125 × 3 nm, discretized 160 × 40 × 1.

```julia
using JuliaMag
using Printf

mesh = Mesh((160, 40, 1), (3.125e-9, 3.125e-9, 3e-9))
mat  = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)
sim  = Simulation(mesh, mat; demag = true)
```

**Step 1 — relax to the S-state.** Saturate along the body diagonal `[1,1,1]`
and minimize the energy; the demagnetizing field curls the ends into the S.

```julia
setmag!(sim, UniformConfig(1, 1, 1))
relax!(sim; stopdm = 1e-6)
@printf("S-state ⟨m⟩ = (%.4f, %.4f, %.4f)\n", average(sim)...)
# ≈ (0.9668, 0.1256, 0.0000)
```

**Step 2 — choose what to record.** Save the time and the averaged
magnetization every 5 ps.

```julia
savequantities!(sim, q_time(), q_m(); every = 5e-12)
```

**Step 3 — apply field 1 and integrate for 1 ns.**
`µ0·H = (-24.6, 4.3, 0)` mT.

```julia
JuliaMag.setexternalfield!(sim.world, (-24.6e-3, 4.3e-3, 0.0))
run!(sim, 1e-9)
writetable(sim.table, "stdproblem4.txt")
```

The table now holds `⟨m⟩(t)`. `⟨mx⟩` starts near 0.97, crosses zero at about
0.14 ns, dips to roughly −0.68, and rings down toward −1, while `⟨my⟩` peaks near
+0.75 — matching the published reference. A ready-to-run version with a plot is
[`examples/stdproblem4.jl`](../examples/stdproblem4.jl).

---

## Tutorial 2 — A disc with a vortex

Non-rectangular geometry is expressed by painting material into a shape on a
rectangular mesh: cells outside the shape are left empty (`Msat = 0`) and take no
part in the simulation. Here a Permalloy disc holds a magnetic vortex.

**Step 1 — define the disc.** Start from an empty background, then paint a
cylinder with material.

```julia
using JuliaMag
using Printf

mesh = Mesh((40, 40, 1), (5e-9, 5e-9, 5e-9))       # 200 × 200 nm mesh
rp   = RegionParams(mesh, material("Permalloy"))    # region 0 = Permalloy (default)
setregion!(rp, 0; Msat = 0.0)                       # …but make region 0 empty
defregion!(rp, 1, Cylinder(180e-9, 1e6))            # paint a 180 nm disc as region 1
```

`Cylinder(diam, height)` is centred at the origin with its axis along z; the
large height makes it span the single layer. Region 1 keeps the Permalloy
parameters copied from the default; region 0 (everything outside the disc) has
`Msat = 0`.

**Step 2 — seed a vortex and relax.** `setmag!` clears the empty cells
automatically, so only the disc carries magnetization.

```julia
sim = Simulation(mesh, rp; demag = true)
setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))   # circulation +1, core +z
relax!(sim; stopdm = 1e-6)
```

**Step 3 — locate the vortex core.**

```julia
x, y, z, pol = vortexcore(sim.m, mesh)
@printf("vortex core at (%.1f, %.1f) nm, polarity %.2f\n", x*1e9, y*1e9, pol)
# core near the disc centre, polarity ≈ +1
```

You can track the core over time by adding `q_vortexcore()` to the table before a
`run!`, e.g. to watch it gyrate under a field or current:

```julia
savequantities!(sim, q_time(), q_m(), q_vortexcore(); every = 20e-12)
JuliaMag.setexternalfield!(sim.world, (5e-3, 0.0, 0.0))   # small in-plane field
run!(sim, 2e-9)
```

**Circulation and polarity.** `VortexConfig(mesh; circ, pol)` takes the
in-plane circulation `circ = ±1` (clockwise/counter-clockwise) and the core
polarity `pol = ±1` (`±z`). Place the core off-centre with `translate`:

```julia
setmag!(sim, translate(VortexConfig(mesh), 40e-9, 0, 0))  # core at x = +40 nm
```

---

## Tutorial 3 — A stripe with a skyrmion and a current

This tutorial combines interfacial DMI (which stabilizes Néel skyrmions),
perpendicular anisotropy, and a Zhang–Li spin-polarized current that drives the
skyrmion along the stripe.

**Step 1 — a material with PMA and interfacial DMI.**

```julia
using JuliaMag
using Printf

mesh = Mesh((100, 40, 1), (2e-9, 2e-9, 1e-9))      # 200 × 80 × 1 nm stripe
mat  = Material(
    Msat  = 5.8e5,     # A/m
    Aex   = 1.5e-11,   # J/m
    alpha = 0.3,       # damping (high → the skyrmion settles quickly)
    Ku    = 8e5,       # J/m³ perpendicular anisotropy…
    anisU = (0, 0, 1), # …along z
    Dind  = 3.0e-3,    # J/m² interfacial DMI (stabilizes a Néel skyrmion)
    pol   = 1.0,       # current spin polarization
    xi    = 0.2,       # Zhang–Li non-adiabaticity
)
sim = Simulation(mesh, mat; demag = true)
```

**Step 2 — seed a Néel skyrmion and relax.** `charge` is the topological charge,
`pol` the core polarity (here the core points `-z` in a `+z` background).

```julia
setmag!(sim, NeelSkyrmionConfig(mesh; charge = 1, pol = -1))
@printf("seed Q = %.2f\n", topologicalcharge(sim.m, mesh))
relax!(sim; stopdm = 1e-6)

xs, ys, _ = skyrmionpos(sim.m, mesh)
@printf("skyrmion at (%.1f, %.1f) nm, Q = %.2f\n",
        xs*1e9, ys*1e9, topologicalcharge(sim.m, mesh))
# a stable Néel skyrmion, |Q| ≈ 1
```

`topologicalcharge` integrates `m·(∂ₓm × ∂ᵧm)/4π`; it is close to ±1 for a single
skyrmion (a little below in magnitude because of discretization). `skyrmionpos`
returns the centroid of the charge density.

**Step 3 — drive it with a current.** Apply an in-plane charge current
`J = (Jx, 0, 0)` and integrate with the Zhang–Li torque added to the LLG. Track
the skyrmion position over time.

```julia
savequantities!(sim, q_time(), q_skyrmionpos(), q_topocharge(); every = 20e-12)

J = (5e12, 0.0, 0.0)              # A/m², along +x
runcurrent!(sim, J, 2e-9; every = 20e-12)

writetable(sim.table, "skyrmion_track.txt")
```

`runcurrent!(sim, J, duration; every)` integrates the LLG with the Zhang–Li
spin-transfer torque of the current `J` added to the right-hand side, saving a
row every `every` seconds. (An applied *field* goes through `setexternalfield!`,
but a spin-transfer *current* is a torque, not a field, so it has its own
driver.) The material's `pol` and `xi` set the polarization and non-adiabaticity.

The current pushes the skyrmion along the stripe, with a transverse deflection —
the skyrmion Hall effect. The table records its trajectory, which you can plot
from `skyrmion_track.txt`.

For a complete current-driven run see the Zhang–Li validation in
[`examples/stdproblem5.jl`](../examples/stdproblem5.jl) (µMAG standard problem 5),
which drives a vortex with a current and matches mumax3 to 9×10⁻⁵.

---

## 4. Choosing what to save

`savequantities!(sim, q1, q2, …; every = Δt)` sets the table columns and the
auto-save interval. The available quantities:

| Quantity | Columns | Meaning |
|----------|---------|---------|
| `q_time()` | `t` | simulated time [s] |
| `q_m()` | `mx, my, mz` | averaged magnetization |
| `q_m_region(id)` | `mx, my, mz` | averaged over one region |
| `q_energy()` | `E_total` | total energy [J] |
| `q_exchangeenergy()`, `q_demagenergy()`, `q_zeemanenergy()`, `q_anisenergy()` | one each | per-term energies |
| `q_maxtorque()` | `maxTorque` | max torque [rad/s] |
| `q_Bext()` | `B_extx,y,z` | applied field [T] |
| `q_vortexcore()` | `x, y, z, polarity` | vortex core |
| `q_skyrmionpos()` | `x, y, z` | skyrmion centroid |
| `q_dwpos()` | `x, y, z` | domain-wall position |
| `q_topocharge()` | `Q` | topological charge |

Write the accumulated table with `writetable(sim.table, "out.txt")` — a
tab-separated file with a header line, readable by any plotting tool.

---

## 5. Initial states

`setmag!(sim, config)` fills the magnetization from a configuration:

| Config | Parameters |
|--------|-----------|
| `UniformConfig(mx, my, mz)` | direction (normalized) |
| `VortexConfig(mesh; circ, pol)` | circulation ±1, core polarity ±1 |
| `AntiVortexConfig(mesh; circ, pol)` | as vortex |
| `NeelSkyrmionConfig(mesh; charge, pol)` | topological charge ±1, core polarity ±1 |
| `BlochSkyrmionConfig(mesh; charge, pol)` | as Néel |
| `VortexWallConfig(mesh, mleft, mright; circ, pol)` | end domains + vortex |
| `TwoDomainConfig(mesh, m1, mwall, m2)` | left, wall, right magnetizations |
| `RandomConfig()` | random unit vectors |

Reposition any localizable state with `translate(config, dx, dy, dz)`. Load a
saved state from an OVF file with `m, header = loadovf("state.ovf")`.

---

## 6. Geometry and multiple materials

Build shapes and combine them:

- Primitives: `Cuboid(sx,sy,sz)`, `Rect(sx,sy)`, `Cylinder(diam,h)`, `Circle(d)`,
  `Ellipsoid(dx,dy,dz)`, `Cone(diam,h)`, `Superball(diam,p)`.
- Layers (for multilayers): `Layer(mesh, k)`, `Layers(mesh, k1, k2)`.
- Transforms: `translate`, `scale`, `rotz`.
- Set operations: `shapeunion`, `shapeintersect`, `shapediff`, `shapecomplement`.

Assign materials to regions with a `RegionParams`:

```julia
rp = RegionParams(mesh, material("Permalloy"))   # region 0 = Permalloy everywhere
defregion!(rp, 1, Layers(mesh, 3, 5))            # paint the top two layers region 1
setregion!(rp, 1; Msat = 1.1e6, Aex = 1.9e-11)   # give region 1 a different material
sim = Simulation(mesh, rp)
```

At a region interface the exchange coupling uses the harmonic mean of the two
stiffnesses. A region with `Msat = 0` is empty (unfilled geometry). Per-region
averages come from `q_m_region(id)`.

---

## 7. Going further

- **Standalone examples:** [`examples/`](../examples/) has the standard problems
  2, 4, and 5 with comparison plots against mumax3 and OOMMF.
- **Desktop GUI:** a dedicated Qt/Makie window — see [`gui/README.md`](../gui/README.md).
  Set it up once with `julia --project=gui gui/setup.jl`, then
  `julia --project=gui gui/app.jl`.
- **GPU:** on a CUDA machine, `using CUDA` alongside JuliaMag loads GPU methods;
  move a state to the device with `togpu(m)`. See
  [`examples/gpu_check.jl`](../examples/gpu_check.jl).
- **Paper:** [`paper/juliamag.tex`](../paper/juliamag.tex) documents the design
  and validation.
