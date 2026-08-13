<!-- All code blocks in this manual are executable JuliaMag API, verified to run. -->
# JuliaMag — User Manual

JuliaMag is a micromagnetic simulator written entirely in Julia. It solves the
Landau–Lifshitz–Gilbert (LLG) equation on a finite-difference mesh, with the
exchange, demagnetizing, anisotropy, Dzyaloshinskii–Moriya, and Zeeman fields,
the Zhang–Li and Slonczewski spin-transfer torques, and finite-temperature
(Langevin) dynamics. It reproduces the µMAG standard problems against mumax3 and
OOMMF, and runs the whole pipeline on a CUDA GPU.

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
part in the simulation. Here a Permalloy disc holds a magnetic vortex, and we
trace its **in-plane hysteresis loop** — sweeping a field along `x` displaces the
vortex core sideways along a reversible branch, until at a critical field the
vortex annihilates and the disc saturates.

A well-formed vortex loop needs a disc that is **wide and thin**: we use 500 nm
diameter × 20 nm thick, a classic experimental vortex size. (A small, thin disc
gives a nearly square loop — the reversible core-displacement branch, the
signature of the vortex loop, needs room for the core to move.)

**Step 1 — define the disc.** Start from an empty background, then paint a
cylinder with material.

```julia
using JuliaMag
using Printf

mesh = Mesh((100, 100, 4), (5e-9, 5e-9, 5e-9))       # 500 × 500 × 20 nm mesh
rp   = RegionParams(mesh, material("Permalloy"))      # region 0 = Permalloy (default)
setregion!(rp, 0; Msat = 0.0)                         # …but make region 0 empty
defregion!(rp, 1, Cylinder(500e-9, 1e6))              # paint a 500 nm disc as region 1
sim  = Simulation(mesh, rp; demag = true)
```

`Cylinder(diam, height)` is centred at the origin with its axis along z; the
large height makes it span all four layers. Region 1 keeps the Permalloy
parameters; region 0 (outside the disc) has `Msat = 0`.

**Step 2 — seed a vortex.** Starting the sweep from the vortex state lets the
minimizer track the vortex continuously as the field changes.

```julia
setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))  # circulation +1, core +z
```

`VortexConfig(mesh; circ, pol)` takes the in-plane circulation `circ = ±1` and
the core polarity `pol = ±1` (`±z`); place the core off-centre with
`translate(VortexConfig(mesh), 40e-9, 0, 0)`.

**Step 3 — choose what to save.** Add the applied field to the output table;
time and ⟨m⟩ are always the first columns, so each row will hold
`t, mx, my, mz, B_extx, B_exty, B_extz`.

```julia
savequantities!(sim, q_Bext())
```

**Step 4 — sweep the in-plane field and record it.** Ramp the field along `x`
from 0 up to `+Bmax`, down to `−Bmax`, and back, relaxing at each step and
writing one table row per step with `savenow!`. Starting each relaxation from the
previous state is what makes it a hysteresis loop. We sweep over ±150 mT — well
past the vortex-annihilation field — so the disc reaches its saturated branch and
the plateau is flat at both ends.

```julia
Bmax, dB = 0.15, 0.005                               # tesla
Bs = vcat(0:dB:Bmax, Bmax:-dB:-Bmax, -Bmax:dB:Bmax)

mxs = Float64[]
for B in Bs
    JuliaMag.setexternalfield!(sim.world, (B, 0.0, 0.0))
    relax!(sim; stopdm = 1e-6)
    savenow!(sim)                                    # one row: Bx, mx, my, mz…
    push!(mxs, average(sim.m)[1])
end

writetable(sim.table, "disc_hysteresis.txt")
@printf("⟨mx⟩ range [%.3f, %.3f]\n", minimum(mxs), maximum(mxs))
```

Plotting `mxs` against `Bs` gives the S-shaped vortex loop: `⟨mx⟩ = 0` at zero
field (the centred vortex), a nearly linear reversible rise as the core is pushed
toward the edge, then an abrupt jump to the saturated branch (`⟨mx⟩ ≈ ±0.78`)
where the vortex annihilates, and a jump back as it renucleates on the return
branch. The plateau sits below 1 because a thin disc never magnetizes fully
in-plane — shape demagnetization keeps the edge spins curled even at 150 mT. The
`disc_hysteresis.txt` table holds `Bx` alongside `⟨mx,my,mz⟩` for each field
step. This is a sizeable run (~13 minutes: 40k cells over ~120 field steps). A
ready-to-run version with the plot is
[`examples/disc_hysteresis.jl`](../examples/disc_hysteresis.jl).

You can also track the core position through the loop by adding `q_vortexcore()`
to `savequantities!`.

---

## Tutorial 3 — A stripe with a skyrmion and a current

This tutorial combines interfacial DMI (which stabilizes Néel skyrmions),
perpendicular anisotropy, and a Zhang–Li spin-polarized current that drives the
skyrmion along an **infinite wire** — a stripe made periodic along `x`, so the
skyrmion re-enters from the far end instead of being pinned or annihilated at an
edge. We save OVF snapshots along the way and measure the drift velocity.

**Step 1 — a periodic stripe with PMA and interfacial DMI.** `pbc = (1,0,0)`
makes the mesh periodic along `x` (one periodic image each side).

```julia
using JuliaMag
using Printf

mesh = Mesh((150, 100, 1), (2e-9, 2e-9, 1e-9); pbc = (1, 0, 0))  # 300 × 200 × 1 nm, periodic x
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

The stripe is wide enough (200 nm) that the transverse skyrmion-Hall deflection
does not push the skyrmion into a `y` edge over the run. Exchange, DMI, demag and
the spin torque all honour the periodic axis.

**Step 2 — seed a Néel skyrmion and relax.** `charge` is the topological charge,
`pol` the core polarity (here the core points `-z` in a `+z` background).

```julia
setmag!(sim, NeelSkyrmionConfig(mesh; charge = 1, pol = -1))
relax!(sim; stopdm = 1e-6)

xs, ys, _ = skyrmionpos(sim.m, mesh)
@printf("skyrmion at (%.1f, %.1f) nm, Q = %.2f\n",
        xs*1e9, ys*1e9, topologicalcharge(sim.m, mesh))
# a stable Néel skyrmion, |Q| ≈ 1
```

`topologicalcharge` integrates `m·(∂ₓm × ∂ᵧm)/4π`; it is close to ±1 for a single
skyrmion (a little below in magnitude because of discretization). `skyrmionpos`
returns the centroid of the charge density — computed *circularly* along a
periodic axis, so it stays correct when the skyrmion straddles the wrap seam.

**Step 3 — drive it with a current and save OVF snapshots.** Apply an in-plane
charge current `J = (Jx,0,0)` and integrate with the Zhang–Li torque. We drive in
three 1-ns intervals, writing an OVF field file at each snapshot (`t = 0,1,2,3`
ns) with `saveovf`.

```julia
savequantities!(sim, q_skyrmionpos(), q_topocharge())   # columns: t, mx,my,mz, sky x,y,z, Q

J, tsnap = (2e12, 0.0, 0.0), 1e-9
saveovf("skyrmion_00.ovf", sim.m, mesh)
for s in 1:3
    runcurrent!(sim, J, tsnap; every = 20e-12)
    saveovf(@sprintf("skyrmion_%02d.ovf", s), sim.m, mesh)
end
writetable(sim.table, "skyrmion_track.txt")
```

`runcurrent!(sim, J, duration; every)` integrates the LLG with the Zhang–Li
spin-transfer torque of the current `J` added to the right-hand side, saving a
table row every `every` seconds. (An applied *field* goes through
`setexternalfield!`, but a spin-transfer *current* is a torque, not a field, so
it has its own driver.) The skyrmion moves *against* the current (electron flow),
toward `−x`, with a transverse skyrmion-Hall deflection along `y`.

**Step 4 — measure the velocity.** The recorded `x(t)` is periodic (it jumps by
`Lx` at each re-entry), so unwrap it before fitting the slope:

```julia
Lx = 150 * 2e-9
function unwrap(xs, L)
    out = copy(xs)
    for i in 2:length(out)
        d = out[i] - out[i-1]
        d >  L/2 && (out[i:end] .-= L)
        d < -L/2 && (out[i:end] .+= L)
    end
    out
end

t  = getindex.(sim.table.rows, 1)               # column 1 = time
xw = unwrap(getindex.(sim.table.rows, 5), Lx)   # column 5 = skyrmion x
v  = sum((t.-t[1]) .* (xw.-xw[1])) / sum((t.-t[1]).^2)
@printf("skyrmion velocity vx = %.1f m/s\n", v)   # ≈ −178 m/s for J = 2×10¹² A/m²
```

The unwrapped `x(t)` is a straight line whose slope is the drift velocity — about
**−178 m/s** here. The full example is
[`examples/skyrmion_drive.jl`](../examples/skyrmion_drive.jl).

**Visualizing the OVF snapshots.** `examples/plot_ovf.jl` reads an OVF file and
draws the `mz` colour map with an in-plane `(mx,my)` quiver overlay:

```
julia --project=examples examples/plot_ovf.jl skyrmion_00.ovf
```

or, as a library, `include("examples/plot_ovf.jl"); plotovf("skyrmion_03.ovf")`.
The snapshots show the Néel hedgehog texture drifting along the wire and
re-entering through the periodic seam.

For another current-driven run see the Zhang–Li validation in
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
saved state from an OVF file with `m, header = loadovf("state.ovf")`, and write
one with `saveovf("state.ovf", sim.m, mesh)` — an OVF 2.0 text file readable by
OOMMF, mumax3, and ParaView. Visualize a saved field (colour map + in-plane
quiver) with `examples/plot_ovf.jl` (see Tutorial 3).

---

## 6. Geometry and multiple materials

Build shapes and combine them:

- Primitives: `Cuboid(sx,sy,sz)`, `Rect(sx,sy)`, `Cylinder(diam,h)`, `Circle(d)`,
  `Ellipsoid(dx,dy,dz)`, `Cone(diam,h)`, `Superball(diam,p)`, `Triangle(x0,y0,…)`,
  `Line(p1,p2,diam)`, `Line2D(x1,y1,x2,y2,diam)`, `Cell(mesh,i,j,k)`.
- Layers (for multilayers): `Layer(mesh, k)`, `Layers(mesh, k1, k2)`; slabs
  `XRange`/`YRange`/`ZRange`.
- Transforms: `translate`, `scale`, `rotz`, `rotx`, `roty`, `mirror`,
  `repeat_shape(shape, px, py, pz)` (periodic tiling).
- Set operations: `shapeunion`, `shapeintersect`, `shapediff`, `shapecomplement`,
  `shapexor`.

Assign materials to regions with a `RegionParams`:

```julia
rp = RegionParams(mesh, material("Permalloy"))   # region 0 = Permalloy everywhere
defregion!(rp, 1, Layers(mesh, 3, 5))            # paint the top two layers region 1
setregion!(rp, 1; Msat = 1.1e6, Aex = 1.9e-11)   # give region 1 a different material
sim = Simulation(mesh, rp)
```

At a region interface the exchange coupling uses the harmonic mean of the two
stiffnesses. A region with `Msat = 0` is empty (unfilled geometry). Per-region
averages come from `q_m_region(id)`. Region-wise parameters also run on the GPU
(§7): `togpu` materializes them to per-cell device arrays.

---

## 7. Finite temperature

Thermal fluctuations are modelled by Brown's method: a random field is added to
the effective field, with statistics fixed by the fluctuation–dissipation
theorem. Per cell and component the thermal field is

```
B_therm = η · sqrt( 2 α kB T / (γ Msat V Δt) )
```

where `η` is a standard normal drawn fresh every step, `α` the damping, `T` the
temperature [K], `V` the cell volume, and `Δt` the step. The `1/Δt` makes this a
white-noise increment, so a finite-temperature run needs a **fixed step**;
`runthermal!` integrates the LLG–Langevin equation with a fixed-step stochastic
Heun scheme.

```julia
mesh = Mesh((1, 1, 1), (5e-9, 5e-9, 5e-9))       # a single-domain nanodot
mat  = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5, Ku = 4e5, anisU = (0, 0, 1))
sim  = Simulation(mesh, mat; demag = false)
setmag!(sim, UniformConfig(0, 0, 1))

savequantities!(sim, q_time(), q_m())
runthermal!(sim, 1e-9, 300.0; dt = 2e-15, every = 1e-11)   # 1 ns at 300 K
writetable(sim.table, "thermal.txt")
```

`runthermal!(sim, duration, T; dt, every)` needs a small step — `dt` around
1e-15–1e-14 s — because the thermal kick per step must stay modest; a strongly
damped material equilibrates faster. Region-wise temperatures use each region's
own `α` and `Msat`. `thermalfield!(B, mesh, params, T, dt)` exposes the field
itself if you want to build a custom integrator. Average diagnostics like ⟨mz⟩(T)
follow from time-averaging over a thermalized run — see
[`examples/thermal_demagnetization.jl`](../examples/thermal_demagnetization.jl),
which traces a nanodot's ⟨mz⟩ falling from 1 toward 0 as the temperature rises.

---

## 8. Running on the GPU

On a machine with a CUDA GPU, loading `CUDA` alongside JuliaMag activates a
package extension that adds GPU methods for every field, torque, solver, and
tracker. Because the whole package dispatches on the array type, the *only*
change to a simulation is moving the state and the `World` to the device — the
same source then runs on the GPU.

```julia
using JuliaMag, CUDA
@assert CUDA.functional()

mesh = Mesh((256, 256, 1), (4e-9, 4e-9, 4e-9))
mat  = material("Permalloy")

wg = togpu(World(mesh, mat; demag = true))   # World → device (demag plan included)
mg = togpu(uniform(mesh, (1, 0, 0)))         # state → device (a CuArray)

mn = Minimizer(wg, mg; stopdm = 1e-6)         # relax on the GPU
minimize!(mn)
it = Integrator(wg, mn.m; tend = 1e-9)        # integrate on the GPU
advance!(it, 1e-9)

println(average(state(it)))                   # reductions run on the device
```

Everything runs on the device: the effective field (exchange, anisotropy, the
FFT demagnetization via CUFFT, DMI, Zeeman), the LLG and spin-transfer torques,
the energy minimizer and the time integrator, the finite-temperature thermal
field and `runthermal!`, and the feature trackers (`vortexcore`, `skyrmionpos`,
`domainwallpos`, `topologicalcharge`). Region-wise
(multi-material) parameters work too — `togpu(::World)` materializes a
`RegionParams` into per-cell device arrays, so a multilayer or a patterned sample
runs on the GPU unchanged.

The magnetization stays on the device until you bring it back with `tocpu(m)` (or
`Array(m)`); do that before saving an OVF or feeding a state to a non-GPU tool.
`Float32` on the GPU follows from building the material in `Float32`.

Verify and benchmark with
[`examples/gpu_check.jl`](../examples/gpu_check.jl) (field-by-field CPU vs GPU),
[`examples/gpu_demag_check.jl`](../examples/gpu_demag_check.jl) (demag + full
effective field + speedup), and
[`examples/stdproblem4_gpu.jl`](../examples/stdproblem4_gpu.jl) (Standard Problem
4 run entirely on the GPU). The GPU wins most on the demag FFT, so the speedup
grows with mesh size and with a more capable GPU.

---

## 9. Going further

- **Standalone examples:** [`examples/`](../examples/) has the standard problems
  2, 4, and 5 with comparison plots against mumax3 and OOMMF.
- **Desktop GUI:** a dedicated Qt/Makie window — see [`gui/README.md`](../gui/README.md).
  Set it up once with `julia --project=gui gui/setup.jl`, then
  `julia --project=gui gui/app.jl`.
- **GPU benchmark on Colab:** [`colab/`](../colab/) has a ready notebook (with an
  Open-in-Colab badge) to run the verification and benchmark on a cloud GPU.
- **Paper:** [`paper/juliamag.tex`](../paper/juliamag.tex) documents the design
  and validation.
