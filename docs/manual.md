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

The checks of JuliaMag against analytic results, OOMMF, mumax3, MicroMagnetic.jl, magnum.np and an
independent finite-element solution are repeated step by step in the
[validation tutorials](validation.md).

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
[Going further](#10-going-further).

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
J/m², bulk DMI `Dbulk` is J/m² too.

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
    push!(mxs, average(sim)[1])                      # over the disc, not the box
end

writetable(sim.table, "disc_hysteresis.txt")
@printf("⟨mx⟩ range [%.3f, %.3f]\n", minimum(mxs), maximum(mxs))
```

Plotting `mxs` against `Bs` gives the S-shaped vortex loop: `⟨mx⟩ = 0` at zero
field (the centred vortex), a nearly linear reversible rise as the core is pushed
toward the edge, then an abrupt jump to the nearly saturated branch
(`⟨mx⟩ ≈ ±0.99`) where the vortex annihilates, and a jump back as it renucleates
on the return branch. `average(sim)` averages over the disc's cells only (mumax3's
table `m`); `average(sim.m)` would average over the whole rectangular box and read
≈ ±0.78 at saturation, diluted by the disc's area fraction π/4. The
`disc_hysteresis.txt` table holds `Bx` alongside `⟨mx,my,mz⟩` for each field
step. This is a sizeable run (40k cells over 153 field steps; ~5 min with 8 threads). A
ready-to-run version (plot: `examples/plot_disc_hysteresis.py`) is
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

The same run with the general driver: `setcurrent!(sim, J)` (Zhang–Li by default),
`tableautosave!(sim, 20e-12)` and `run!(sim, 1e-9)`, with an adaptive solver.
`runcurrent!` is the older fixed-step route, kept for existing scripts.

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

## 4. Running a simulation: solvers, currents and output

A `Simulation` script follows the shape of a mumax3 input file. Each mumax3 verb
has a JuliaMag counterpart:

| mumax3 | JuliaMag |
|--------|----------|
| `m = vortex(1, 1)` | `setmag!(sim, VortexConfig(mesh))` |
| `B_ext = vector(bx, by, bz)` | `setexternalfield!(sim, (bx, by, bz))` |
| `J = …`, `FixedLayer`, `Pol`, `Lambda`, `EpsilonPrime`, `xi` | `setcurrent!(sim, J; …)` (with the material's `pol`, `xi`, `lambda`, `epsilonPrime`) |
| `Temp = 300` | `settemperature!(sim, 300)` |
| `SetSolver(5)`, `MaxErr`, `FixDt`, `MinDt`, `MaxDt`, `Headroom` | `setsolver!(sim, :rk45; maxerr, fixdt, mindt, maxdt, headroom)` |
| `minimize()` | `relax!(sim)` |
| `relax()` | `relaxllg!(sim)` |
| `run(1e-9)`, `steps(n)`, `runwhile(cond)` | `run!(sim, 1e-9)`, `steps!(sim, n)`, `runwhile!(sim, cond)` |
| `B_ext = vector(0, 0, f(t))` | `setdrive!(sim, (w, t) -> setexternalfield!(w, (0, 0, f(t))))` |
| `TableAdd(maxTorque)`, `TableAutosave(5e-12)`, `TableSave()` | `tableadd!(sim, q_maxtorque())`, `tableautosave!(sim, 5e-12)`, `tablesave!(sim)` |
| `AutoSave(m, 1e-10)`, `Save(m)` | `autosave!(sim, 1e-10)`, `save!(sim)` |
| output directory `script.out/` | `Simulation(…; outdir = "run.out")` or `setoutdir!(sim, dir)` |

A complete current-driven run:

```julia
using JuliaMag
mesh = Mesh((64, 64, 8), (2e-9, 2e-9, 2e-9))
mat  = Material(Msat = 8e5, Aex = 1.08e-11, alpha = 0.01, pol = 0.37, lambda = 1.0,
                epsilonPrime = 0.01)
rp = RegionParams(mesh, mat); setregion!(rp, 0; Msat = 0.0)
setgeometry!(rp, Cylinder(128e-9, 32e-9); id = 1, edgesmooth = 4)

sim = Simulation(mesh, rp; outdir = "disc.out")
setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))
relax!(sim)

tableadd!(sim, q_maxtorque()); tableadd!(sim, q_vortexcore())
tableautosave!(sim, 5e-12)                      # streamed to disc.out/table.txt
autosave!(sim, 1e-9)                            # disc.out/m000000.ovf, m000001.ovf, …
setcurrent!(sim, (0, 0, 3.1e12); zhangli = false, slonczewski = true,
            fixedlayer = (cos(20π/180), sin(20π/180), 0))
run!(sim, 40e-9)                                # adaptive Dormand–Prince, MaxErr 1e-5
```

**Solvers.** `run!` integrates with the solver chosen by name:

| name | method | error estimate |
|------|--------|----------------|
| `:rk45` (default) | Dormand–Prince, FSAL (mumax3 solver 5) | embedded 4th order |
| `:rk23` | Bogacki–Shampine, FSAL (mumax3 solver 3) | embedded 2nd order |
| `:rk4` | classic Runge–Kutta (mumax3 solver 4) | \|k1 − k4\|·Δt |
| `:heun` | Heun (mumax3 solver 2) | embedded Euler |
| `:euler` | Euler (mumax3 solver 1) | adaptive: Δt = MaxErr / max torque |
| `:tsit5` | Tsitouras 5(4), OrdinaryDiffEq | `abstol`, `reltol` |

The first five reproduce mumax3's stages, normalizations, error norm and
step-size control operation for operation. The step is accepted when
`err < maxerr`, and the next one is `Δt·clamp(headroom·(maxerr/err)^(1/p), ½, 2)`,
clipped so that no step crosses the end of the run. `setsolver!(sim, :rk4;
fixdt = 5e-14)` fixes the step. Keywords given to `run!` (`run!(sim, 1e-9; solver =
:rk23, maxerr = 1e-6)`) apply to that run only. `sim.nsteps`, `sim.nundone`,
`sim.nevals` and `sim.peakerr` count the work, as mumax3's `NSteps`, `NUndone`,
`NEvals` and `PeakErr` do.

**Spin-polarized current.** `setcurrent!(sim, J; zhangli = true, slonczewski =
false, fixedlayer, thickness, params)` stores a uniform current density `J`
[A/m²]. Every solver then adds the chosen spin torques to the LLG right-hand side
(`llgrhs!`). This includes the adaptive ones: a current-driven run no longer needs
a hand-written fixed-step loop. The Slonczewski torque uses `J[3]`, the
fixed-layer polarization `fixedlayer` and the free-layer `thickness` (default:
the mesh thickness).

Unlike mumax3, each torque is switched on explicitly. mumax3 adds the Zhang–Li
term whenever `J` is set unless `DisableZhangLiTorque = true`, and with a
perpendicular current that term shifts a vortex gyration frequency by about 4%
(paper, Sec. "Current-driven vortex in a disc"). The STT parameters come from one
`Material`: the material itself, or for a `RegionParams` the first magnetic
region (`sttparams`). `setcurrent!(sim, nothing)` removes the current.

**Fields and currents in space and time.** The applied field is the uniform `B_ext` plus terms,
each a map on the mesh times a function of time (mumax3's `B_ext.Add(mask, f(t))`); the same for the
current density of the spin torques.

```julia
# a Gaussian pulse, in space and in time
pulse = fieldmap(mesh) do x, y, z                    # tesla per unit of f
    (exp(-((x - 40e-9)^2 + y^2) / (2 * (30e-9)^2)), 0, 0)
end
addfield!(sim, pulse, t -> 0.02exp(-((t - 0.2e-9) / 50e-12)^2))

# a field in one region, oscillating, and a constant one
addfield!(sim, regionmap(sim, 2, (0, 1, 0)), t -> 0.008cos(2π * 5e9 * t))
addfield!(sim, regionmap(sim, 1, (0, 0, 1)), 0.002)

# a current through a nanocontact: the torques first, then the map (A/m²)
setcurrent!(sim, (0, 0, 0); zhangli = false, slonczewski = true, fixedlayer = (1, 0, 0))
addcurrent!(sim, regionmap(sim, 1, (0, 0, 1e12)))
```

- The functions are evaluated at the time of every stage of the solvers. The maps stay on the GPU,
  with nothing copied from the host during a run.
- Every part of the code that uses the field sees the terms: the effective field, the Zeeman and
  total energies, the minimizer, GNEB and the eigenmodes, at the time last set.
- `fieldmap(mesh, shape, v)` gives `v` inside a shape; `clearfields!` and `clearcurrents!` remove the
  terms.
- A term adds to the uniform part. mumax3's `B_ext.SetRegion(r, v)` replaces it in region r instead.
- With a current, each cell takes the spin-torque parameters of its region (Msat, pol, xi, lambda,
  epsilonPrime, alpha), as mumax3. Pass `params` to `setcurrent!` for one set of parameters everywhere.
- `setdrive!(sim, (w, t) -> …)` changes the uniform field or current with time.
- The older integrators (`:tsit5`, `runthermal!`, `Solver`) refuse functions of time; the last two
  also refuse spin torques.

**Currents from anisotropic magnetoresistance.** Instead of prescribing the current density, JuliaMag
can compute it. It solves for the electric potential in the sample, fed through contacts, with a
resistivity that depends on the local magnetization:

    ∇·(σ(m) ∇φ) = 0,   j = −σ(m) ∇φ,   ρ = ρ⊥ I + (ρ∥ − ρ⊥) m mᵀ.

The current then flows around the domain walls and the vortex cores, and the planar Hall field
appears where m is oblique to j.

```julia
# a disc (region 1) between two non-magnetic leads (4, 5) with contacts at their ends (2, 3)
setconduction!(sim; rho_par = 204e-9, rho_perp = 200e-9, regions = [1, 2, 3, 4, 5])
for r in 2:5
    setresistivity!(sim, r; rho_par = 200e-9)        # isotropic leads (ρ∥ = ρ⊥)
end
addcontact!(sim, 2; current = 2e-3)                  # 2 mA into the sample (a function of t is allowed)
addcontact!(sim, 3; potential = 0.0)                 # the ground
setcurrent!(sim, (0, 0, 0); zhangli = true)          # the spin torques take the computed j
tableadd!(sim, q_resistance()); tableadd!(sim, q_contacts(2))
autosave!(sim, 50e-12; name = "j", field = currentmap)
autosave!(sim, 50e-12; name = "phi", field = potentialmap)
run!(sim, 3e-9)
```

- **When it is solved.** The potential is solved at every stage of the solvers, for the state of
  that stage, and at every output. The current density of the conduction adds to `J` and to the
  terms of `addcurrent!`. `solveconduction!(sim)` returns the solution for the present state:
  - `phi`, the potential [V];
  - `j`, the current density [A/m²];
  - `contacts`, the potential `V` and the current `I` of each contact.
- **Which cells conduct.**
  - Every region with Msat conducts, by default; pass `regions` to choose them instead.
  - Set resistivities per region with `setresistivity!`; `Inf` makes a region an insulator.
  - A region without Msat (a lead) conducts isotropically with ρ⊥.
  - A partially filled cell (edge smoothing) conducts with its fill times σ. Its current density is
    that of the material.
  - A change of the regions, the geometry or Msat is taken into account at the next solve.
- **Contacts.**
  - A contact is a region held at a potential, or fed a total current, each constant or a function
    of time.
  - Contacts are perfect conductors. Their cells carry no current density, so make them
    non-magnetic.
  - Without a contact held at a potential, the last current contact is the ground (0 V), and the
    currents must sum to zero.
  - Cells not connected to any contact become insulators, with a warning.
  - A contact cut off from the contacts held at a potential is an error.
- **Precision and device.** The solver works in Float64 on the CPU, for every precision and device.
  On the GPU, m is copied to the host and j back at each stage. A solve costs about one to three
  evaluations of the effective field: BiCGSTAB, preconditioned by the isotropic problem, starting
  from the last solution.
- **Not included.** The Oersted field of j(r) is not computed.
- **Validation.** `gpu_validation/amr/` has the analytic strip, an independent finite-element
  reference (agreement 4e-9 in R) and the linearity of the AMR effect on a gyrating vortex.

**Output.**
- **The table.** It always starts with `t` and ⟨m⟩, the magnet average (fill
  weighted, empty cells excluded, as mumax3's table `m`); add columns with
  `tableadd!`.
- **The save schedule.** `tableautosave!(sim, Δt)` and `autosave!(sim, Δt)` follow
  mumax3's rule: a row or an OVF file is written after the first step at which
  `t − t₀ ≥ (n+1)·Δt`, starting with the current state. With an adaptive step the
  rows therefore fall on the steps, not on exact multiples of `Δt`, as in mumax3.
  With a fixed step that divides `Δt` they land on the multiples exactly.
- **No schedule.** Without an autosave interval, `run!` saves a row at its start
  and end.
- **Files.** With an output directory, the table is streamed to `table.txt` as
  rows are saved, and OVF files are numbered per name from `000000`.
- **The run log.** Every `run!` prints its wall-clock start and end time, the
  device (CPU model, or GPU name and compute capability), the precision, the
  solver and, at the end, the steps, undone steps and evaluations. Pass
  `verbose = false` to `Simulation` to silence it, or `log = io` to redirect it.

The available table quantities:

| Quantity | Columns | Meaning |
|----------|---------|---------|
| `q_time()` | `t` | simulated time [s] (always present) |
| `q_m()` | `mx, my, mz` | magnet-averaged magnetization (always present) |
| `q_m_region(id)` | `mx, my, mz` | averaged over one region |
| `q_energy()` | `E_total` | total energy [J] |
| `q_exchangeenergy()`, `q_demagenergy()`, `q_zeemanenergy()`, `q_anisenergy()` | one each | per-term energies |
| `q_maxtorque()` | `maxTorque` | max \|dm/dt\|/γ of the full right-hand side (spin torques included) [T], as mumax3 |
| `q_Bext()` | `B_extx,y,z` | applied field [T], averaged over the mesh (with the terms) |
| `q_J()` | `Jx,y,z` | current density [A/m²], averaged over the mesh (prescribed `J` and terms) |
| `q_contacts(n)` | `V1, I1, …, Vn, In` | potential [V] and current fed into the sample [A] of each contact (conduction) |
| `q_resistance(a=1, b=2)` | `R` | (V_a − V_b)/I_a [Ω], the two-terminal resistance (conduction) |
| `q_vortexcore()` | `x, y, z, polarity` | vortex core (mumax3 `ext_corepos`) |
| `q_skyrmionpos()` | `x, y, z` | skyrmion centroid |
| `q_dwpos()` | `x, y, z` | domain-wall position |
| `q_topocharge()` | `Q` | topological charge |

A custom column is a `Quantity(name, unit, ncomp, (world, m, t) -> values;
device = false)`. Set `device = true` only if the function also works on a GPU
state; otherwise a GPU simulation evaluates it on a host copy.
`savequantities!(sim, q…; every = Δt)` remains as a one-call shorthand for
`tableadd!` + `tableautosave!`. `writetable(sim, "out.txt")` writes the in-memory
table.

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
stiffnesses, and the DMI the mean of the two `D` that mumax3 uses (harmonic if they
have the same sign, 0 if one is 0). A region with `Msat = 0` is empty (unfilled geometry). Per-region
averages come from `q_m_region(id)`. Region-wise parameters also run on the GPU
(§8): `togpu` materializes them to per-cell device arrays.

**Anti-aliased edges.** `defregion!` samples the cell centre, giving a staircase
boundary. For a smooth edge use `setgeometry!(rp, shape; id, edgesmooth)`, which
sub-samples `edgesmooth^3` points per cell and stores the fraction inside the
shape as a **fill** `∈ [0,1]`. The fill scales the moment of the cell
(`fill · Msat[region]`) in the demagnetizing field, the energies and the thermal
field, so a boundary cell contributes proportionally, and the total moment converges
to the true shape area as `edgesmooth` grows (`edgesmooth = 0` reproduces the plain
staircase). The exchange, DMI and anisotropy fields use the `Msat` of the region, as
in mumax3 (whose thermal field uses it too). The GPU follows the same rules.

```julia
rp = RegionParams(mesh, material("Permalloy")); setregion!(rp, 0; Msat = 0.0)
setgeometry!(rp, Cylinder(500e-9, 1e6); id = 1, edgesmooth = 8)   # smooth disc edge
```

**Polycrystalline grains.** `voronoi!(rp, grainsize, numregions; seed)` tessellates
the sample into columnar Voronoi grains, each assigned a random region, and
`randomanisotropy!(rp, numregions; Ku, seed)` gives each region a random easy
axis — the standard polycrystal:

```julia
rp = RegionParams(mesh, material("Cobalt"))
voronoi!(rp, 30e-9, 200; seed = 1)                 # ~30 nm grains
randomanisotropy!(rp, 200; Ku = 5e5, seed = 1)     # random easy axis per grain
```

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

The same run through the general driver, with the table autosave and any solver
that takes a fixed step (Heun is mumax3's choice for thermal runs):

```julia
settemperature!(sim, 300.0)
tableautosave!(sim, 1e-11)
run!(sim, 1e-9; solver = :heun, fixdt = 2e-15)   # an adaptive step is refused at T > 0
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

A whole `Simulation` moves with `togpu(sim)`. The script is otherwise unchanged:
`relax!`, `run!` (every solver), `setcurrent!`, the table and the OVF output work
on the device. Table columns without a GPU method, such as the energies, are
evaluated on a host copy of the state. `tocpu(sim)` brings the simulation back.

```julia
sim = togpu(Simulation(mesh, rp; outdir = "run.out"))
setmag!(sim, VortexConfig(mesh)); relax!(sim)
tableadd!(sim, q_energy()); tableautosave!(sim, 5e-12)
run!(sim, 1e-9)        # log: "device: GPU 0 (Quadro P1000, compute capability 6.1, …)"
```

The magnetization stays on the device until you bring it back with `tocpu(m)` (or
`Array(m)`); do that before saving an OVF, computing an energy, or feeding a state
to a non-GPU tool (the energy functions and `clearempty!` are CPU-only — call them
on `Array(m)`). `Float32` on the GPU follows from building the material in
`Float32`.

For a geometry with empty cells (`Msat = 0`, or a partially filled edge), clear
the state's empty cells **before** moving it to the device: `clearempty!(m, rp)`
then `togpu(m)`. `setmag!` on a `Simulation` already does this, so the normal
workflow is safe; only a hand-built `setconfig` array needs the explicit clear.
Empty cells left nonzero on the GPU would seed a spurious moment there.

Verify and benchmark with
[`examples/gpu_check.jl`](../examples/gpu_check.jl) (field-by-field CPU vs GPU),
[`examples/gpu_demag_check.jl`](../examples/gpu_demag_check.jl) (demag + full
effective field + speedup), and
[`examples/stdproblem4_gpu.jl`](../examples/stdproblem4_gpu.jl) (Standard Problem
4 run entirely on the GPU). The GPU wins most on the demag FFT, so the speedup
grows with mesh size and with a more capable GPU.

---

## 9. Energy barriers and eigenmodes

**Minimum-energy paths.** `GNEB` joins two relaxed states by a chain of images and relaxes
it; the highest image climbs to the saddle point.

```julia
mesh = Mesh((32, 1, 1), (2e-9, 2e-9, 2e-9))
mat  = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5, Ku = 5e5, anisU = (1, 0, 0))
w    = World(mesh, mat; demag = false, Bext = (-0.375, 0, 0))
up, down = uniform(mesh, (1, 0, 0)), uniform(mesh, (-1, 0, 0))
wall = zeromag(Float64, mesh)                        # a waypoint: a wall in the middle
for i in 1:32
    θ = π / (1 + exp(-(i - 16) / 2.5))
    wall[:, i, 1, 1] .= (cos(θ), sin(θ), 0.02)
end
neb = GNEB(w, [up, normalize!(wall), down]; nimages = 13)
relax!(neb; stopforce = 1e-6)                        # T
barrier(neb)                                         # J
saveimages(neb, "band")                              # OVF files and neb.txt
```

Forces and the spring constant are in tesla. The initial path matters: it must not be
confined by a symmetry that the true path breaks. Check that the result does not change with
more images, and that the saddle is one:

```julia
s = certifysaddle(neb)
s.index        # number of negative curvatures of the energy: 1 for a transition state
s.values       # the lowest curvatures [T]; near-zero ones are symmetries
```

`curvatures(w, m)` gives the same for any state. A relaxed state with a negative curvature is
not a minimum: the minimizer stopped on a symmetric stationary point.

**Eigenmodes.** `eigenmodes` linearizes the equation of motion about an equilibrium, which has to
be relaxed tightly:

```julia
film = Mesh((12, 12, 1), (5e-9, 5e-9, 5e-9))
wf   = World(film, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.01); Bext = (0.2, 0.05, 0))
mn   = Minimizer(wf, uniform(film, (1, 0, 0)); stopdm = 1e-10)
minimize!(mn)
em = eigenmodes(wf, mn.m; nmodes = 4)
em.frequencies                                       # Hz
em.linewidths                                        # decay rates of the amplitude, 1/s
savemode("modes", em, 1, film)                       # complex mode δm as two OVF files
```

Small systems are solved as dense matrices, larger ones without a matrix (`method = :lanczos`),
also on the GPU (`togpu`). Edge-smoothed geometries (partially filled cells) are limited to
small systems, and states with a zero mode (a texture that can be translated in a periodic film)
are refused. Both features work in Float64 by default; in Float32 the tolerances are wider.

---

## 10. Going further

- **Standalone examples:** [`examples/`](../examples/) has the standard problems
  2, 4, and 5 with comparison plots against mumax3 and OOMMF.
- **Validation tutorials:** [`validation.md`](validation.md) repeats every check against other codes
  and analytic results, with the scripts and the numbers they give.
- **Desktop GUI:** a Qt/Makie window — see [`gui/README.md`](../gui/README.md). Set it up once with
  `julia --project=gui gui/setup.jl`, then `julia -t auto --project=gui gui/app.jl` (it needs at
  least two threads: the simulation runs on a worker thread, the window on the main one). It opens
  a script that leaves a `sim::Simulation` (or builds a simple one), relaxes it, runs it with its
  own solver and output, pauses or stops between steps, and sets the applied field during a run.
  It shows ⟨m⟩(t), E(t), the state of the run and one layer of m.
- **GPU benchmark on Colab:** [`colab/`](../colab/) has a ready notebook (with an
  Open-in-Colab badge) to run the verification and benchmark on a cloud GPU.
- **Paper:** [`paper/juliamag.tex`](../paper/juliamag.tex) documents the design
  and validation.
