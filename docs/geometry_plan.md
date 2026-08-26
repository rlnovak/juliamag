# Plan — richer geometry composition

Extend JuliaMag's shape system to match mumax3's, so complex geometries are built
by composing simple shapes with transforms and set operations, with anti-aliased
(fractional-fill) edges. References: mumax3 `engine/shape.go`, `engine/geom.go`;
grains from `engine/ext_grainboundaries.go`, `engine/ext_make3dgrains.go`,
`engine/ext_makegrains.go`.

Work happens on the `geometry-composition` branch.

## What already exists (parity with mumax3 shape.go)

- **Primitives:** `Cuboid`, `Rect`, `Square`, `Cylinder`, `Circle`, `Ellipsoid`,
  `Ellipse`, `Cone`, `Superball`, `XRange`/`YRange`/`ZRange`, `Layers`/`Layer`,
  `Universe`, `Empty`.
- **Transforms:** `translate`, `scale`, `rotz`.
- **Set operations:** `shapeunion`, `shapeintersect`, `shapediff`,
  `shapecomplement` (a `Shape` is a `(x,y,z)->Bool` predicate in centred metres).

## Gaps vs mumax3, and the plan

### Phase 1 — missing shapes and transforms (additive, `src/shape.jl`) — ✅ DONE

Implemented: `Triangle`, `Line`, `Line2D`, `Cell`, `rotx`, `roty`, `mirror`,
`repeat_shape`, `shapexor` — all exported, all with tests (2319 → 2335). No change
to `Regions`/`defregion!`. Details below.

Pure predicate functions, no architecture change. Each gets a docstring and tests.

| new | signature | notes |
|-----|-----------|-------|
| `Cell` | `Cell(mesh, ix, iy, iz)` | single cell by 1-based index (a `Cuboid` at that centre) |
| `Triangle` | `Triangle(x0,y0, x1,y1, x2,y2)` | 2D, via barycentric/sign test |
| `Line2D` | `Line2D(x1,y1, x2,y2, diam; cap=:round)` | 2D capsule: distance-to-segment ≤ diam/2 |
| `Line` | `Line(p1, p2, diam; cap=:round)` | 3D capsule (distance to a 3D segment) |
| `rotx` | `rotx(shape, θ)` | rotate about x (mirror of `rotz`) |
| `roty` | `roty(shape, θ)` | rotate about y |
| `mirror` | `mirror(shape; x=false,y=false,z=false)` | reflect across chosen planes |
| `repeat_shape` | `repeat_shape(shape, px, py, pz)` | tile with periods `p*` (0 = no repeat); mumax3 `Repeat`. Named `repeat_shape` to avoid clashing with `Base.repeat` |
| `shapexor` | `shapexor(a, b)` | inside exactly one |

Operator forms already read well; keep the named functions as the API and
optionally add `∪`/`∩`/`\`/`!` aliases. `Line`/`Triangle` are the ones users ask
for most (leads, contacts, wedges).

**Deliverable:** `src/shape.jl` grows; `test/test_shape.jl` covers each new shape
(a point known inside and one outside, plus a transform round-trip). No change to
`Regions`/`defregion!`.

### Phase 2 — edge smoothing (fractional fill) — ✅ DONE

Implemented: a `fill::Array{T,3}` field on `RegionParams` (default all 1), the
`setgeometry!(rp, shape; id, edgesmooth)` sampler (`edgesmooth^3` sub-points →
fill 0..1), and `msat(rp,i,j,k) = Msat[region]·fill[cell]` so every field reads
the effective Msat with no kernel change. `isempty_cell` treats fill 0 as empty.
GPU: the fill folds into the materialized Msat in `togpu(::RegionParams)`.
Validated: `edgesmooth=0` reproduces the staircase exactly; a half-covered cell
gets fill ≈ 0.5; the disc moment converges to the analytic area (err +1.3% → 0.01%
from es 0 → 8); GPU effective field matches the CPU to 3.7e-16. Original plan
below.

Today `defregion!` samples only the **cell centre** (mumax3's `edgeSmooth = 0`,
staircase). mumax3 optionally sub-samples `edgeSmooth^3` points per cell to get a
fractional fill `0..1` for boundary cells, which then scales `Msat`.

The clean way in JuliaMag, without disturbing the tested integer-region model:

1. Add an **optional per-cell fill fraction** `fill::Array{T,3}` (default all 1),
   carried alongside the region map. A helper
   `setgeometry!(rp, shape; edgesmooth=0)` samples `shape` with `edgesmooth^3`
   sub-points per cell and records the fraction; `edgesmooth=0` reproduces today's
   centre-only staircase exactly (so nothing regresses).
2. The **effective `Msat`** seen by the field accessors becomes
   `fill[cell] * Msat_region[cell]`. Because every field already reads `Msat`
   through `msat(params,i,j,k)`, this is a one-line change *inside that accessor*
   for a fractional-fill params type — the field kernels are untouched. A cell
   with `fill = 0` is empty exactly as `Msat = 0` is today.
3. GPU: the fill array uploads as one more per-cell device array in
   `GpuRegionParams`; the effective-Msat product is a broadcast.

Sub-sampling loop (mirror of mumax3 `geom.go`), for `S = edgesmooth`:
```
Δ = -c/2 + c/(2S) + (c/S)*d   for d in 0:S-1   (per axis)
fill = (# sub-points inside shape) / S^3
```

**Deliverable:** `setgeometry!`/`deffill!` with `edgesmooth`, a `fill` field,
the accessor product, GPU support, and tests: `edgesmooth=0` reproduces the
staircase; a half-plane through a cell gives `fill ≈ 0.5`; a disc's edge cells get
intermediate fills and the total moment matches the analytic area to
`O(1/edgesmooth)`.

### Phase 3 — polycrystalline grains — ✅ DONE (2D columnar Voronoi)

Implemented in `src/voronoi.jl`: `voronoi!(rp, grainsize, numregions; seed)`
tessellates space into columnar grains (tiles of `grainsize·TILE`, TILE=2, with
Poisson(TILE²) seeds per tile, deterministic per-tile RNG, nearest-seed over a 3×3
tile neighbourhood — the mumax3 `ext_makegrains.go` algorithm), each grain a random
region id. `randomanisotropy!(rp, numregions; Ku, seed)` gives each region a
uniformly-random easy axis — the standard polycrystal setup. Validated:
reproducible per seed, seed-dependent, columnar through z, ids in range,
boundary-cell density rises as grains shrink, random axes uniform-on-sphere and
normalized, and a polycrystal's effective field is finite. Tests: +10.

Still future within Phase 3: 3D grains (`ext_make3dgrains.go`) and grain-boundary
exchange scaling (a reduced Aex on a reassigned boundary region, per
`ext_grainboundaries.go`). Original plan below.

### Phase 3 (original plan) — polycrystalline grains (refs noted)

mumax3 builds grains by Voronoi tessellation and assigns each grain a region
(`ext_makegrains.go`, `ext_make3dgrains.go`) with optional inter-grain exchange
scaling (`ext_grainboundaries.go`) and per-grain height/roughness. Plan:

- `voronoi_regions!(rp, grainsize, nregions; seed)` — tessellate space into grains
  (seeded Poisson points + nearest-seed), round-robin a region id per grain, so
  each grain can carry its own material (anisotropy axis, Msat…).
- Grain-boundary exchange: an optional scalar `< 1` multiplying the harmonic-mean
  stiffness across cells in different grains (a boundary-coupling map).
- 3D grains and `GrainRoughness` (per-grain zmin/zmax) layer on top of Phase 2's
  fractional fill.

This is the largest piece and is deferred; the references above are the port
targets when we take it on.

### Deferred — `ImageShape`

Reading a black/white PNG as a 2D mask (`engine/shape.go` `ImageShape`). Needs an
image dependency (FileIO/PNGFiles). Left for later.

## Order of work

1. **Phase 1** shapes/transforms — small, high value, low risk.
2. **Phase 2** edge smoothing — the architecture piece; keep `edgesmooth=0` the
   default so existing behaviour is byte-for-byte unchanged.
3. Merge `geometry-composition` to `main` after the CPU suite and a GPU
   spot-check pass.
4. **Phase 3** grains — separate follow-up.

Throughout: additive only (no edits to tested field/torque code), CPU tests green
at each step, GPU parity checked on labmac04 for anything the extension touches
(the `fill` product).
