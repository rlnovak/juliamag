# Polycrystalline grains via Voronoi tessellation.
#
# Ported from mumax3 (engine/ext_makegrains.go). Space is tiled into squares of
# side `grainsize · TILE`; each tile carries a Poisson-distributed number of grain
# seeds (mean λ = TILE²), placed uniformly, each seed given a random region id in
# `0:numregions-1`. Every cell is assigned the region of its nearest seed, with a
# 3×3 tile-neighbourhood search so grains straddle tile boundaries cleanly. The
# tessellation is columnar (grains extend through z); the seed placement is
# deterministic per tile from the master `seed`, so the geometry is reproducible.
#
# Assigning a random region per grain lets a polycrystal be built by giving each
# region its own material (e.g. a randomly oriented easy axis) with `setregion!`.

using Random

const _TILE = 2                       # tile side, measured in grains
const _LAMBDA = _TILE * _TILE         # expected seeds per tile (Poisson mean)

# Knuth's algorithm for a Poisson(λ) sample.
function _poisson(rng, λ::Float64)
    L = exp(-λ); k = 0; p = 1.0
    while true
        k += 1
        p *= rand(rng)
        p <= L && return k - 1
    end
end

# Deterministic RNG for a tile, so a tile's seeds are the same however the mesh is
# scanned (the 3×3 neighbour search revisits tiles).
@inline function _tilerng(tx::Int, ty::Int, master::UInt64)
    h = (UInt64(ty % (1<<24) + (1<<24)) << 24) +
        (UInt64(tx % (1<<24) + (1<<24)) ⊻ master)
    return MersenneTwister(h)
end

# Grain seeds of one tile: vectors of (x, y) centres [m] and region ids.
function _tileseeds(tx::Int, ty::Int, tilesize::Float64, numregions::Int, master::UInt64)
    rng = _tilerng(tx, ty, master)
    n = _poisson(rng, Float64(_LAMBDA))
    xs = Float64[]; ys = Float64[]; rs = UInt8[]
    for _ in 1:n
        push!(xs, (tx + rand(rng)) * tilesize)      # absolute position [m]
        push!(ys, (ty + rand(rng)) * tilesize)
        push!(rs, UInt8(rand(rng, 0:numregions-1)))
    end
    return xs, ys, rs
end

"""
    voronoi!(regions, grainsize, numregions; seed=0) -> regions

Fill `regions` with a Voronoi tessellation into grains of typical size
`grainsize` [m], each grain assigned a random region id in `0:numregions-1`.
Grains are columnar (constant through z). `seed` selects the (reproducible)
tessellation. Use with per-region parameters to build a polycrystal.
"""
function voronoi!(regions::Regions, grainsize::Real, numregions::Integer; seed::Integer = 0)
    1 <= numregions <= MAXREGIONS || throw(ArgumentError("numregions must be 1–$MAXREGIONS"))
    mesh = regions.mesh
    Nx, Ny, Nz = mesh.size
    tilesize = Float64(grainsize) * _TILE
    master = UInt64(seed) ⊻ 0x9e3779b97f4a7c15
    nr = Int(numregions)

    # Grains are columnar → assign the first layer, then copy through z.
    @inbounds for j in 1:Ny, i in 1:Nx
        x, y, _ = cellcenter(mesh, i, j, 1)
        tx = floor(Int, x / tilesize); ty = floor(Int, y / tilesize)
        best = Inf; bestr = UInt8(0)
        for sx in tx-1:tx+1, sy in ty-1:ty+1
            xs, ys, rs = _tileseeds(sx, sy, tilesize, nr, master)
            for s in eachindex(xs)
                d2 = (x - xs[s])^2 + (y - ys[s])^2
                if d2 < best
                    best = d2; bestr = rs[s]
                end
            end
        end
        for k in 1:Nz
            regions.id[i, j, k] = bestr
        end
    end
    return regions
end

"""
    voronoi!(rp::RegionParams, grainsize, numregions; seed=0) -> rp

Tessellate the region map of a `RegionParams` into Voronoi grains (see
[`voronoi!`](@ref) for `Regions`). Then give each region its own material with
`setregion!`, e.g. a randomly oriented anisotropy axis per grain.
"""
function voronoi!(rp::RegionParams, grainsize::Real, numregions::Integer; seed::Integer = 0)
    voronoi!(rp.regions, grainsize, numregions; seed = seed)
    return rp
end

"""
    randomanisotropy!(rp, numregions; Ku, seed=0) -> rp

Give each of regions `0:numregions-1` the anisotropy constant `Ku` with a
uniformly random easy-axis direction — the common polycrystalline setup when
combined with [`voronoi!`](@ref). Reproducible from `seed`.
"""
function randomanisotropy!(rp::RegionParams{T}, numregions::Integer;
                           Ku::Real, seed::Integer = 0) where {T}
    rng = MersenneTwister(UInt64(seed) ⊻ 0xd1b54a32d192ed03)
    for id in 0:Int(numregions)-1
        # Uniform point on the sphere.
        z = 2rand(rng) - 1; φ = 2π * rand(rng); r = sqrt(max(0.0, 1 - z^2))
        setregion!(rp, id; Ku = Ku, anisU = (r*cos(φ), r*sin(φ), z))
    end
    return rp
end
