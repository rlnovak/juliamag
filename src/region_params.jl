# Per-region material parameters.
#
# RegionParams is the multi-material counterpart of Material (params.jl). It pairs
# a Regions map (which cell belongs to which region) with per-region lookup
# tables for every material parameter. The field routines read it through the
# same accessors (msat, aex, ...) as a scalar Material — multiple dispatch selects
# the table lookup here versus the constant there — so no field code changes.
#
# Build one from a default Material, then override parameters per region:
#
#     rp = RegionParams(mesh, permalloy)          # region 0 = permalloy everywhere
#     defregion!(rp, 1, Layers(mesh, 3, 5))       # paint the top layers region 1
#     setregion!(rp, 1; Msat = 1.4e6, Aex = 2e-11)# give region 1 its own material

"""
    RegionParams(mesh, default::Material) -> RegionParams

Per-region material parameters over `mesh`, every region initialized to the
`default` material and every cell assigned to region 0. Assign geometry with
`defregion!` and per-region values with `setregion!`.
"""
struct RegionParams{T<:AbstractFloat} <: AbstractParams
    regions::Regions
    Msat::Vector{T}
    Aex::Vector{T}
    alpha::Vector{T}
    Ku::Vector{T}
    anisU::Vector{NTuple{3,T}}
    Dind::Vector{T}
    Dbulk::Vector{T}
    pol::Vector{T}
    xi::Vector{T}
    lambda::Vector{T}
    epsilonPrime::Vector{T}
    # Per-cell fill fraction in [0,1]: 1 = fully inside the geometry, <1 = a
    # boundary cell partially covered (edge smoothing), 0 = empty. The moment of
    # a cell (demag, energies, thermal field) is fill[cell] · Msat[region]; exchange,
    # DMI and anisotropy fields use Msat[region]. All ones by default, so a geometry
    # painted with defregion! (centre sampling) behaves exactly as before.
    fill::Array{T,3}
end

function RegionParams(mesh::Mesh, default::Material{T}) where {T}
    n = MAXREGIONS
    fill_(x) = fill(x, n)
    RegionParams{T}(Regions(mesh),
                    fill_(default.Msat), fill_(default.Aex), fill_(default.alpha),
                    fill_(default.Ku), fill_(default.anisU),
                    fill_(default.Dind), fill_(default.Dbulk),
                    fill_(default.pol), fill_(default.xi),
                    fill_(default.lambda), fill_(default.epsilonPrime),
                    ones(T, mesh.size...))
end

Base.eltype(::RegionParams{T}) where {T} = T

"""
    RegionParams{T}(rp::RegionParams)

The same region map and per-region tables in floating-point precision `T` (e.g.
`RegionParams{Float32}(rp)` for a single-precision run). The region map is shared;
the parameter tables and the cell fill are converted.
"""
RegionParams{T}(rp::RegionParams) where {T<:AbstractFloat} =
    RegionParams{T}(rp.regions, T.(rp.Msat), T.(rp.Aex), T.(rp.alpha), T.(rp.Ku),
                    [NTuple{3,T}(u) for u in rp.anisU], T.(rp.Dind), T.(rp.Dbulk),
                    T.(rp.pol), T.(rp.xi), T.(rp.lambda), T.(rp.epsilonPrime),
                    T.(rp.fill))

"""
    setregion!(rp, id; Msat, Aex, alpha, Ku, anisU, Dind, Dbulk, pol, xi, lambda, epsilonPrime)

Set one or more material parameters for region `id`. Only the keywords given are
changed; the rest keep their current (default) values. `anisU` is normalized.
"""
function setregion!(rp::RegionParams{T}, id::Integer;
                    Msat = nothing, Aex = nothing, alpha = nothing,
                    Ku = nothing, anisU = nothing, Dind = nothing, Dbulk = nothing,
                    pol = nothing, xi = nothing, lambda = nothing,
                    epsilonPrime = nothing) where {T}
    0 <= id < MAXREGIONS || throw(ArgumentError("region id must be 0–$(MAXREGIONS-1), got $id"))
    r = id + 1                       # 1-based table index
    Msat !== nothing && (rp.Msat[r] = T(Msat))
    Aex !== nothing && (rp.Aex[r] = T(Aex))
    alpha !== nothing && (rp.alpha[r] = T(alpha))
    Ku !== nothing && (rp.Ku[r] = T(Ku))
    if anisU !== nothing
        u = NTuple{3,T}(anisU); nrm = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
        nrm > 0 || throw(ArgumentError("anisU must not be the zero vector"))
        rp.anisU[r] = u ./ nrm
    end
    Dind !== nothing && (rp.Dind[r] = T(Dind))
    Dbulk !== nothing && (rp.Dbulk[r] = T(Dbulk))
    pol !== nothing && (rp.pol[r] = T(pol))
    xi !== nothing && (rp.xi[r] = T(xi))
    lambda !== nothing && (rp.lambda[r] = T(lambda))
    epsilonPrime !== nothing && (rp.epsilonPrime[r] = T(epsilonPrime))
    return rp
end

"Paint region `id` into the geometry (forwards to the Regions map)."
defregion!(rp::RegionParams, id::Integer, shape::Shape) = (defregion!(rp.regions, id, shape); rp)
defregioncell!(rp::RegionParams, id::Integer, i, j, k) = (defregioncell!(rp.regions, id, i, j, k); rp)

"""
    setgeometry!(rp, shape; id=1, edgesmooth=0) -> rp

Paint `shape` as region `id` with an anti-aliased edge. Each cell is sampled at
`edgesmooth^3` sub-points (mumax3's edge smoothing); the fraction inside `shape`
becomes the cell's fill (0..1), which scales the moment of the cell. A boundary cell
half-covered gets `fill ≈ 0.5`; a fully covered cell gets 1. `edgesmooth = 0` (or
1) samples the cell centre only — the plain staircase geometry, identical to
`defregion!`.

Cells with fill `> 0` are assigned region `id` and that fill; cells fully outside
are left untouched (they keep their current region and fill). To cut a shape out
of empty space, first make the background empty with `setregion!(rp, 0; Msat = 0)`
— the outside cells are then empty through Msat = 0. Build a geometry by painting
from the background outward.
"""
function setgeometry!(rp::RegionParams{T}, shape::Shape;
                      id::Integer = 1, edgesmooth::Integer = 0) where {T}
    0 <= id < MAXREGIONS || throw(ArgumentError("region id must be 0–$(MAXREGIONS-1), got $id"))
    mesh = rp.regions.mesh
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    S = max(1, Int(edgesmooth))                 # sub-samples per axis (1 = centre only)
    invS3 = one(T) / T(S^3)
    # Sub-sample offsets within a cell: -c/2 + c/(2S) + (c/S)·d, d = 0..S-1.
    offx = ntuple(d -> (-cx/2 + cx/(2S) + (cx/S)*(d-1)), S)
    offy = ntuple(d -> (-cy/2 + cy/(2S) + (cy/S)*(d-1)), S)
    offz = ntuple(d -> (-cz/2 + cz/(2S) + (cz/S)*(d-1)), S)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x0, y0, z0 = cellcenter(mesh, i, j, k)
        inside = 0
        for dz in 1:S, dy in 1:S, dx in 1:S
            shape(x0 + offx[dx], y0 + offy[dy], z0 + offz[dz]) && (inside += 1)
        end
        if inside > 0
            rp.regions.id[i, j, k] = UInt8(id)
            rp.fill[i, j, k] = inside * invS3
        end
    end
    return rp
end

"""
    isempty_cell(params, i, j, k) -> Bool

True if the cell carries no material (Msat = 0), i.e. it lies in unfilled
geometry. A scalar Material is never empty.
"""
@inline isempty_cell(m::Material, i, j, k) = false
@inline isempty_cell(rp::RegionParams, i, j, k) =
    @inbounds rp.Msat[_rid(rp, i, j, k)] == 0 || rp.fill[i, j, k] == 0

"""
    clearempty!(m, params) -> m

Set the magnetization to zero in every cell with no material (Msat = 0). Empty
cells then produce no torque and no demag source — the way mumax3 represents
non-rectangular geometry on a rectangular mesh. Call after building an initial
state on a `RegionParams` with unfilled regions.
"""
function clearempty!(m::AbstractArray{T,4}, params::AbstractParams) where {T}
    @inbounds for k in axes(m, 4), j in axes(m, 3), i in axes(m, 2)
        if isempty_cell(params, i, j, k)
            m[1, i, j, k] = 0; m[2, i, j, k] = 0; m[3, i, j, k] = 0
        end
    end
    return m
end

# --- Accessors: look up the region of the cell, then the per-region table ---
# The +1 converts the 0-based region id to a 1-based vector index.
@inline _rid(rp::RegionParams, i, j, k) = rp.regions.id[i, j, k] + 1
# Effective Msat is the region's Msat scaled by the cell's fill fraction, so a
# partially-covered boundary cell contributes proportionally (edge smoothing) and
# an empty cell (fill 0 or Msat 0) contributes nothing.
@inline msat(rp::RegionParams, i, j, k)     = @inbounds rp.Msat[_rid(rp, i, j, k)] * rp.fill[i, j, k]
# Region Msat WITHOUT the fill factor. mumax3 scales the magnetization by the
# cell fill only where the moment enters as a source (the demag input,
# engine/demag.go SetMFull) and in the energies; the exchange, DMI and anisotropy
# fields divide by the region's full Msat (cuda/amul.h inv_Msat uses the Msat LUT,
# never the geometry vol). The fill-scaled msat() in those prefactors would inflate
# the field by 1/fill at partially-covered boundary cells. (The thermal field uses
# msat(), the moment of the cell: fluctuation-dissipation. mumax3 uses Msat there.)
@inline msat_region(rp::RegionParams, i, j, k) = @inbounds rp.Msat[_rid(rp, i, j, k)]
@inline fillof(rp::RegionParams, i, j, k)   = @inbounds rp.fill[i, j, k]
@inline aex(rp::RegionParams, i, j, k)      = @inbounds rp.Aex[_rid(rp, i, j, k)]
@inline alphaof(rp::RegionParams, i, j, k)  = @inbounds rp.alpha[_rid(rp, i, j, k)]
@inline ku(rp::RegionParams, i, j, k)       = @inbounds rp.Ku[_rid(rp, i, j, k)]
@inline anisu(rp::RegionParams, i, j, k)    = @inbounds rp.anisU[_rid(rp, i, j, k)]
@inline dind(rp::RegionParams, i, j, k)     = @inbounds rp.Dind[_rid(rp, i, j, k)]
@inline dbulk(rp::RegionParams, i, j, k)    = @inbounds rp.Dbulk[_rid(rp, i, j, k)]
@inline polof(rp::RegionParams, i, j, k)    = @inbounds rp.pol[_rid(rp, i, j, k)]
@inline xiof(rp::RegionParams, i, j, k)     = @inbounds rp.xi[_rid(rp, i, j, k)]
@inline lambdaof(rp::RegionParams, i, j, k) = @inbounds rp.lambda[_rid(rp, i, j, k)]
@inline epsprime(rp::RegionParams, i, j, k) = @inbounds rp.epsilonPrime[_rid(rp, i, j, k)]

# has* predicates scan only the regions that are actually present.
_present(rp::RegionParams) = regionlist(rp.regions) .+ 1     # 1-based indices in use
hasku(rp::RegionParams)    = any(!iszero, @view rp.Ku[_present(rp)])
hasdind(rp::RegionParams)  = any(!iszero, @view rp.Dind[_present(rp)])
hasdbulk(rp::RegionParams) = any(!iszero, @view rp.Dbulk[_present(rp)])
hasdmi(rp::RegionParams)   = hasdind(rp) || hasdbulk(rp)
hasstt(rp::RegionParams)   = any(!iszero, @view rp.pol[_present(rp)])

# The demag plan needs a representative Msat to scale the magnetization. With
# region-dependent Msat this is handled per-cell in the demag path (stage 5);
# here we expose the maximum present Msat for setup/reporting.
maxmsat(rp::RegionParams) = maximum(@view rp.Msat[_present(rp)])

# Representative scalar damping (region 0's) for the global LLG torque.
damping(rp::RegionParams) = rp.alpha[1]

"""
    average_region(m, rp, id) -> (mx, my, mz)

Averaged magnetization over the cells assigned to region `id`, every cell with
equal weight — mumax3's `m.Region(id).Average()` (engine/oneregion.go: the sum over
the region's cells divided by their count). With a smoothed edge (fractional fill)
this is NOT mumax3's table `m`; use [`average_magnet`](@ref) for that.
"""
function average_region(m::AbstractArray{T,4}, rp::RegionParams, id::Integer) where {T}
    r = UInt8(id)
    S = promote_type(T, Float64)          # Float64 accumulator also for Float32 m
    sx = sy = sz = zero(S); n = 0
    @inbounds for k in axes(m, 4), j in axes(m, 3), i in axes(m, 2)
        if rp.regions.id[i, j, k] == r
            sx += m[1,i,j,k]; sy += m[2,i,j,k]; sz += m[3,i,j,k]; n += 1
        end
    end
    n == 0 && return (T(NaN), T(NaN), T(NaN))
    return (T(sx / n), T(sy / n), T(sz / n))
end

"""
    average_magnet(m, params) -> (mx, my, mz)

Volume average of the magnetization over the magnet: each cell is weighted by its
fill fraction and empty cells (Msat = 0 or fill 0) are excluded,

    ⟨m⟩ = Σ fill·m / Σ fill.

This is mumax3's table `m` and `m.Average()` (engine/average.go `sAverageMagnet`:
`dot(m, geometry) / sum(geometry)`) under the usual mapping, where mumax3's empty
space comes from `setgeom` (vol 0) and JuliaMag's from an Msat = 0 background. (A
mumax3 script that empties space with Msat = 0 regions and no `setgeom` averages
over the whole box instead.) On a staircase geometry (fill 0 or 1) it
equals [`average_region`](@ref) over the magnet's cells; with edge smoothing the
partially covered boundary cells count with their covered fraction. For a scalar
`Material` (no geometry) it is the plain mesh average.
"""
function average_magnet(m::AbstractArray{T,4}, rp::RegionParams) where {T}
    S = promote_type(T, Float64)          # Float64 accumulator also for Float32 m
    sx = sy = sz = zero(S); w = zero(S)
    @inbounds for k in axes(m, 4), j in axes(m, 3), i in axes(m, 2)
        isempty_cell(rp, i, j, k) && continue
        f = S(fillof(rp, i, j, k))
        sx += f*m[1,i,j,k]; sy += f*m[2,i,j,k]; sz += f*m[3,i,j,k]; w += f
    end
    iszero(w) && return (T(NaN), T(NaN), T(NaN))
    return (T(sx / w), T(sy / w), T(sz / w))
end
average_magnet(m::AbstractArray{T,4}, ::Material) where {T} = average(m)

Base.size(rp::RegionParams) = size(rp.regions)
