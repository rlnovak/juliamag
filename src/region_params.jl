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
end

function RegionParams(mesh::Mesh, default::Material{T}) where {T}
    n = MAXREGIONS
    fill_(x) = fill(x, n)
    RegionParams{T}(Regions(mesh),
                    fill_(default.Msat), fill_(default.Aex), fill_(default.alpha),
                    fill_(default.Ku), fill_(default.anisU),
                    fill_(default.Dind), fill_(default.Dbulk),
                    fill_(default.pol), fill_(default.xi),
                    fill_(default.lambda), fill_(default.epsilonPrime))
end

Base.eltype(::RegionParams{T}) where {T} = T

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

# --- Accessors: look up the region of the cell, then the per-region table ---
# The +1 converts the 0-based region id to a 1-based vector index.
@inline _rid(rp::RegionParams, i, j, k) = rp.regions.id[i, j, k] + 1
@inline msat(rp::RegionParams, i, j, k)     = @inbounds rp.Msat[_rid(rp, i, j, k)]
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

Base.size(rp::RegionParams) = size(rp.regions)
