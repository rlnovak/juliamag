# Micromagnetic energies.
#
# Every energy term follows the same mumax3 convention (engine/energy.go): given
# a field term B [T] and the magnetization, the energy is
#
#     E = prefactor · Msat · cellVolume · Σ_cells (m · B)
#
# with prefactor = -1/2 for the self-interacting terms (exchange, demag,
# anisotropy — the ½ avoids double counting a field the magnetization itself
# sources) and prefactor = -1 for the external Zeeman field. Summed over cells
# and multiplied by the cell volume this gives the total energy in Joules.
#
# The primary use is convergence checking: under the energy minimizer or damped
# LLG the total energy must decrease monotonically toward equilibrium.

"""
    fieldenergy(m, B, mesh, mat; selfterm=true) -> E

Energy [J] of state `m` in field `B` [T]:
`E = prefactor · Msat · V_cell · Σ (m·B)`, with `prefactor = -1/2` for a
self-interacting field (`selfterm=true`, the default) or `-1` for an external
field (`selfterm=false`).
"""
function fieldenergy(m::AbstractArray{T,4}, B::AbstractArray{T,4},
                     mesh::Mesh, params::AbstractParams; selfterm::Bool = true) where {T}
    prefactor = selfterm ? -0.5 : -1.0
    V = cellvolume(mesh)
    Nx, Ny, Nz = mesh.size
    # Accumulate in (at least) Float64: a Float32 running sum over ~10⁶ cells
    # would lose several digits. The result is returned in T.
    S = promote_type(T, Float64)
    s = zero(S)
    # Msat is per-cell (a region interface may change it), so it stays inside the
    # sum: E = prefactor · V · Σ Msat[cell] (m·B).
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mb = m[1,i,j,k]*B[1,i,j,k] + m[2,i,j,k]*B[2,i,j,k] + m[3,i,j,k]*B[3,i,j,k]
        s += S(msat(params, i, j, k) * mb)
    end
    return T(prefactor * V * s)
end

"""
    exchangeenergy(m, mesh, params) -> E   [J]
"""
function exchangeenergy(m::AbstractArray{T,4}, mesh::Mesh, params::AbstractParams) where {T}
    B = similar(m)
    exchange!(B, m, mesh, params)
    return fieldenergy(m, B, mesh, params; selfterm = true)
end

"""
    anisotropyenergy(m, mesh, params) -> E   [J]

Uniaxial anisotropy energy. Zero when no region sets `Ku`.
"""
function anisotropyenergy(m::AbstractArray{T,4}, mesh::Mesh, params::AbstractParams) where {T}
    hasku(params) || return zero(T)
    B = similar(m)
    anisotropy!(B, m, mesh, params)
    return fieldenergy(m, B, mesh, params; selfterm = true)
end

"""
    dmienergy(m, mesh, params) -> E   [J]
"""
function dmienergy(m::AbstractArray{T,4}, mesh::Mesh, params::AbstractParams) where {T}
    hasdmi(params) || return zero(T)
    B = similar(m)
    dmi!(B, m, mesh, params)
    return fieldenergy(m, B, mesh, params; selfterm = true)
end

"""
    demagenergy(m, plan, mesh, mat) -> E   [J]
"""
function demagenergy(m::AbstractArray{T,4}, plan::DemagPlan, mesh::Mesh, params::AbstractParams) where {T}
    B = similar(m)
    fill!(B, zero(T))
    # The field must come from the same source the dynamics uses: for per-region
    # params that is M = Msat[region]·fill·m (mumax3 SetMFull), not the uniform-Msat
    # path, which ignores the cell fill and gives a wrong E_demag on edge-smoothed
    # geometry (mumax3 GetDemagEnergy = -½·V·dot(M_full, B_demag)).
    if params isa Material
        demagfield!(B, m, plan)
    else
        demagfield!(B, m, plan, params, mesh)
    end
    return fieldenergy(m, B, mesh, params; selfterm = true)
end

"""
    zeemanenergy(m, Bext, mesh, params) -> E   [J]

External-field energy; note the prefactor is -1, not -1/2.
"""
function zeemanenergy(m::AbstractArray{T,4}, Bext, mesh::Mesh, params::AbstractParams) where {T}
    B = similar(m)
    zeeman!(B, Bext)
    return fieldenergy(m, B, mesh, params; selfterm = false)
end

"""
    energy!(B, m, world) -> E   [J, Float64]

The effective field of `m` into `B` ([`effectivefield!`](@ref)) and the total
energy, from that field in a single reduction. Every self-interaction term
(exchange, anisotropy, DMI, demag) is quadratic in m and enters with -1/2, and
Zeeman is linear and enters with -1, so

    E = -1/2 · V · Σ_cells Msat (m · (B_eff + B_ext))

with the same per-cell weights as the term energies (Msat·fill for a region
material). It equals [`totalenergy`](@ref) to rounding, and costs one field
evaluation instead of one per term. Used by the energy-path (GNEB) and eigenmode
solvers, where E and its gradient −V·Msat·B_eff are needed together. Accumulated and
returned in Float64 (barriers are small differences of large totals).
"""
function energy!(B::AbstractArray{T,4}, m::AbstractArray{T,4}, w::World{T}) where {T}
    effectivefield!(B, m, w)
    return _energy_from_field(m, B, w.Bext, w.material, w.mesh)
end

"""
    isconservative(world) -> Bool

Whether the effective field is exactly the gradient of the energy,
B_eff = −(1/(V·Msat)) ∂E/∂m, cell by cell. That is what energy-path (GNEB) methods
need. It holds for exchange, uniaxial anisotropy, demag (also with fractional cells)
and Zeeman. Three discretizations break it:
- exchange on edge-smoothed cells: the field uses the region's Msat, the energy
  weights it by Msat·fill (as in mumax3);
- DMI next to an open boundary or an empty cell: the clamped (Neumann) neighbour of
  `dmi!` makes the discrete operator non-symmetric there. (mumax3's chiral boundary
  condition does not have this problem; it is not implemented here.) Interfacial
  DMI acts in x and y only, so it is conservative when x and y are periodic and no
  cell is empty;
- a DMI constant that differs between magnetic cells: `dmi!` uses the central cell's
  D for both directions of a bond.
"""
function isconservative(w::World)
    p = w.material
    f = _cellfill(p)
    f !== nothing && any(x -> 0 < x < 1, f) && return false
    if hasdmi(p)
        any(iszero, _msatcells(p, w.mesh)) && return false
        _dmiuniform(p) || return false
        dirs = hasdbulk(p) ? (1, 2, 3) : (1, 2)
        any(d -> !isperiodic(w.mesh, d) && w.mesh.size[d] > 1, dirs) && return false
    end
    return true
end

# Whether the DMI constants are the same in every region present.
_dmiuniform(::AbstractParams) = true
_dmiuniform(p::RegionParams) = allequal(@view p.Dind[_present(p)]) &&
                               allequal(@view p.Dbulk[_present(p)])

# Per-cell geometry fill (nothing: every cell full) and per-cell moment Msat·fill as a
# host (1,Nx,Ny,Nz) array. The GPU region parameters have their own methods.
_cellfill(::AbstractParams) = nothing
_cellfill(p::RegionParams) = p.fill
function _msatcells(p::AbstractParams, mesh::Mesh)
    T = eltype(p)
    w = zeros(T, 1, mesh.size...)
    for k in 1:mesh.size[3], j in 1:mesh.size[2], i in 1:mesh.size[1]
        w[1, i, j, k] = msat(p, i, j, k)
    end
    return w
end

function _energy_from_field(m::AbstractArray{T,4}, B::AbstractArray{T,4}, Bext,
                            params::AbstractParams, mesh::Mesh) where {T}
    ex, ey, ez = T(Bext[1]), T(Bext[2]), T(Bext[3])
    Nx, Ny, Nz = mesh.size
    s = 0.0
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mb = m[1,i,j,k] * (B[1,i,j,k] + ex) + m[2,i,j,k] * (B[2,i,j,k] + ey) +
             m[3,i,j,k] * (B[3,i,j,k] + ez)
        s += Float64(msat(params, i, j, k)) * Float64(mb)
    end
    return -0.5 * cellvolume(mesh) * s
end

"""
    totalenergy(m, world) -> E   [J]

Sum of every active energy term for state `m` under `world`. This is the
quantity that must decrease monotonically as the system relaxes.
"""
function totalenergy(m::AbstractArray{T,4}, w::World{T}) where {T}
    E = exchangeenergy(m, w.mesh, w.material)
    if hasku(w.material)
        E += anisotropyenergy(m, w.mesh, w.material)
    end
    if hasdmi(w.material)
        E += dmienergy(m, w.mesh, w.material)
    end
    if w.demagplan !== nothing
        E += demagenergy(m, w.demagplan, w.mesh, w.material)
    end
    if any(!iszero, w.Bext)
        E += zeemanenergy(m, w.Bext, w.mesh, w.material)
    end
    return E
end
