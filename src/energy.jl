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
    prefactor = selfterm ? T(-0.5) : T(-1)
    V = T(cellvolume(mesh))
    Nx, Ny, Nz = mesh.size
    s = zero(T)
    # Msat is per-cell (a region interface may change it), so it stays inside the
    # sum: E = prefactor · V · Σ Msat[cell] (m·B).
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mb = m[1,i,j,k]*B[1,i,j,k] + m[2,i,j,k]*B[2,i,j,k] + m[3,i,j,k]*B[3,i,j,k]
        s += msat(params, i, j, k) * mb
    end
    return prefactor * V * s
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
    demagfield!(B, m, plan)
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
