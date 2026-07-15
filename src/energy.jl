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
                     mesh::Mesh, mat::Material; selfterm::Bool = true) where {T}
    s = zero(T)
    @inbounds for I in CartesianIndices(axes(m)[2:4])
        s += m[1, I]*B[1, I] + m[2, I]*B[2, I] + m[3, I]*B[3, I]
    end
    prefactor = selfterm ? T(-0.5) : T(-1)
    return prefactor * mat.Msat * T(cellvolume(mesh)) * s
end

"""
    exchangeenergy(m, mesh, mat) -> E   [J]
"""
function exchangeenergy(m::AbstractArray{T,4}, mesh::Mesh, mat::Material) where {T}
    B = similar(m)
    exchange!(B, m, mesh, mat)
    return fieldenergy(m, B, mesh, mat; selfterm = true)
end

"""
    anisotropyenergy(m, mesh, mat) -> E   [J]

Uniaxial anisotropy energy. Zero when `mat.Ku == 0`.
"""
function anisotropyenergy(m::AbstractArray{T,4}, mesh::Mesh, mat::Material) where {T}
    mat.Ku == 0 && return zero(T)
    B = similar(m)
    anisotropy!(B, m, mesh, mat)
    return fieldenergy(m, B, mesh, mat; selfterm = true)
end

"""
    dmienergy(m, mesh, mat) -> E   [J]
"""
function dmienergy(m::AbstractArray{T,4}, mesh::Mesh, mat::Material) where {T}
    (mat.Dind == 0 && mat.Dbulk == 0) && return zero(T)
    B = similar(m)
    dmi!(B, m, mesh, mat)
    return fieldenergy(m, B, mesh, mat; selfterm = true)
end

"""
    demagenergy(m, plan, mesh, mat) -> E   [J]
"""
function demagenergy(m::AbstractArray{T,4}, plan::DemagPlan, mesh::Mesh, mat::Material) where {T}
    B = similar(m)
    fill!(B, zero(T))
    demagfield!(B, m, plan)
    return fieldenergy(m, B, mesh, mat; selfterm = true)
end

"""
    zeemanenergy(m, Bext, mesh, mat) -> E   [J]

External-field energy; note the prefactor is -1, not -1/2.
"""
function zeemanenergy(m::AbstractArray{T,4}, Bext, mesh::Mesh, mat::Material) where {T}
    B = similar(m)
    zeeman!(B, Bext)
    return fieldenergy(m, B, mesh, mat; selfterm = false)
end

"""
    totalenergy(m, world) -> E   [J]

Sum of every active energy term for state `m` under `world`. This is the
quantity that must decrease monotonically as the system relaxes.
"""
function totalenergy(m::AbstractArray{T,4}, w::World{T}) where {T}
    E = exchangeenergy(m, w.mesh, w.material)
    if w.material.Ku != 0
        E += anisotropyenergy(m, w.mesh, w.material)
    end
    if w.material.Dind != 0 || w.material.Dbulk != 0
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
