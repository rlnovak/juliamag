# Uniaxial magnetocrystalline anisotropy field.
#
# Energy density  E = Ku (1 - (m·u)²)  is minimized when m is parallel to the
# easy axis u. The associated effective field is
#
#     B_anis = (2 Ku / Msat) (m·u) u          (Tesla, no μ0 — mumax3 convention)
#
# Ku > 0 gives an easy axis along u; Ku < 0 gives an easy plane perpendicular
# to u. Unlike an energy, the field does not carry the cell volume — that only
# appears when integrating the energy over the sample.

"""
    anisotropy!(B, m, mesh, mat; add=false)

Add (or write) the uniaxial anisotropy field of state `m` into `B` [T].

A no-op when `mat.Ku == 0` and `add=true`; writes zeros when `add=false`.
"""
function anisotropy!(B::AbstractArray{T,4}, m::AbstractArray{T,4},
                     mesh::Mesh, params::AbstractParams; add::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size

    if !hasku(params) && !add
        fill!(B, zero(T))
        return B
    end

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Msc = msat(params, i, j, k)
        pref = Msc == 0 ? zero(T) : T(2 * ku(params, i, j, k) / Msc)   # Tesla, no μ0
        ux, uy, uz = anisu(params, i, j, k)
        mu = m[1, i, j, k] * ux + m[2, i, j, k] * uy + m[3, i, j, k] * uz
        a = pref * mu
        bx, by, bz = a * ux, a * uy, a * uz
        if add
            B[1, i, j, k] += bx
            B[2, i, j, k] += by
            B[3, i, j, k] += bz
        else
            B[1, i, j, k] = bx
            B[2, i, j, k] = by
            B[3, i, j, k] = bz
        end
    end
    return B
end
