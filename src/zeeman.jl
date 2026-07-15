# Zeeman (external / applied) field.
#
# This is simply the externally applied field B_ext [T], uniform over the sample
# for now. It does not depend on m, but is written with the same (B, m, ...; add)
# interface as the other terms so the effective-field assembly treats them all
# alike.

"""
    zeeman!(B, Bext; add=false)

Add (or write) a uniform applied field `Bext` (a 3-tuple, in Tesla) into `B`.
"""
function zeeman!(B::AbstractArray{T,4}, Bext; add::Bool = false) where {T}
    bx, by, bz = T(Bext[1]), T(Bext[2]), T(Bext[3])
    Nx, Ny, Nz = size(B, 2), size(B, 3), size(B, 4)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
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
