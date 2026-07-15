# Exchange field.
#
# The Heisenberg exchange favours parallel neighbouring moments. In the
# continuum micromagnetic limit its effective field is
#
#     B_exch = (2 A / Msat) ∇²m
#
# with A the exchange stiffness [J/m]. This is a genuine Tesla field in the
# OOMMF/mumax3 convention B_eff = -(1/Msat) δE/δm; no μ0 appears here (it enters
# only the demag and Zeeman terms). The Laplacian is discretized with the
# standard 6-neighbour stencil on the finite-difference grid,
#
#     ∇²m|_c ≈ Σ_axes (m_left - 2 m_c + m_right) / Δ²
#
# At a free (non-periodic) boundary mumax3 uses a Neumann condition: the missing
# neighbour is taken equal to the central cell, so its contribution to the sum
# vanishes and ∂m/∂n = 0. Across a periodic axis the stencil wraps around.

"""
    exchange!(B, m, mesh, mat; add=false)

Add (or write) the exchange field of state `m` into `B` [T].

`B` and `m` are both `(3, Nx, Ny, Nz)`. When `add=false` the exchange field
overwrites `B`; when `add=true` it is accumulated on top of what is already
there, so several field terms can share one buffer.
"""
function exchange!(B::AbstractArray{T,4}, m::AbstractArray{T,4},
                   mesh::Mesh, mat::Material; add::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh, 1), isperiodic(mesh, 2), isperiodic(mesh, 3)

    # Prefactor of the whole field, folding in 1/Δ² per axis. The effective
    # field is B_eff = -(1/Msat) δE/δm in Tesla (the OOMMF/mumax3 convention),
    # so the exchange field is 2A/Msat · ∇²m — NO μ0 here. (μ0 enters only the
    # demag and Zeeman terms, which are genuine B fields.)
    pref = T(2 * mat.Aex / mat.Msat)
    wx = pref / T(cx^2)
    wy = pref / T(cy^2)
    wz = pref / T(cz^2)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        for c in 1:3
            mc = m[c, i, j, k]

            # Neumann boundary: clamp the neighbour to the central cell so it
            # drops out of (m_nbr - m_c). Periodic axis: wrap the index.
            il = i > 1 ? i - 1 : (px ? Nx : 1)
            ir = i < Nx ? i + 1 : (px ? 1 : Nx)
            jl = j > 1 ? j - 1 : (py ? Ny : 1)
            jr = j < Ny ? j + 1 : (py ? 1 : Ny)
            kl = k > 1 ? k - 1 : (pz ? Nz : 1)
            kr = k < Nz ? k + 1 : (pz ? 1 : Nz)

            lap = wx * (m[c, il, j, k] + m[c, ir, j, k] - 2mc) +
                  wy * (m[c, i, jl, k] + m[c, i, jr, k] - 2mc)
            if Nz > 1
                lap += wz * (m[c, i, j, kl] + m[c, i, j, kr] - 2mc)
            end

            if add
                B[c, i, j, k] += lap
            else
                B[c, i, j, k] = lap
            end
        end
    end
    return B
end
