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
# Harmonic mean of two exchange stiffnesses (mumax3's rule at a region
# interface). Reduces to a when a == b, and to 0 if either is 0 (an empty
# neighbour severs the exchange coupling).
@inline function harmonicmean(a::T, b::T) where {T}
    (a == 0 || b == 0) && return zero(T)
    return 2a * b / (a + b)
end

function exchange!(B::AbstractArray{T,4}, m::AbstractArray{T,4},
                   mesh::Mesh, params::AbstractParams; add::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh, 1), isperiodic(mesh, 2), isperiodic(mesh, 3)
    ix2 = T(1 / cx^2); iy2 = T(1 / cy^2); iz2 = T(1 / cz^2)

    # Per neighbour, mumax3 accumulates  B += (2 a / Msat) w (m_nbr - m_c)/Δ²,
    # where a is the harmonic mean of the central and neighbour stiffnesses (so a
    # region interface or an empty cell is handled correctly) and Msat is the
    # central cell's. The effective field carries no μ0 (OOMMF/mumax3 convention).
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # An empty cell (region Msat 0 or fill 0) carries no field. But the
        # prefactor divides by the region's FULL Msat, not Msat·fill: mumax3's
        # exchange uses inv_Msat over the Msat LUT (cuda/amul.h) and never the
        # geometry fill, which only scales the demag source. Dividing by Msat·fill
        # here would inflate B_exch by 1/fill at partially-filled boundary cells.
        if msat(params, i, j, k) == 0            # empty cell: no field
            for c in 1:3
                add || (B[c, i, j, k] = 0)
            end
            continue
        end
        Msc = msat_region(params, i, j, k)       # full region Msat (no fill)
        Ac = aex(params, i, j, k)
        pref = T(2 / Msc)

        il = i > 1 ? i - 1 : (px ? Nx : 1)
        ir = i < Nx ? i + 1 : (px ? 1 : Nx)
        jl = j > 1 ? j - 1 : (py ? Ny : 1)
        jr = j < Ny ? j + 1 : (py ? 1 : Ny)
        kl = k > 1 ? k - 1 : (pz ? Nz : 1)
        kr = k < Nz ? k + 1 : (pz ? 1 : Nz)

        # Empty-neighbour handling matches mumax3 (cuda/exchange.cu): a missing
        # neighbour (empty cell, or a non-periodic edge that clamps onto the cell
        # itself) contributes nothing — mumax replaces its m with m0 so (m_-m0)=0,
        # a free (Neumann) boundary. We take the same route by substituting the
        # central m for an empty/clamped neighbour, which also makes the empty
        # cell's stiffness irrelevant (so a nonzero background-region Aex can't
        # leak a spurious -Ac·m_c/Δ² term at the geometry edge).
        el = isempty_cell(params, il, j, k) || (il == i); er = isempty_cell(params, ir, j, k) || (ir == i)
        fl = isempty_cell(params, i, jl, k) || (jl == j); fr = isempty_cell(params, i, jr, k) || (jr == j)

        axl = el ? Ac : harmonicmean(Ac, aex(params, il, j, k))
        axr = er ? Ac : harmonicmean(Ac, aex(params, ir, j, k))
        ayl = fl ? Ac : harmonicmean(Ac, aex(params, i, jl, k))
        ayr = fr ? Ac : harmonicmean(Ac, aex(params, i, jr, k))

        for c in 1:3
            mc = m[c, i, j, k]
            mxl = el ? mc : m[c, il, j, k]; mxr = er ? mc : m[c, ir, j, k]
            myl = fl ? mc : m[c, i, jl, k]; myr = fr ? mc : m[c, i, jr, k]
            f = pref * ( axl * (mxl - mc) * ix2 + axr * (mxr - mc) * ix2 +
                         ayl * (myl - mc) * iy2 + ayr * (myr - mc) * iy2 )
            if Nz > 1
                gl = isempty_cell(params, i, j, kl) || (kl == k); gr = isempty_cell(params, i, j, kr) || (kr == k)
                azl = gl ? Ac : harmonicmean(Ac, aex(params, i, j, kl))
                azr = gr ? Ac : harmonicmean(Ac, aex(params, i, j, kr))
                mzl = gl ? mc : m[c, i, j, kl]; mzr = gr ? mc : m[c, i, j, kr]
                f += pref * ( azl * (mzl - mc) * iz2 + azr * (mzr - mc) * iz2 )
            end
            if add
                B[c, i, j, k] += f
            else
                B[c, i, j, k] = f
            end
        end
    end
    return B
end
