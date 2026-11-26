# Spin-transfer torque (STT).
#
# Two current-driven torques, ported from mumax3 (cuda/zhangli2.cu,
# cuda/slonczewski2.cu). Both ADD to an existing torque array (the LLG torque),
# since they act alongside the field torque, not in place of it.
#
# Zhang-Li (current in the plane of the film): a spin-polarized charge current
# flowing through a magnetization texture exerts an adiabatic + non-adiabatic
# torque. Drives domain walls and skyrmions.
#
# Slonczewski (current perpendicular to the plane): in a spin valve, a current
# polarized by a fixed layer transfers angular momentum to the free layer.
# Drives magnetization switching and auto-oscillation.

"""
    zhanglitorque!(τ, m, mesh, mat, J; add=true)

Add the Zhang-Li spin-transfer torque to `τ` [rad/s]. `J` is the charge current
density `(Jx, Jy, Jz)` [A/m²] (uniform over the sample here). Requires
`mat.pol ≠ 0`. Uses `mat.xi` as the non-adiabaticity β.

    τ_zl = -1/(1+α²) [ (1+ξα) m×(m×hs) + (ξ-α) m×hs ],
    hs   = (b·J·∇) m,   b = μB/(2 qe γLL Msat (1+ξ²))
"""
function zhanglitorque!(τ::AbstractArray{T,4}, m::AbstractArray{T,4},
                        mesh::Mesh, mat::Material, J::Tuple; add::Bool = true) where {T}
    mat.pol == 0 && return τ
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh, 1), isperiodic(mesh, 2), isperiodic(mesh, 3)
    α  = T(mat.alpha)
    ξ  = T(mat.xi)

    # mumax3's PREFACTOR = μB/(2 qe γLL) carries a 1/γLL because mumax applies
    # γLL in the time integrator (dm = γLL·torque·dt). JuliaMag's torque!/RK apply
    # γLL inside the LLG torque instead, so the STT term must carry γLL explicitly
    # to sit on the same scale — the γLL here cancels the 1/γLL in PREFACTOR,
    # leaving b = μB/(2 qe Msat (1+ξ²)).
    b = T(μB / (2 * qe) / (mat.Msat * (1 + ξ^2)))     # already ×γLL vs mumax
    Jx = T(mat.pol * J[1]); Jy = T(mat.pol * J[2]); Jz = T(mat.pol * J[3])
    # mumax's exact arithmetic: hs += (b/c)·J·(m[nbr+]-m[nbr-]) (no 1/2).
    bcx = b / T(cx); bcy = b / T(cy); bcz = b / T(cz)

    gfac = T(-1 / (1 + α^2))
    c1 = (1 + ξ*α)
    c2 = (ξ - α)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # (u·∇)m via central differences with clamped (Neumann) / periodic bounds.
        hx = zero(T); hy = zero(T); hz = zero(T)
        if Jx != 0
            il = i > 1 ? i-1 : (px ? Nx : 1); ir = i < Nx ? i+1 : (px ? 1 : Nx)
            w = bcx * Jx
            hx += w*(m[1,ir,j,k]-m[1,il,j,k]); hy += w*(m[2,ir,j,k]-m[2,il,j,k]); hz += w*(m[3,ir,j,k]-m[3,il,j,k])
        end
        if Jy != 0
            jl = j > 1 ? j-1 : (py ? Ny : 1); jr = j < Ny ? j+1 : (py ? 1 : Ny)
            w = bcy * Jy
            hx += w*(m[1,i,jr,k]-m[1,i,jl,k]); hy += w*(m[2,i,jr,k]-m[2,i,jl,k]); hz += w*(m[3,i,jr,k]-m[3,i,jl,k])
        end
        if Jz != 0 && Nz > 1
            kl = k > 1 ? k-1 : (pz ? Nz : 1); kr = k < Nz ? k+1 : (pz ? 1 : Nz)
            w = bcz * Jz
            hx += w*(m[1,i,j,kr]-m[1,i,j,kl]); hy += w*(m[2,i,j,kr]-m[2,i,j,kl]); hz += w*(m[3,i,j,kr]-m[3,i,j,kl])
        end

        mx, my, mz = m[1,i,j,k], m[2,i,j,k], m[3,i,j,k]
        # m × hs
        px1 = my*hz - mz*hy; py1 = mz*hx - mx*hz; pz1 = mx*hy - my*hx
        # m × (m × hs)
        qx = my*pz1 - mz*py1; qy = mz*px1 - mx*pz1; qz = mx*py1 - my*px1

        tx = gfac * (c1*qx + c2*px1)
        ty = gfac * (c1*qy + c2*py1)
        tz = gfac * (c1*qz + c2*pz1)
        if add
            τ[1,i,j,k] += tx; τ[2,i,j,k] += ty; τ[3,i,j,k] += tz
        else
            τ[1,i,j,k] = tx; τ[2,i,j,k] = ty; τ[3,i,j,k] = tz
        end
    end
    return τ
end

# The same with a current density per cell J (3,Nx,Ny,Nz) and the parameters of each cell, as mumax3
# zhangli2.cu: the J of the cell, the central differences of m around it, Msat of the region (a cell
# without Msat has no torque), pol, xi and alpha of the cell.
function zhanglitorque!(τ::AbstractArray{T,4}, m::AbstractArray{T,4}, mesh::Mesh, par::AbstractParams,
                        J::AbstractArray{T,4}; add::Bool = true) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh, 1), isperiodic(mesh, 2), isperiodic(mesh, 3)
    pre = μB / (2 * qe)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Ms = msat_region(par, i, j, k); pol = T(polof(par, i, j, k))
        if Ms == 0 || pol == 0
            add || (τ[1,i,j,k] = τ[2,i,j,k] = τ[3,i,j,k] = zero(T))
            continue
        end
        α = T(alphaof(par, i, j, k)); ξ = T(xiof(par, i, j, k))
        b = T(pre / (Ms * (1 + ξ^2)))
        Jx = pol * J[1,i,j,k]; Jy = pol * J[2,i,j,k]; Jz = pol * J[3,i,j,k]
        hx = zero(T); hy = zero(T); hz = zero(T)
        if Jx != 0
            il = i > 1 ? i-1 : (px ? Nx : 1); ir = i < Nx ? i+1 : (px ? 1 : Nx)
            w = (b / T(cx)) * Jx
            hx += w*(m[1,ir,j,k]-m[1,il,j,k]); hy += w*(m[2,ir,j,k]-m[2,il,j,k]); hz += w*(m[3,ir,j,k]-m[3,il,j,k])
        end
        if Jy != 0
            jl = j > 1 ? j-1 : (py ? Ny : 1); jr = j < Ny ? j+1 : (py ? 1 : Ny)
            w = (b / T(cy)) * Jy
            hx += w*(m[1,i,jr,k]-m[1,i,jl,k]); hy += w*(m[2,i,jr,k]-m[2,i,jl,k]); hz += w*(m[3,i,jr,k]-m[3,i,jl,k])
        end
        if Jz != 0 && Nz > 1
            kl = k > 1 ? k-1 : (pz ? Nz : 1); kr = k < Nz ? k+1 : (pz ? 1 : Nz)
            w = (b / T(cz)) * Jz
            hx += w*(m[1,i,j,kr]-m[1,i,j,kl]); hy += w*(m[2,i,j,kr]-m[2,i,j,kl]); hz += w*(m[3,i,j,kr]-m[3,i,j,kl])
        end
        gfac = T(-1 / (1 + α^2)); c1 = (1 + ξ*α); c2 = (ξ - α)
        mx, my, mz = m[1,i,j,k], m[2,i,j,k], m[3,i,j,k]
        px1 = my*hz - mz*hy; py1 = mz*hx - mx*hz; pz1 = mx*hy - my*hx
        qx = my*pz1 - mz*py1; qy = mz*px1 - mx*pz1; qz = mx*py1 - my*px1
        tx = gfac * (c1*qx + c2*px1); ty = gfac * (c1*qy + c2*py1); tz = gfac * (c1*qz + c2*pz1)
        if add
            τ[1,i,j,k] += tx; τ[2,i,j,k] += ty; τ[3,i,j,k] += tz
        else
            τ[1,i,j,k] = tx; τ[2,i,j,k] = ty; τ[3,i,j,k] = tz
        end
    end
    return τ
end

"""
    slonczewskitorque!(τ, m, mesh, mat, Jz, p, thickness; add=true)

Add the Slonczewski (spin-valve) torque to `τ` [rad/s]. `Jz` is the
perpendicular current density [A/m²], `p` the fixed-layer polarization direction
(normalized here), `thickness` the free-layer thickness [m]. Requires
`mat.pol ≠ 0`; uses `mat.lambda` (Λ) and `mat.epsilonPrime` (ε').

    β  = (ħ/qe) Jz / (thickness Msat)
    ε  = pol Λ² / ((Λ²+1) + (Λ²-1)(p·m))
    τ  = gilb[(A+αB) m×(p×m) + (B-αA) (p×m)],  A=βε, B=βε', gilb=1/(1+α²)
"""
function slonczewskitorque!(τ::AbstractArray{T,4}, m::AbstractArray{T,4},
                            mesh::Mesh, mat::Material, Jz::Real, p, thickness::Real;
                            add::Bool = true) where {T}
    (mat.pol == 0 || Jz == 0) && return τ
    Nx, Ny, Nz = mesh.size
    α = T(mat.alpha)
    Λ = T(mat.lambda)
    εp = T(mat.epsilonPrime)

    pn = normalize3(NTuple{3,T}(p))
    px, py, pz = pn
    # mumax3: β = (ħ/qe) Jz/(t Ms), with γLL applied in the integrator. JuliaMag
    # carries γLL inside the torque, so multiply β by γLL to match the LLG scale.
    β = T(γLL * (ħ / qe) * Jz / (thickness * mat.Msat))
    Λ2 = Λ^2
    gilb = T(1 / (1 + α^2))

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mx, my, mz = m[1,i,j,k], m[2,i,j,k], m[3,i,j,k]
        pm = px*mx + py*my + pz*mz
        ε = T(mat.pol) * Λ2 / ((Λ2 + 1) + (Λ2 - 1) * pm)
        A = β * ε
        B = β * εp
        mxpxmFac = gilb * (A + α*B)
        pxmFac   = gilb * (B - α*A)

        # p × m
        pxm_x = py*mz - pz*my; pxm_y = pz*mx - px*mz; pxm_z = px*my - py*mx
        # m × (p × m)
        mxpxm_x = my*pxm_z - mz*pxm_y
        mxpxm_y = mz*pxm_x - mx*pxm_z
        mxpxm_z = mx*pxm_y - my*pxm_x

        tx = mxpxmFac*mxpxm_x + pxmFac*pxm_x
        ty = mxpxmFac*mxpxm_y + pxmFac*pxm_y
        tz = mxpxmFac*mxpxm_z + pxmFac*pxm_z
        if add
            τ[1,i,j,k] += tx; τ[2,i,j,k] += ty; τ[3,i,j,k] += tz
        else
            τ[1,i,j,k] = tx; τ[2,i,j,k] = ty; τ[3,i,j,k] = tz
        end
    end
    return τ
end

# The same with a current density per cell (its z component) and the parameters of each cell, as
# mumax3 slonczewski2.cu.
function slonczewskitorque!(τ::AbstractArray{T,4}, m::AbstractArray{T,4}, mesh::Mesh, par::AbstractParams,
                            J::AbstractArray{T,4}, p, thickness::Real; add::Bool = true) where {T}
    Nx, Ny, Nz = mesh.size
    px, py, pz = normalize3(NTuple{3,T}(p))
    pre = γLL * (ħ / qe) / thickness
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Jz = J[3,i,j,k]; Ms = msat_region(par, i, j, k); pol = T(polof(par, i, j, k))
        if Jz == 0 || Ms == 0 || pol == 0
            add || (τ[1,i,j,k] = τ[2,i,j,k] = τ[3,i,j,k] = zero(T))
            continue
        end
        α = T(alphaof(par, i, j, k)); Λ2 = T(lambdaof(par, i, j, k))^2; εp = T(epsprime(par, i, j, k))
        β = T(pre / Ms) * Jz
        gilb = T(1 / (1 + α^2))
        mx, my, mz = m[1,i,j,k], m[2,i,j,k], m[3,i,j,k]
        pm = px*mx + py*my + pz*mz
        ε = pol * Λ2 / ((Λ2 + 1) + (Λ2 - 1) * pm)
        A = β * ε; B = β * εp
        f1 = gilb * (A + α*B); f2 = gilb * (B - α*A)
        pxm_x = py*mz - pz*my; pxm_y = pz*mx - px*mz; pxm_z = px*my - py*mx
        tx = f1*(my*pxm_z - mz*pxm_y) + f2*pxm_x
        ty = f1*(mz*pxm_x - mx*pxm_z) + f2*pxm_y
        tz = f1*(mx*pxm_y - my*pxm_x) + f2*pxm_z
        if add
            τ[1,i,j,k] += tx; τ[2,i,j,k] += ty; τ[3,i,j,k] += tz
        else
            τ[1,i,j,k] = tx; τ[2,i,j,k] = ty; τ[3,i,j,k] = tz
        end
    end
    return τ
end
