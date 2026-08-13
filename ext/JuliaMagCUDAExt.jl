# CUDA extension for JuliaMag.
#
# Loaded automatically when the user does `using CUDA` alongside JuliaMag (a
# Julia package extension; CUDA is a weak dependency, so the core package has no
# GPU requirement). It provides GPU methods for the hot paths, selected by
# multiple dispatch on CuArray — exactly the design the whole package was built
# around: the field and torque routines take AbstractArray{T,4}, so moving the
# magnetization to the GPU switches to these methods with no change to user code.
#
# STATUS: written without GPU hardware available; intended to be verified on a
# CUDA device (e.g. Google Colab). The scalar-Material path (uniform parameters)
# is covered here; region-wise parameters on the GPU require uploading the region
# tables and are a follow-up.
#
# Strategy: the per-cell field stencils are expressed with array programming
# (shifted views + broadcast) rather than hand-written scalar loops, so the same
# expression runs as a fused GPU kernel on a CuArray and as vectorized CPU code
# on an Array. CUDA.CUFFT backs the demag convolution.

module JuliaMagCUDAExt

using JuliaMag
using CUDA
using CUDA.CUFFT
using LinearAlgebra: mul!
import JuliaMag: exchange!, anisotropy!, zeeman!, torque!, normalize!, average,
                 demagfield!, dmi!, dmi_interfacial!, dmi_bulk!, zhanglitorque!, slonczewskitorque!,
                 vortexcore, skyrmionpos, domainwallpos, topologicalcharge, _interp_max,
                 thermalfield!, cellvolume,
                 Mesh, Material, RegionParams, isperiodic, μ0, γLL, kB, μB, qe, ħ, normalize3,
                 hasku, hasdmi, hasdind, hasdbulk, damping, _demag!, AbstractParams,
                 DemagPlan, World, _damping_torque!, _cayley_step!, _bb_sums

const CuField{T} = CuArray{T,4}

# --- Helpers ---------------------------------------------------------------

# Move a state or field to / from the GPU.
JuliaMag.togpu(m::Array{T,4}) where {T} = CuArray(m)
JuliaMag.tocpu(m::CuArray{T,4}) where {T} = Array(m)

# Circularly / Neumann-shifted copy of component c along an axis. For GPU we
# build neighbour arrays with CUDA-friendly indexing via circshift (periodic) or
# a clamped gather. Here we use circshift for periodic axes and a replicate-edge
# shift for Neumann, both of which are GPU array ops.
function _shift(m::CuArray{T,4}, c::Int, axis::Int, dir::Int, periodic::Bool) where {T}
    comp = @view m[c:c, :, :, :]                      # keep 4D shape
    if periodic
        return circshift(comp, ntuple(d -> d == axis+1 ? dir : 0, 4))
    else
        # Neumann: shift and replicate the edge (so nbr - centre = 0 at the wall).
        s = circshift(comp, ntuple(d -> d == axis+1 ? dir : 0, 4))
        return s   # edge handling below via masking is skipped for brevity
    end
end

# --- Exchange (uniform Material) -------------------------------------------

function exchange!(B::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material;
                   add::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh,1), isperiodic(mesh,2), isperiodic(mesh,3)
    pref = T(2 * mat.Aex / mat.Msat)
    wx = pref / T(cx^2); wy = pref / T(cy^2); wz = pref / T(cz^2)

    # ∇²m via shifted arrays; broadcast fuses into one GPU kernel per component.
    lap = similar(m)
    @views for c in 1:3
        mc = m[c:c, :, :, :]
        l = wx .* (_edgeshift(mc, 1, +1, px) .+ _edgeshift(mc, 1, -1, px) .- 2 .* mc) .+
            wy .* (_edgeshift(mc, 2, +1, py) .+ _edgeshift(mc, 2, -1, py) .- 2 .* mc)
        if Nz > 1
            l = l .+ wz .* (_edgeshift(mc, 3, +1, pz) .+ _edgeshift(mc, 3, -1, pz) .- 2 .* mc)
        end
        lap[c:c, :, :, :] .= l
    end
    if add
        B .+= lap
    else
        B .= lap
    end
    return B
end

# Neighbour array along a spatial axis with Neumann (edge-replicate) or periodic
# boundary — pure array ops, GPU-friendly.
function _edgeshift(comp::AbstractArray{T,4}, axis::Int, dir::Int, periodic::Bool) where {T}
    d = axis + 1                                       # array dim (component is dim 1)
    if periodic
        return circshift(comp, ntuple(k -> k == d ? dir : 0, 4))
    end
    # Neumann: shift, then overwrite the wrapped edge slice with the edge itself
    # so (nbr - centre) = 0 there.
    s = circshift(comp, ntuple(k -> k == d ? dir : 0, 4))
    return _replicate_edge!(s, comp, d, dir)
end

function _replicate_edge!(s, comp, d, dir)
    N = size(comp, d)
    if dir == +1
        # forward shift wraps index N into position 1; that position should equal
        # the centre (edge), giving zero difference. Replace the first slice.
        idx = ntuple(k -> k == d ? (1:1) : Colon(), 4)
        s[idx...] .= @view comp[idx...]
    else
        idx = ntuple(k -> k == d ? (N:N) : Colon(), 4)
        s[idx...] .= @view comp[idx...]
    end
    return s
end

# --- Anisotropy (uniform Material) -----------------------------------------

function anisotropy!(B::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material;
                     add::Bool = false) where {T}
    if mat.Ku == 0
        add || fill!(B, zero(T))
        return B
    end
    pref = T(2 * mat.Ku / mat.Msat)
    ux, uy, uz = mat.anisU
    mx = @view m[1:1, :, :, :]; my = @view m[2:2, :, :, :]; mz = @view m[3:3, :, :, :]
    mu = ux .* mx .+ uy .* my .+ uz .* mz              # (m·u), GPU broadcast
    if add
        B[1:1,:,:,:] .+= pref .* mu .* ux
        B[2:2,:,:,:] .+= pref .* mu .* uy
        B[3:3,:,:,:] .+= pref .* mu .* uz
    else
        B[1:1,:,:,:] .= pref .* mu .* ux
        B[2:2,:,:,:] .= pref .* mu .* uy
        B[3:3,:,:,:] .= pref .* mu .* uz
    end
    return B
end

# --- DMI (uniform Material) ------------------------------------------------
# Central derivative ∂(component c)/∂(axis) on the GPU, reusing the Neumann/
# periodic edge shift of the exchange field. comp is a (1,Nx,Ny,Nz) view.
# circshift(+1) puts m[i-1] at i and circshift(-1) puts m[i+1] at i, so the
# forward neighbour m[i+1] is _edgeshift(...,-1) and m[i-1] is _edgeshift(...,+1);
# the central difference (m[i+1]-m[i-1]) is therefore edgeshift(-1) - edgeshift(+1).
_dcentral(comp, axis, inv2c, periodic) =
    (_edgeshift(comp, axis, -1, periodic) .- _edgeshift(comp, axis, +1, periodic)) .* inv2c

function dmi_interfacial!(B::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material;
                          add::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh,1), isperiodic(mesh,2), isperiodic(mesh,3)
    i2x = T(1/(2cx)); i2y = T(1/(2cy))
    pref = T(2 * mat.Dind / mat.Msat)
    mx = @view m[1:1,:,:,:]; my = @view m[2:2,:,:,:]; mz = @view m[3:3,:,:,:]
    bx = pref .* _dcentral(mz, 1, i2x, px)                              # ∂mz/∂x
    by = pref .* _dcentral(mz, 2, i2y, py)                              # ∂mz/∂y
    bz = pref .* (.-_dcentral(mx, 1, i2x, px) .- _dcentral(my, 2, i2y, py))  # -∂mx/∂x - ∂my/∂y
    if add
        B[1:1,:,:,:] .+= bx; B[2:2,:,:,:] .+= by; B[3:3,:,:,:] .+= bz
    else
        B[1:1,:,:,:] .= bx; B[2:2,:,:,:] .= by; B[3:3,:,:,:] .= bz
    end
    return B
end

function dmi_bulk!(B::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material;
                   add::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh,1), isperiodic(mesh,2), isperiodic(mesh,3)
    i2x = T(1/(2cx)); i2y = T(1/(2cy)); i2z = T(1/(2cz))
    pref = T(2 * mat.Dbulk / mat.Msat)
    mx = @view m[1:1,:,:,:]; my = @view m[2:2,:,:,:]; mz = @view m[3:3,:,:,:]
    z = CUDA.zeros(T, 1, Nx, Ny, Nz)
    dmz_dy = _dcentral(mz, 2, i2y, py); dmy_dz = Nz > 1 ? _dcentral(my, 3, i2z, pz) : z
    dmx_dz = Nz > 1 ? _dcentral(mx, 3, i2z, pz) : z; dmz_dx = _dcentral(mz, 1, i2x, px)
    dmy_dx = _dcentral(my, 1, i2x, px); dmx_dy = _dcentral(mx, 2, i2y, py)
    bx = .-pref .* (dmz_dy .- dmy_dz)                                   # -pref·(∇×m)_x
    by = .-pref .* (dmx_dz .- dmz_dx)
    bz = .-pref .* (dmy_dx .- dmx_dy)
    if add
        B[1:1,:,:,:] .+= bx; B[2:2,:,:,:] .+= by; B[3:3,:,:,:] .+= bz
    else
        B[1:1,:,:,:] .= bx; B[2:2,:,:,:] .= by; B[3:3,:,:,:] .= bz
    end
    return B
end

# --- Spin-transfer torques (uniform Material) ------------------------------

function zhanglitorque!(τ::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material, J;
                        add::Bool = true) where {T}
    mat.pol == 0 && return τ
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh,1), isperiodic(mesh,2), isperiodic(mesh,3)
    α = T(mat.alpha); ξ = T(mat.xi)
    b = T(μB / (2 * qe) / (mat.Msat * (1 + ξ^2)))                       # ×γLL vs mumax (see CPU)
    Jx = T(mat.pol*J[1]); Jy = T(mat.pol*J[2]); Jz = T(mat.pol*J[3])
    mx = @view m[1:1,:,:,:]; my = @view m[2:2,:,:,:]; mz = @view m[3:3,:,:,:]
    # hs = (b/c)·J·(m[nbr+] - m[nbr-]) summed over active current axes (no 1/2).
    Z = CUDA.zeros(T, 1, Nx, Ny, Nz)
    # m[nbr+] - m[nbr-] = m[i+1] - m[i-1] = edgeshift(-1) - edgeshift(+1) (see _dcentral).
    diff(comp, ax, per) = _edgeshift(comp, ax, -1, per) .- _edgeshift(comp, ax, +1, per)
    hx = copy(Z); hy = copy(Z); hz = copy(Z)
    if Jx != 0
        w = b*Jx/T(cx); hx = hx .+ w.*diff(mx,1,px); hy = hy .+ w.*diff(my,1,px); hz = hz .+ w.*diff(mz,1,px)
    end
    if Jy != 0
        w = b*Jy/T(cy); hx = hx .+ w.*diff(mx,2,py); hy = hy .+ w.*diff(my,2,py); hz = hz .+ w.*diff(mz,2,py)
    end
    if Jz != 0 && Nz > 1
        w = b*Jz/T(cz); hx = hx .+ w.*diff(mx,3,pz); hy = hy .+ w.*diff(my,3,pz); hz = hz .+ w.*diff(mz,3,pz)
    end
    gfac = T(-1/(1+α^2)); c1 = T(1+ξ*α); c2 = T(ξ-α)
    p1x = my.*hz .- mz.*hy; p1y = mz.*hx .- mx.*hz; p1z = mx.*hy .- my.*hx   # m × hs
    qx = my.*p1z .- mz.*p1y; qy = mz.*p1x .- mx.*p1z; qz = mx.*p1y .- my.*p1x  # m×(m×hs)
    tx = gfac.*(c1.*qx .+ c2.*p1x); ty = gfac.*(c1.*qy .+ c2.*p1y); tz = gfac.*(c1.*qz .+ c2.*p1z)
    if add
        τ[1:1,:,:,:] .+= tx; τ[2:2,:,:,:] .+= ty; τ[3:3,:,:,:] .+= tz
    else
        τ[1:1,:,:,:] .= tx; τ[2:2,:,:,:] .= ty; τ[3:3,:,:,:] .= tz
    end
    return τ
end

function slonczewskitorque!(τ::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material,
                            Jz::Real, p, thickness::Real; add::Bool = true) where {T}
    (mat.pol == 0 || Jz == 0) && return τ
    α = T(mat.alpha); Λ = T(mat.lambda); εp = T(mat.epsilonPrime)
    pn = normalize3(NTuple{3,T}(p)); ppx, ppy, ppz = pn
    β = T(γLL * (ħ/qe) * Jz / (thickness * mat.Msat))                  # ×γLL (see CPU)
    Λ2 = Λ^2; gilb = T(1/(1+α^2))
    mx = @view m[1:1,:,:,:]; my = @view m[2:2,:,:,:]; mz = @view m[3:3,:,:,:]
    pm = ppx.*mx .+ ppy.*my .+ ppz.*mz
    ε  = T(mat.pol) .* Λ2 ./ ((Λ2 + 1) .+ (Λ2 - 1).*pm)
    A = β .* ε; Bc = β * εp
    f1 = gilb .* (A .+ α.*Bc); f2 = gilb .* (Bc .- α.*A)
    pxm_x = ppy.*mz .- ppz.*my; pxm_y = ppz.*mx .- ppx.*mz; pxm_z = ppx.*my .- ppy.*mx  # p × m
    mxpxm_x = my.*pxm_z .- mz.*pxm_y; mxpxm_y = mz.*pxm_x .- mx.*pxm_z; mxpxm_z = mx.*pxm_y .- my.*pxm_x
    tx = f1.*mxpxm_x .+ f2.*pxm_x; ty = f1.*mxpxm_y .+ f2.*pxm_y; tz = f1.*mxpxm_z .+ f2.*pxm_z
    if add
        τ[1:1,:,:,:] .+= tx; τ[2:2,:,:,:] .+= ty; τ[3:3,:,:,:] .+= tz
    else
        τ[1:1,:,:,:] .= tx; τ[2:2,:,:,:] .= ty; τ[3:3,:,:,:] .= tz
    end
    return τ
end

# --- Zeeman ----------------------------------------------------------------

function zeeman!(B::CuField{T}, Bext; add::Bool = false) where {T}
    bx, by, bz = T(Bext[1]), T(Bext[2]), T(Bext[3])
    if add
        B[1:1,:,:,:] .+= bx; B[2:2,:,:,:] .+= by; B[3:3,:,:,:] .+= bz
    else
        B[1:1,:,:,:] .= bx; B[2:2,:,:,:] .= by; B[3:3,:,:,:] .= bz
    end
    return B
end

# --- LLG torque ------------------------------------------------------------

function torque!(dm::CuField{T}, m::CuField{T}, B::CuField{T}, alpha::Real;
                 gamma::Real = γLL) where {T}
    γ′ = T(gamma / (1 + alpha^2)); α = T(alpha)
    mx = @view m[1:1,:,:,:]; my = @view m[2:2,:,:,:]; mz = @view m[3:3,:,:,:]
    bx = @view B[1:1,:,:,:]; by = @view B[2:2,:,:,:]; bz = @view B[3:3,:,:,:]
    # p = m × B
    px = my.*bz .- mz.*by; py = mz.*bx .- mx.*bz; pz = mx.*by .- my.*bx
    # q = m × p
    qx = my.*pz .- mz.*py; qy = mz.*px .- mx.*pz; qz = mx.*py .- my.*px
    dm[1:1,:,:,:] .= .-γ′ .* (px .+ α.*qx)
    dm[2:2,:,:,:] .= .-γ′ .* (py .+ α.*qy)
    dm[3:3,:,:,:] .= .-γ′ .* (pz .+ α.*qz)
    return dm
end

# --- Normalize and average -------------------------------------------------

function normalize!(m::CuField{T}) where {T}
    mx = @view m[1:1,:,:,:]; my = @view m[2:2,:,:,:]; mz = @view m[3:3,:,:,:]
    n = sqrt.(mx.^2 .+ my.^2 .+ mz.^2)
    inv_n = ifelse.(n .> 0, one(T) ./ n, zero(T))
    m[1:1,:,:,:] .= mx .* inv_n
    m[2:2,:,:,:] .= my .* inv_n
    m[3:3,:,:,:] .= mz .* inv_n
    return m
end

function average(m::CuField{T}) where {T}
    n = size(m,2) * size(m,3) * size(m,4)
    (sum(@view m[1,:,:,:]) / n, sum(@view m[2,:,:,:]) / n, sum(@view m[3,:,:,:]) / n)
end

# --- Demagnetization on the GPU (FFT convolution via CUFFT) -----------------
#
# Mirror of the CPU DemagPlan/_demagfield! (src/demag_field.jl), but with every
# array on the device and the three scalar loops (pad, pointwise tensor·vector,
# copy-out) replaced by array operations so nothing indexes a CuArray on the CPU.
# The transformed kernel is uploaded once; each call does pad → rFFT → pointwise
# → irFFT → crop, all on the GPU.
struct GpuDemagPlan{T<:AbstractFloat,PF,PI}
    padsize::NTuple{3,Int}
    dataregion::NTuple{3,UnitRange{Int}}
    prefactor::T
    Kxx::CuArray{Complex{T},3}; Kyy::CuArray{Complex{T},3}; Kzz::CuArray{Complex{T},3}
    Kxy::CuArray{Complex{T},3}; Kxz::CuArray{Complex{T},3}; Kyz::CuArray{Complex{T},3}
    padm::NTuple{3,CuArray{T,3}}
    mhat::NTuple{3,CuArray{Complex{T},3}}
    bhat::NTuple{3,CuArray{Complex{T},3}}
    pfor::PF
    pinv::PI
end

# Upload a CPU DemagPlan to the GPU: the transformed kernel is already in Fourier
# space, so it is copied straight over; new CUFFT plans and device scratch are
# built for the padded size.
function JuliaMag.togpu(plan::DemagPlan{T}) where {T}
    psize = plan.padsize
    scratch = CUDA.zeros(T, psize...)
    pfor = plan_rfft(scratch)
    mhat = ntuple(_ -> CuArray(zeros(Complex{T}, size(plan.Kxx))), 3)
    bhat = ntuple(_ -> CuArray(zeros(Complex{T}, size(plan.Kxx))), 3)
    padm = ntuple(_ -> CUDA.zeros(T, psize...), 3)
    pinv = plan_irfft(mhat[1], psize[1])
    GpuDemagPlan{T,typeof(pfor),typeof(pinv)}(
        psize, plan.dataregion, plan.prefactor,
        CuArray(plan.Kxx), CuArray(plan.Kyy), CuArray(plan.Kzz),
        CuArray(plan.Kxy), CuArray(plan.Kxz), CuArray(plan.Kyz),
        padm, mhat, bhat, pfor, pinv)
end

# Uniform-Msat demag field on the GPU. `B` and `m` are (3,Nx,Ny,Nz) CuArrays.
function demagfield!(B::CuField{T}, m::CuField{T}, plan::GpuDemagPlan{T};
                     add::Bool = false) where {T}
    rx, ry, rz = plan.dataregion
    pref = plan.prefactor

    # Pad each component into the device scratch and forward-transform.
    for c in 1:3
        pc = plan.padm[c]
        fill!(pc, zero(T))
        @views pc[rx, ry, rz] .= m[c, :, :, :]
        mul!(plan.mhat[c], plan.pfor, pc)
    end

    mx, my, mz = plan.mhat
    bx, by, bz = plan.bhat
    # Pointwise tensor·vector product in Fourier space (broadcast, one kernel each).
    @. bx = plan.Kxx*mx + plan.Kxy*my + plan.Kxz*mz
    @. by = plan.Kxy*mx + plan.Kyy*my + plan.Kyz*mz
    @. bz = plan.Kxz*mx + plan.Kyz*my + plan.Kzz*mz

    # Inverse-transform and crop the data region back into B.
    for c in 1:3
        mul!(plan.padm[c], plan.pinv, plan.bhat[c])
        cropped = @view plan.padm[c][rx, ry, rz]
        if add
            @views B[c, :, :, :] .+= pref .* cropped
        else
            @views B[c, :, :, :] .= pref .* cropped
        end
    end
    return B
end

# --- Energy-minimizer kernels on the GPU -----------------------------------
# GPU versions of the three helpers factored out of the CPU minimizer step. Each
# is pure array programming, so the Barzilai-Borwein minimizer runs unchanged on
# a CuArray state.

# Damping torque τ = -γ m × (m × B).
function _damping_torque!(τ::CuField{T}, m::CuField{T}, B::CuField{T}, γ) where {T}
    mx = @view m[1:1,:,:,:]; my = @view m[2:2,:,:,:]; mz = @view m[3:3,:,:,:]
    bx = @view B[1:1,:,:,:]; by = @view B[2:2,:,:,:]; bz = @view B[3:3,:,:,:]
    px = my.*bz .- mz.*by; py = mz.*bx .- mx.*bz; pz = mx.*by .- my.*bx  # m × B
    g = T(γ)
    τ[1:1,:,:,:] .= .-g .* (my.*pz .- mz.*py)                            # -γ m×(m×B)
    τ[2:2,:,:,:] .= .-g .* (mz.*px .- mx.*pz)
    τ[3:3,:,:,:] .= .-g .* (mx.*py .- my.*px)
    return τ
end

# Cayley step in place; returns max‖Δm‖² (reduction on the device).
function _cayley_step!(m::CuField{T}, τ::CuField{T}, dt) where {T}
    d = T(dt)
    t2 = d^2 .* (τ[1:1,:,:,:].^2 .+ τ[2:2,:,:,:].^2 .+ τ[3:3,:,:,:].^2)  # per cell
    inv_d = one(T) ./ (4 .+ t2); f = 4 .- t2
    mnew = similar(m)
    for c in 1:3
        mnew[c:c,:,:,:] .= (f .* m[c:c,:,:,:] .+ 4 .* d .* τ[c:c,:,:,:]) .* inv_d
    end
    d2 = (mnew[1:1,:,:,:].-m[1:1,:,:,:]).^2 .+ (mnew[2:2,:,:,:].-m[2:2,:,:,:]).^2 .+
         (mnew[3:3,:,:,:].-m[3:3,:,:,:]).^2
    dmmax = maximum(d2)
    copyto!(m, mnew)
    return dmmax
end

# Barzilai-Borwein inner products (ss, sy, yy); s = m - mlast, y = τlast - τ.
function _bb_sums(m::CuField{T}, mlast::CuField{T}, τlast::CuField{T}, τ::CuField{T}) where {T}
    s = m .- mlast
    y = τlast .- τ
    return (sum(s .* s), sum(s .* y), sum(y .* y))
end

# --- Feature trackers on the GPU -------------------------------------------
# All four locate a feature by a reduction over the magnetization; the reductions
# run on the device (sum / findmax) with the per-cell quantities built by
# broadcast, so no scalar CuArray indexing. A handful of scalars (the argmax cell
# and its 3×3 neighbourhood for the vortex sub-cell interpolation; the Nx column
# averages for the wall crossing) come back to the host, which is cheap.

# Topological-charge density q = m·(∂ₓm × ∂ᵧm) over the first z-layer, as a
# (1,Nx,Ny,1) CuArray. Central differences with clamped (Neumann) boundaries,
# matching the CPU _topo_density (reuses the exchange edge shift, which reproduces
# the CPU clamp exactly).
function _topo_density_gpu(m::CuField{T}, mesh::Mesh) where {T}
    Nx, Ny, _ = mesh.size
    cx, cy, _ = mesh.cellsize
    ic2x = T(1/(2cx)); ic2y = T(1/(2cy))
    m1 = @view m[:, :, :, 1:1]                       # first z-layer, keep 4D
    mx = @view m1[1:1,:,:,:]; my = @view m1[2:2,:,:,:]; mz = @view m1[3:3,:,:,:]
    dxx = _dcentral(mx,1,ic2x,false); dxy = _dcentral(my,1,ic2x,false); dxz = _dcentral(mz,1,ic2x,false)
    dyx = _dcentral(mx,2,ic2y,false); dyy = _dcentral(my,2,ic2y,false); dyz = _dcentral(mz,2,ic2y,false)
    cxo = dxy.*dyz .- dxz.*dyy                        # ∂ₓm × ∂ᵧm
    cyo = dxz.*dyx .- dxx.*dyz
    czo = dxx.*dyy .- dxy.*dyx
    return mx.*cxo .+ my.*cyo .+ mz.*czo             # (1,Nx,Ny,1)
end

function topologicalcharge(m::CuField{T}, mesh::Mesh) where {T}
    Nx, Ny, _ = mesh.size
    cx, cy, _ = mesh.cellsize
    return sum(_topo_density_gpu(m, mesh)) * T(cx * cy / (4π))
end

function skyrmionpos(m::CuField{T}, mesh::Mesh) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py = isperiodic(mesh,1), isperiodic(mesh,2)
    q = _topo_density_gpu(m, mesh)                   # (1,Nx,Ny,1)
    w = abs.(q)
    sw = sum(w)
    sw == 0 && return (T(NaN), T(NaN), T(NaN))
    # Broadcast the cell-index grids along x and y (shape (1,Nx,1,1)/(1,1,Ny,1)).
    ix = CuArray(reshape(T.(1:Nx), 1, Nx, 1, 1))
    iy = CuArray(reshape(T.(1:Ny), 1, 1, Ny, 1))
    x = if px
        ax = T(2π/Nx); θ = atan(sum(w .* sin.((ix.-1).*ax)), sum(w .* cos.((ix.-1).*ax)))
        θ < 0 && (θ += T(2π)); (θ/ax + 1 - (Nx+1)/2) * cx
    else
        (sum(w .* ix) / sw - (Nx+1)/2) * cx
    end
    y = if py
        ay = T(2π/Ny); φ = atan(sum(w .* sin.((iy.-1).*ay)), sum(w .* cos.((iy.-1).*ay)))
        φ < 0 && (φ += T(2π)); (φ/ay + 1 - (Ny+1)/2) * cy
    else
        (sum(w .* iy) / sw - (Ny+1)/2) * cy
    end
    z = (1 - (Nz+1)/2) * cz
    return (T(x), T(y), T(z))
end

function vortexcore(m::CuField{T}, mesh::Mesh) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    # Argmax of |mz| over the interior of the first layer, found on the device.
    absmz = abs.(@view m[3, 2:Nx-1, 2:Ny-1, 1])
    isempty(absmz) && return (T(NaN), T(NaN), T(NaN), zero(T))
    _, idx = findmax(absmz)                          # index into the interior block
    maxi = idx[1] + 1; maxj = idx[2] + 1             # back to full-array indices
    # Pull the 3×3 neighbourhood to the host for the sub-cell interpolation.
    nb = Array(@view m[3, maxi-1:maxi+1, maxj-1:maxj+1, 1])
    a(i,j) = abs(nb[i,j])
    dx = _interp_max(a(2,2), a(1,2), a(3,2))
    dy = _interp_max(a(2,2), a(2,1), a(2,3))
    pol = nb[2,2]
    x = (maxi + dx - (Nx+1)/2) * cx
    y = (maxj + dy - (Ny+1)/2) * cy
    z = (1 - (Nz+1)/2) * cz
    return (T(x), T(y), T(z), T(pol))
end

function domainwallpos(m::CuField{T}, mesh::Mesh; comp::Int = 1) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    # Column averages over y of the chosen component (device reduction), to host.
    col = Array(vec(sum(@view m[comp, :, :, 1]; dims = 2))) ./ Ny   # length Nx
    @inbounds for i in 2:Nx
        if (col[i-1] > 0) != (col[i] > 0)
            frac = col[i-1] / (col[i-1] - col[i])
            x = ((i - 1) + frac - (Nx+1)/2) * cx
            return (T(x), T(0), T((1 - (Nz+1)/2) * cz))
        end
    end
    return (T(NaN), T(NaN), T(NaN))
end

# --- Region-wise (multi-material) parameters on the GPU --------------------
# A RegionParams is a small per-region lookup table plus a cell→region map. For
# the GPU we materialize each parameter into a per-cell device array once (gather
# the LUT through the region map on the host, then upload), so the field kernels
# become broadcasts over those arrays with no per-cell indexing on the device.
struct GpuRegionParams{T<:AbstractFloat} <: AbstractParams
    Msat::CuArray{T,4}      # (1,Nx,Ny,Nz)
    Aex::CuArray{T,4}
    Ku::CuArray{T,4}
    ux::CuArray{T,4}; uy::CuArray{T,4}; uz::CuArray{T,4}
    Dind::CuArray{T,4}
    Dbulk::CuArray{T,4}
    alpha::CuArray{T,4}     # per-cell damping (for the thermal field)
    alpha0::T               # region-0 damping, the global LLG torque scale
    hasku::Bool
    hasdind::Bool
    hasdbulk::Bool
end
Base.eltype(::GpuRegionParams{T}) where {T} = T
JuliaMag.hasku(p::GpuRegionParams)   = p.hasku
JuliaMag.hasdind(p::GpuRegionParams) = p.hasdind
JuliaMag.hasdbulk(p::GpuRegionParams)= p.hasdbulk
JuliaMag.hasdmi(p::GpuRegionParams)  = p.hasdind || p.hasdbulk
JuliaMag.damping(p::GpuRegionParams) = p.alpha0

# Gather a per-region LUT into a per-cell (1,Nx,Ny,Nz) array via the region map.
function _percell(lut::Vector{T}, idmap::Array{UInt8,3}) where {T}
    Nx, Ny, Nz = size(idmap)
    a = Array{T}(undef, 1, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        a[1,i,j,k] = lut[idmap[i,j,k] + 1]
    end
    return a
end

function JuliaMag.togpu(rp::RegionParams{T}) where {T}
    idm = rp.regions.id
    pc(v) = CuArray(_percell(v, idm))
    ux = CuArray(_percell([u[1] for u in rp.anisU], idm))
    uy = CuArray(_percell([u[2] for u in rp.anisU], idm))
    uz = CuArray(_percell([u[3] for u in rp.anisU], idm))
    GpuRegionParams{T}(pc(rp.Msat), pc(rp.Aex), pc(rp.Ku), ux, uy, uz,
                       pc(rp.Dind), pc(rp.Dbulk), pc(rp.alpha), T(rp.alpha[1]),
                       hasku(rp), hasdind(rp), hasdbulk(rp))
end

# Demag dispatch for a world whose material was materialized to GpuRegionParams:
# use the region-aware demag (Msat[cell]·m, plan prefactor μ0).
_demag!(B, m, w::World{T,P,M}; add) where {T,P,M<:GpuRegionParams} =
    demagfield!(B, m, w.demagplan, w.material, w.mesh; add = add)

# Harmonic mean of two stiffness arrays (0 if either is 0), matching the CPU rule.
_hmean(a, b) = ifelse.((a .== 0) .| (b .== 0), zero(eltype(a)), 2 .* a .* b ./ (a .+ b))

# Exchange with per-cell Msat/Aex and harmonic-mean interface coupling.
function exchange!(B::CuField{T}, m::CuField{T}, mesh::Mesh, p::GpuRegionParams{T};
                   add::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size; cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh,1), isperiodic(mesh,2), isperiodic(mesh,3)
    ix2 = T(1/cx^2); iy2 = T(1/cy^2); iz2 = T(1/cz^2)
    A = p.Aex
    # Per-neighbour harmonic-mean stiffness (central with each shifted neighbour).
    axl = _hmean(A, _edgeshift(A,1,-1,px)); axr = _hmean(A, _edgeshift(A,1,+1,px))
    ayl = _hmean(A, _edgeshift(A,2,-1,py)); ayr = _hmean(A, _edgeshift(A,2,+1,py))
    nz = Nz > 1
    azl = nz ? _hmean(A, _edgeshift(A,3,-1,pz)) : A
    azr = nz ? _hmean(A, _edgeshift(A,3,+1,pz)) : A
    Msc = p.Msat
    pref = ifelse.(Msc .== 0, zero(T), T(2) ./ Msc)   # empty cells → 0 field
    lap = similar(m)
    @views for c in 1:3
        mc = m[c:c,:,:,:]
        f = pref .* ( axl .* (_edgeshift(mc,1,-1,px) .- mc) .* ix2 .+ axr .* (_edgeshift(mc,1,+1,px) .- mc) .* ix2 .+
                      ayl .* (_edgeshift(mc,2,-1,py) .- mc) .* iy2 .+ ayr .* (_edgeshift(mc,2,+1,py) .- mc) .* iy2 )
        if nz
            f = f .+ pref .* ( azl .* (_edgeshift(mc,3,-1,pz) .- mc) .* iz2 .+ azr .* (_edgeshift(mc,3,+1,pz) .- mc) .* iz2 )
        end
        lap[c:c,:,:,:] .= f
    end
    add ? (B .+= lap) : (B .= lap)
    return B
end

function anisotropy!(B::CuField{T}, m::CuField{T}, mesh::Mesh, p::GpuRegionParams{T};
                     add::Bool = false) where {T}
    if !p.hasku
        add || fill!(B, zero(T)); return B
    end
    Msc = p.Msat
    pref = ifelse.(Msc .== 0, zero(T), T(2) .* p.Ku ./ Msc)
    mx = @view m[1:1,:,:,:]; my = @view m[2:2,:,:,:]; mz = @view m[3:3,:,:,:]
    mu = p.ux.*mx .+ p.uy.*my .+ p.uz.*mz
    if add
        B[1:1,:,:,:] .+= pref.*mu.*p.ux; B[2:2,:,:,:] .+= pref.*mu.*p.uy; B[3:3,:,:,:] .+= pref.*mu.*p.uz
    else
        B[1:1,:,:,:] .= pref.*mu.*p.ux; B[2:2,:,:,:] .= pref.*mu.*p.uy; B[3:3,:,:,:] .= pref.*mu.*p.uz
    end
    return B
end

function dmi_interfacial!(B::CuField{T}, m::CuField{T}, mesh::Mesh, p::GpuRegionParams{T};
                          add::Bool = false) where {T}
    cx, cy, _ = mesh.cellsize
    px, py = isperiodic(mesh,1), isperiodic(mesh,2)
    i2x = T(1/(2cx)); i2y = T(1/(2cy))
    Msc = p.Msat
    pref = ifelse.(Msc .== 0, zero(T), T(2) .* p.Dind ./ Msc)
    mx = @view m[1:1,:,:,:]; my = @view m[2:2,:,:,:]; mz = @view m[3:3,:,:,:]
    bx = pref .* _dcentral(mz,1,i2x,px)
    by = pref .* _dcentral(mz,2,i2y,py)
    bz = pref .* (.-_dcentral(mx,1,i2x,px) .- _dcentral(my,2,i2y,py))
    if add
        B[1:1,:,:,:] .+= bx; B[2:2,:,:,:] .+= by; B[3:3,:,:,:] .+= bz
    else
        B[1:1,:,:,:] .= bx; B[2:2,:,:,:] .= by; B[3:3,:,:,:] .= bz
    end
    return B
end

function dmi_bulk!(B::CuField{T}, m::CuField{T}, mesh::Mesh, p::GpuRegionParams{T};
                   add::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size; cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh,1), isperiodic(mesh,2), isperiodic(mesh,3)
    i2x = T(1/(2cx)); i2y = T(1/(2cy)); i2z = T(1/(2cz))
    Msc = p.Msat
    pref = ifelse.(Msc .== 0, zero(T), T(2) .* p.Dbulk ./ Msc)
    mx = @view m[1:1,:,:,:]; my = @view m[2:2,:,:,:]; mz = @view m[3:3,:,:,:]
    z = CUDA.zeros(T,1,Nx,Ny,Nz)
    dmz_dy=_dcentral(mz,2,i2y,py); dmy_dz=Nz>1 ? _dcentral(my,3,i2z,pz) : z
    dmx_dz=Nz>1 ? _dcentral(mx,3,i2z,pz) : z; dmz_dx=_dcentral(mz,1,i2x,px)
    dmy_dx=_dcentral(my,1,i2x,px); dmx_dy=_dcentral(mx,2,i2y,py)
    bx = .-pref.*(dmz_dy.-dmy_dz); by = .-pref.*(dmx_dz.-dmz_dx); bz = .-pref.*(dmy_dx.-dmx_dy)
    if add
        B[1:1,:,:,:] .+= bx; B[2:2,:,:,:] .+= by; B[3:3,:,:,:] .+= bz
    else
        B[1:1,:,:,:] .= bx; B[2:2,:,:,:] .= by; B[3:3,:,:,:] .= bz
    end
    return B
end

# dmi! dispatcher for the GPU region params (mirrors the CPU one).
function dmi!(B::CuField{T}, m::CuField{T}, mesh::Mesh, p::GpuRegionParams{T};
             add::Bool = false) where {T}
    if p.hasdind
        return dmi_interfacial!(B, m, mesh, p; add = add)
    elseif p.hasdbulk
        return dmi_bulk!(B, m, mesh, p; add = add)
    else
        add || fill!(B, zero(T)); return B
    end
end

# Region-aware demag on the GPU: the convolution acts on Msat[cell]·m and the
# prefactor is μ0 alone (Msat enters per cell, not through the plan's scalar). The
# uploaded plan's prefactor is μ0·Msref, so we rescale by 1/Msref to get μ0.
function demagfield!(B::CuField{T}, m::CuField{T}, plan::GpuDemagPlan{T},
                     p::GpuRegionParams{T}, mesh::Mesh; add::Bool = false) where {T}
    msref = plan.prefactor / T(μ0)                   # plan.prefactor = μ0·Msref
    Mm = (p.Msat .* m) ./ msref                       # (Msat[cell]·m)/Msref
    demagfield!(B, Mm, plan; add = add)              # ·(μ0·Msref) = μ0·Msat[cell]·m
end

# --- Thermal (Langevin) field on the GPU -----------------------------------
# Same fluctuation-dissipation field as the CPU: B = η·sqrt(2 α kB T/(γ Msat V Δt)).
# The unit normals are drawn on the device with CUDA.randn! and scaled in place.
function thermalfield!(Btherm::CuField{T2}, mesh::Mesh, mat::Material,
                       temp::Real, dt::Real; rng = nothing) where {T2}
    CUDA.randn!(Btherm)
    if temp <= 0 || mat.Msat == 0
        fill!(Btherm, zero(T2)); return Btherm
    end
    V = cellvolume(mesh)
    s = sqrt(T2(2 * kB * temp * mat.alpha / (γLL * mat.Msat * V * dt)))
    Btherm .*= s
    return Btherm
end

function thermalfield!(Btherm::CuField{T2}, mesh::Mesh, p::GpuRegionParams,
                       temp::Real, dt::Real; rng = nothing) where {T2}
    CUDA.randn!(Btherm)
    if temp <= 0
        fill!(Btherm, zero(T2)); return Btherm
    end
    V = cellvolume(mesh)
    kfac = T2(2 * kB * temp / (γLL * V * dt))        # per-cell: ·α/Msat, 0 if Msat=0
    α = p.alpha; Ms = p.Msat
    s = ifelse.(Ms .== 0, zero(T2), sqrt.(kfac .* α ./ Ms))
    Btherm .*= s
    return Btherm
end

# Move a whole World to the GPU: upload the demag plan (if any), the material
# (scalar Material stays as-is; RegionParams is materialized to device arrays),
# and the effective-field scratch buffer. effectivefield!(::CuArray, ::CuArray,
# ::World) then dispatches every term to a GPU method.
function JuliaMag.togpu(w::World{T}) where {T}
    gplan = w.demagplan === nothing ? nothing : togpu(w.demagplan)
    gmat  = w.material isa RegionParams ? togpu(w.material) : w.material
    Bbuf  = CuArray(w._Bbuf)
    World{T,typeof(gplan),typeof(gmat),typeof(Bbuf)}(
        w.mesh, gmat, gplan, w.Bext, Bbuf)
end

end # module
