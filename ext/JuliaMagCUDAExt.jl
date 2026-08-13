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
                 demagfield!, Mesh, Material, isperiodic, μ0, γLL, DemagPlan, World,
                 _damping_torque!, _cayley_step!, _bb_sums

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

# Move a whole World to the GPU: upload the demag plan (if any) and the
# effective-field scratch buffer. The mesh, material, and applied field are
# unchanged, so effectivefield!(::CuArray, ::CuArray, ::World) then dispatches
# every term (exchange/anisotropy/demag/zeeman) to a GPU method.
function JuliaMag.togpu(w::World{T}) where {T}
    gplan = w.demagplan === nothing ? nothing : togpu(w.demagplan)
    Bbuf  = CuArray(w._Bbuf)
    World{T,typeof(gplan),typeof(w.material),typeof(Bbuf)}(
        w.mesh, w.material, gplan, w.Bext, Bbuf)
end

end # module
