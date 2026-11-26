# CUDA extension for JuliaMag.
#
# Loaded automatically when the user does `using CUDA` alongside JuliaMag (a
# Julia package extension; CUDA is a weak dependency, so the core package has no
# GPU requirement). It provides GPU methods for the hot paths, selected by
# multiple dispatch on CuArray — exactly the design the whole package was built
# around: the field and torque routines take AbstractArray{T,4}, so moving the
# magnetization to the GPU switches to these methods with no change to user code.
#
# Strategy (mumax3's): every per-cell term — exchange, anisotropy, DMI, Zeeman, the
# LLG and spin-transfer torques, normalization, the minimizer step — is one
# hand-written CUDA kernel, one thread per cell, no temporaries (cuda_kernels.jl).
# The demag convolution uses batched cuFFT transforms of the three components, the
# real symmetric Fourier kernel with the inverse normalization folded in, and one
# kernel each for the padding, the Fourier-space product and the copy-out. Every
# kernel mirrors the arithmetic of the CPU loop in src/, so CPU and GPU agree to
# rounding in Float64 and Float32 (gpu_validation/test_gpu.jl). The feature
# trackers, which run once per output row, keep array-broadcast implementations.

module JuliaMagCUDAExt

using JuliaMag
using CUDA
using CUDA.CUFFT
using LinearAlgebra: mul!
import JuliaMag: exchange!, anisotropy!, zeeman!, torque!, normalize!, average, maxtorque,
                 demagfield!, dmi!, dmi_interfacial!, dmi_bulk!, zhanglitorque!, slonczewskitorque!,
                 vortexcore, skyrmionpos, domainwallpos, topologicalcharge, _interp_max,
                 thermalfield!, cellvolume,
                 Mesh, Material, RegionParams, isperiodic, μ0, γLL, kB, μB, qe, ħ, normalize3,
                 hasku, hasdmi, hasdind, hasdbulk, damping, _demag!, AbstractParams,
                 DemagPlan, World, _damping_torque!, _cayley_step!, _bb_sums

const CuField{T} = CuArray{T,4}

include("cuda_kernels.jl")

# --- Helpers ---------------------------------------------------------------

# Move a state or field to / from the GPU.
JuliaMag.togpu(m::Array{T,4}) where {T} = CuArray(m)
JuliaMag.tocpu(m::CuArray{T,4}) where {T} = Array(m)

# Mesh sizes and periodicity in the form the kernels take.
_dims(mesh::Mesh) = Int32.(mesh.size)
_pbc(mesh::Mesh) = (isperiodic(mesh, 1), isperiodic(mesh, 2), isperiodic(mesh, 3))
_ncells(m::CuField) = Int32(size(m, 2) * size(m, 3) * size(m, 4))

# --- Exchange (uniform Material) -------------------------------------------
# src/exchange.jl with scalar parameters: no cell is empty, the stiffness and the
# 2/Msat prefactor are uniform.
function exchange!(B::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material;
                   add::Bool = false) where {T}
    Nx, Ny, Nz = _dims(mesh); px, py, pz = _pbc(mesh)
    cx, cy, cz = mesh.cellsize
    Ms = T(mat.Msat)
    _launch(_k_exchange!, Nx * Ny * Nz, B, m, T(mat.Aex), Ms, Ms, Nx, Ny, Nz,
            T(1 / cx^2), T(1 / cy^2), T(1 / cz^2), px, py, pz, add)
    return B
end

# Neighbour array along a spatial axis with Neumann (edge-replicate) or periodic
# boundary — pure array ops, used by the feature trackers (once per output row).
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
    ux, uy, uz = mat.anisU
    _launch(_k_anisotropy!, _ncells(m), B, m, T(mat.Ku), T(mat.Msat), T(mat.Msat), T(ux), T(uy),
            T(uz), _ncells(m), add)
    return B
end

# --- DMI (uniform Material) ------------------------------------------------
# Central derivative ∂(component c)/∂(axis) as an array expression, with the
# Neumann/periodic edge shift above — used only by the topological-charge tracker;
# the DMI field itself is the _k_dmi_* kernel. comp is a (1,Nx,Ny,Nz) view.
# circshift(+1) puts m[i-1] at i and circshift(-1) puts m[i+1] at i, so the
# forward neighbour m[i+1] is _edgeshift(...,-1) and m[i-1] is _edgeshift(...,+1);
# the central difference (m[i+1]-m[i-1]) is therefore edgeshift(-1) - edgeshift(+1).
_dcentral(comp, axis, inv2c, periodic) =
    (_edgeshift(comp, axis, -1, periodic) .- _edgeshift(comp, axis, +1, periodic)) .* inv2c

function dmi_interfacial!(B::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material;
                          add::Bool = false, openbc::Bool = false) where {T}
    Nx, Ny, Nz = _dims(mesh); px, py, _ = _pbc(mesh)
    cx, cy, _ = mesh.cellsize
    _launch(_k_dmi_interfacial!, Nx * Ny * Nz, B, m, T(mat.Dind), T(mat.Aex),
            T(mat.Msat), T(mat.Msat), Nx, Ny, Nz, T(1 / (2cx)), T(1 / (2cy)), px, py, openbc, add)
    return B
end

function dmi_bulk!(B::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material;
                   add::Bool = false, openbc::Bool = false) where {T}
    Nx, Ny, Nz = _dims(mesh); px, py, pz = _pbc(mesh)
    cx, cy, cz = mesh.cellsize
    _launch(_k_dmi_bulk!, Nx * Ny * Nz, B, m, T(mat.Dbulk), T(mat.Aex),
            T(mat.Msat), T(mat.Msat), Nx, Ny, Nz, T(1 / (2cx)), T(1 / (2cy)), T(1 / (2cz)),
            px, py, pz, openbc, add)
    return B
end

# --- Spin-transfer torques (uniform Material) ------------------------------
# Same prefactors as the CPU methods (src/stt.jl), computed on the host in T.

function zhanglitorque!(τ::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material, J;
                        add::Bool = true) where {T}
    mat.pol == 0 && return τ
    Nx, Ny, Nz = _dims(mesh); px, py, pz = _pbc(mesh)
    cx, cy, cz = mesh.cellsize
    α = T(mat.alpha); ξ = T(mat.xi)
    b = T(μB / (2 * qe) / (mat.Msat * (1 + ξ^2)))
    Jx = T(mat.pol * J[1]); Jy = T(mat.pol * J[2]); Jz = T(mat.pol * J[3])
    wx = (b / T(cx)) * Jx; wy = (b / T(cy)) * Jy; wz = (b / T(cz)) * Jz
    gfac = T(-1 / (1 + α^2)); c1 = (1 + ξ*α); c2 = (ξ - α)
    _launch(_k_zhangli!, Nx * Ny * Nz, τ, m, Nx, Ny, Nz, wx, wy, wz,
            Jx != 0, Jy != 0, Jz != 0 && Nz > 1, px, py, pz, gfac, c1, c2, add)
    return τ
end

function slonczewskitorque!(τ::CuField{T}, m::CuField{T}, mesh::Mesh, mat::Material,
                            Jz::Real, p, thickness::Real; add::Bool = true) where {T}
    (mat.pol == 0 || Jz == 0) && return τ
    α = T(mat.alpha); Λ = T(mat.lambda); εp = T(mat.epsilonPrime)
    ppx, ppy, ppz = normalize3(NTuple{3,T}(p))
    β = T(γLL * (ħ / qe) * Jz / (thickness * mat.Msat))
    gilb = T(1 / (1 + α^2))
    _launch(_k_slonczewski!, _ncells(m), τ, m, ppx, ppy, ppz, β, T(mat.pol), Λ^2, εp, α,
            gilb, _ncells(m), add)
    return τ
end

# --- Zeeman ----------------------------------------------------------------

function zeeman!(B::CuField{T}, Bext; add::Bool = false) where {T}
    _launch(_k_zeeman!, _ncells(B), B, T(Bext[1]), T(Bext[2]), T(Bext[3]), _ncells(B), add)
    return B
end

# --- LLG torque ------------------------------------------------------------

function torque!(dm::CuField{T}, m::CuField{T}, B::CuField{T}, alpha::Real;
                 gamma::Real = γLL) where {T}
    _launch(_k_torque!, _ncells(m), dm, m, B, T(gamma / (1 + alpha^2)), T(alpha), _ncells(m))
    return dm
end

# --- Normalize and average -------------------------------------------------

function normalize!(m::CuField{T}) where {T}
    _launch(_k_normalize!, _ncells(m), m, _ncells(m))
    return m
end

# Largest per-cell norm max |dm| (convergence and error-control reductions): one
# kernel into a two-slot device accumulator, no N-sized temporary.
function maxtorque(dm::CuField{T}) where {T}
    acc = JuliaMag._newacc(dm)
    JuliaMag._acc_max!(acc, dm, nothing, one(T))
    v = JuliaMag._acc_read(acc)[1]
    CUDA.unsafe_free!(acc)
    return T(sqrt(v))
end

# Error accumulator of the time steppers (src/timestepper.jl) on the device: the
# IEEE bit patterns of the squared norms in a CuVector{UInt32/UInt64}, updated with
# atomic max (unsigned order = value order for non-negative floats; NaN above all).
_ubits(::Type{Float32}) = UInt32
_ubits(::Type{Float64}) = UInt64
_fbits(::Type{UInt32}) = Float32
_fbits(::Type{UInt64}) = Float64
JuliaMag._newacc(m::CuArray{T}) where {T} = CUDA.zeros(_ubits(T), 2)
JuliaMag._acc_reset!(acc::CuVector{U}) where {U<:Unsigned} = (fill!(view(acc, 1:1), zero(U)); acc)
function JuliaMag._acc_read(acc::CuVector{U}) where {U<:Unsigned}
    h = Array(acc)
    F = _fbits(U)
    return (Float64(reinterpret(F, h[1])), Float64(reinterpret(F, h[2])))
end
function JuliaMag._acc_max!(acc::CuVector{U}, a::CuField{T}, b, scale) where {U<:Unsigned,T}
    U === _ubits(T) || throw(ArgumentError("accumulator $(U) does not match $(T)"))
    N = _ncells(a)
    _launch(_k_accmaxnorm2!, N, acc, a, b === nothing ? a : b, T(scale), N, b !== nothing)
    return acc
end

# Reductions accumulate in Float64, as the CPU methods do (also for Float32 m).
function average(m::CuField{T}) where {T}
    n = size(m,2) * size(m,3) * size(m,4)
    s(c) = sum(Float64, @view m[c,:,:,:])
    return (T(s(1) / n), T(s(2) / n), T(s(3) / n))
end

# --- Demagnetization on the GPU (FFT convolution via CUFFT) -----------------
#
# Device mirror of the CPU DemagPlan (src/demag_field.jl): the real Fourier kernel
# with the 1/N of the inverse folded in, batched rFFT / unnormalized backward rFFT
# over the three components ((P1,P2,P3,3) buffers, transform over dims 1:3), and
# one kernel each for the padding, the Fourier-space product and the copy-out —
# 3 kernels plus 2 batched transforms per field evaluation.
struct GpuDemagPlan{T<:AbstractFloat,PF,PI}
    padsize::NTuple{3,Int}
    dataregion::NTuple{3,UnitRange{Int}}
    prefactor::T
    Kxx::CuArray{T,3}; Kyy::CuArray{T,3}; Kzz::CuArray{T,3}
    Kxy::CuArray{T,3}; Kxz::CuArray{T,3}; Kyz::CuArray{T,3}
    padm::CuArray{T,4}            # padding zeroed once, never written again
    mhat::CuArray{Complex{T},4}
    bout::CuArray{T,4}
    pfor::PF
    pinv::PI
end

function JuliaMag.togpu(plan::DemagPlan{T}) where {T}
    psize = plan.padsize
    padm = CUDA.zeros(T, psize..., 3)
    mhat = CUDA.zeros(Complex{T}, size(plan.mhat)...)
    bout = CUDA.zeros(T, psize..., 3)
    pfor = plan_rfft(padm, (1, 2, 3))
    pinv = plan_brfft(mhat, psize[1], (1, 2, 3))
    GpuDemagPlan{T,typeof(pfor),typeof(pinv)}(
        psize, plan.dataregion, plan.prefactor,
        CuArray(plan.Kxx), CuArray(plan.Kyy), CuArray(plan.Kzz),
        CuArray(plan.Kxy), CuArray(plan.Kxz), CuArray(plan.Kyz),
        padm, mhat, bout, pfor, pinv)
end

# Out-of-place complex-to-real cuFFT overwrites its input, so CUDA.jl's mul! first
# copies the input to a scratch buffer. Here the input (the field transform) is
# scratch anyway, so execute the plan directly and skip that copy.
@inline function _exec_c2r!(y, p, x)
    if isdefined(CUFFT, :unsafe_execute_trailing!)
        CUFFT.unsafe_execute_trailing!(p, x, y)
    else
        mul!(y, p, x)
    end
    return y
end

# Shared core: S = 1 (uniform Msat, plan prefactor μ0·Msat) or the per-cell Msat·fill
# array (prefactor μ0).
function _gpu_demag!(B::CuField{T}, m::CuField{T}, plan::GpuDemagPlan{T}, S, pref::T,
                     add::Bool) where {T}
    Nx, Ny, Nz = Int32.(size(m)[2:4])
    N = Nx * Ny * Nz
    _launch(_k_demag_pad!, N, plan.padm, m, S, Nx, Ny, Nz)
    mul!(plan.mhat, plan.pfor, plan.padm)
    nh = Int32(length(plan.Kxx))
    _launch(_k_demag_kmul!, nh, plan.mhat, plan.Kxx, plan.Kyy, plan.Kzz,
            plan.Kxy, plan.Kxz, plan.Kyz, nh)
    _exec_c2r!(plan.bout, plan.pinv, plan.mhat)
    _launch(_k_demag_crop!, N, B, plan.bout, pref, Nx, Ny, Nz, add)
    return B
end

# Uniform-Msat demag field on the GPU. `B` and `m` are (3,Nx,Ny,Nz) CuArrays.
demagfield!(B::CuField{T}, m::CuField{T}, plan::GpuDemagPlan{T}; add::Bool = false) where {T} =
    _gpu_demag!(B, m, plan, one(T), plan.prefactor, add)

# --- Energy-minimizer kernels on the GPU -----------------------------------
# GPU versions of the three helpers factored out of the CPU minimizer step
# (src/minimizer.jl), so the Barzilai-Borwein minimizer runs unchanged on a CuArray.

# Damping torque τ = -γ m × (m × B).
function _damping_torque!(τ::CuField{T}, m::CuField{T}, B::CuField{T}, γ) where {T}
    _launch(_k_damping_torque!, _ncells(m), τ, m, B, T(γ), _ncells(m))
    return τ
end

# Cayley step in place; returns max‖Δm‖² (reduction on the device).
function _cayley_step!(m::CuField{T}, τ::CuField{T}, dt) where {T}
    N = _ncells(m)
    d2 = CuArray{T}(undef, N)
    _launch(_k_cayley!, N, m, τ, d2, T(dt), N)
    return maximum(d2)
end

# Barzilai-Borwein inner products (ss, sy, yy); s = m - mlast, y = τlast - τ.
# Lazy mapreduce (no temporaries), computed and accumulated in Float64 like the CPU
# method, returned in T.
function _bb_sums(m::CuField{T}, mlast::CuField{T}, τlast::CuField{T}, τ::CuField{T}) where {T}
    d(a, b) = Float64(a) - Float64(b)
    ss = mapreduce((a, b) -> d(a, b) * d(a, b), +, m, mlast)
    sy = mapreduce((a, b, c, e) -> d(a, b) * d(c, e), +, m, mlast, τlast, τ)
    yy = mapreduce((c, e) -> d(c, e) * d(c, e), +, τlast, τ)
    return (T(ss), T(sy), T(yy))
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
    # Argmax of |mz| over the interior of every layer, found on the device (as the
    # CPU method and mumax3 ext_corepos; column-major order = mumax3's x, y, z).
    absmz = abs.(@view m[3, 2:Nx-1, 2:Ny-1, :])
    isempty(absmz) && return (T(NaN), T(NaN), T(NaN), zero(T))
    _, idx = findmax(absmz)                          # index into the interior block
    maxi = idx[1] + 1; maxj = idx[2] + 1; k = idx[3] # back to full-array indices
    # Pull the 3×3 neighbourhood to the host for the sub-cell interpolation.
    nb = Array(@view m[3, maxi-1:maxi+1, maxj-1:maxj+1, k])
    a(i,j) = abs(nb[i,j])
    dx = _interp_max(a(2,2), a(1,2), a(3,2))
    dy = _interp_max(a(2,2), a(2,1), a(2,3))
    pol = nb[2,2]
    x = (maxi + dx - (Nx+1)/2) * cx
    y = (maxj + dy - (Ny+1)/2) * cy
    z = (k - (Nz+1)/2) * cz
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
# the LUT through the region map on the host, then upload); the field kernels read
# them by the cell's linear index (_pc in cuda_kernels.jl).
struct GpuRegionParams{T<:AbstractFloat} <: AbstractParams
    Msat::CuArray{T,4}      # (1,Nx,Ny,Nz) — Msat[region]·fill (effective moment)
    Msat_region::CuArray{T,4} # Msat[region] WITHOUT fill (exchange, DMI, anisotropy prefactors)
    fill::CuArray{T,4}      # per-cell geometry fill fraction in [0,1]
    Aex::CuArray{T,4}
    Ku::CuArray{T,4}
    ux::CuArray{T,4}; uy::CuArray{T,4}; uz::CuArray{T,4}
    Dind::CuArray{T,4}
    Dbulk::CuArray{T,4}
    alpha::CuArray{T,4}     # per-cell damping (for the thermal field)
    alpha0::T               # the damping if it is uniform over the magnet, NaN otherwise
    hasku::Bool
    hasdind::Bool
    hasdbulk::Bool
end
Base.eltype(::GpuRegionParams{T}) where {T} = T
JuliaMag.hasku(p::GpuRegionParams)   = p.hasku
JuliaMag.hasdind(p::GpuRegionParams) = p.hasdind
JuliaMag.hasdbulk(p::GpuRegionParams)= p.hasdbulk
JuliaMag.hasdmi(p::GpuRegionParams)  = p.hasdind || p.hasdbulk
JuliaMag.uniformdamping(p::GpuRegionParams) = isnan(p.alpha0) ? nothing : p.alpha0
JuliaMag.damping(p::GpuRegionParams) = JuliaMag._needuniform(JuliaMag.uniformdamping(p))
function JuliaMag._lltorque_cells!(dm::CuField{T}, m::CuField{T}, B::CuField{T}, p::GpuRegionParams{T}) where {T}
    _launch(_k_torque_cells!, _ncells(m), dm, m, B, p.alpha, T(γLL), _ncells(m))
    return dm
end
# Host copies of the per-cell fill and Msat·fill (isconservative, GNEB weights).
JuliaMag._cellfill(p::GpuRegionParams) = Array(p.fill)
JuliaMag._msatcells(p::GpuRegionParams, ::Mesh) = Array(p.Msat)

# Fill-weighted magnet average (mumax3 table m), as the CPU average_magnet: weight =
# fill, zero where the region Msat is 0 (empty cells).
function JuliaMag.average_magnet(m::CuField{T}, p::GpuRegionParams{T}) where {T}
    # weight = fill, 0 in empty cells; Float64 accumulation as on the CPU
    wt(f, ms) = ms == 0 ? 0.0 : Float64(f)
    W = mapreduce(wt, +, p.fill, p.Msat_region)
    iszero(W) && return (T(NaN), T(NaN), T(NaN))
    s(c) = mapreduce((a, f, ms) -> Float64(a) * wt(f, ms), +,
                     (@view m[c:c,:,:,:]), p.fill, p.Msat_region)
    return (T(s(1) / W), T(s(2) / W), T(s(3) / W))
end

# Total energy from the effective field (src/energy.jl energy!): one Float64
# reduction of Msat·m·(B + Bext) with the CPU's per-cell weights (Msat·fill).
function JuliaMag._energy_from_field(m::CuField{T}, B::CuField{T}, Bext,
                                     p::GpuRegionParams{T}, mesh::Mesh) where {T}
    ex, ey, ez = T(Bext[1]), T(Bext[2]), T(Bext[3])
    f(mx, my, mz, bx, by, bz, ms) =
        Float64(ms) * Float64(mx * (bx + ex) + my * (by + ey) + mz * (bz + ez))
    s = mapreduce(f, +, (@view m[1:1,:,:,:]), (@view m[2:2,:,:,:]), (@view m[3:3,:,:,:]),
                  (@view B[1:1,:,:,:]), (@view B[2:2,:,:,:]), (@view B[3:3,:,:,:]), p.Msat;
                  init = 0.0)
    return -0.5 * cellvolume(mesh) * s
end
function JuliaMag._energy_from_field(m::CuField{T}, B::CuField{T}, Bext,
                                     p::Material, mesh::Mesh) where {T}
    ex, ey, ez = T(Bext[1]), T(Bext[2]), T(Bext[3])
    f(mx, my, mz, bx, by, bz) = Float64(mx * (bx + ex) + my * (by + ey) + mz * (bz + ez))
    s = mapreduce(f, +, (@view m[1:1,:,:,:]), (@view m[2:2,:,:,:]), (@view m[3:3,:,:,:]),
                  (@view B[1:1,:,:,:]), (@view B[2:2,:,:,:]), (@view B[3:3,:,:,:]); init = 0.0)
    return -0.5 * cellvolume(mesh) * Float64(p.Msat) * s
end

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
    Nx, Ny, Nz = size(idm)
    pc(v) = CuArray(_percell(v, idm))
    ux = CuArray(_percell([u[1] for u in rp.anisU], idm))
    uy = CuArray(_percell([u[2] for u in rp.anisU], idm))
    uz = CuArray(_percell([u[3] for u in rp.anisU], idm))
    # Fold the per-cell fill fraction into the materialized Msat for the terms
    # where the moment is a source (demag, thermal, energy),
    # exactly like the CPU accessor msat(rp,i,j,k) = Msat[region]·fill[cell].
    # BUT keep the unscaled region Msat and the fill separately: the exchange, DMI and
    # anisotropy prefactors divide by the FULL region Msat (mumax3 cuda/amul.h inv_Msat over
    # the Msat LUT, never the geometry vol), and the fill mask marks empty
    # neighbours for the Neumann boundary. See the CPU exchange! for the rationale.
    Msat_region = CuArray(_percell(rp.Msat, idm))
    fillcu = CuArray(reshape(Array{T}(rp.fill), 1, Nx, Ny, Nz))
    Msat_eff = Msat_region .* fillcu
    GpuRegionParams{T}(Msat_eff, Msat_region, fillcu, pc(rp.Aex), pc(rp.Ku), ux, uy, uz,
                       pc(rp.Dind), pc(rp.Dbulk), pc(rp.alpha), T(something(JuliaMag.uniformdamping(rp), NaN)),
                       hasku(rp), hasdind(rp), hasdbulk(rp))
end

# Demag dispatch for a world whose material was materialized to GpuRegionParams:
# use the region-aware demag (Msat[cell]·m, plan prefactor μ0).
_demag!(B, m, w::World{T,P,M}; add) where {T,P,M<:GpuRegionParams} =
    demagfield!(B, m, w.demagplan, w.material, w.mesh; add = add)

# Exchange with per-cell Msat/Aex and harmonic-mean interface coupling
# (src/exchange.jl). A cell is empty when its EFFECTIVE Msat (Msat[region]·fill) is
# 0 — not when its fill is 0: background cells outside the geometry keep fill 1
# with region Msat 0. An empty neighbour is a free (Neumann) boundary, mumax3's
# cuda/exchange.cu is0 rule; the prefactor divides by the FULL region Msat.
function exchange!(B::CuField{T}, m::CuField{T}, mesh::Mesh, p::GpuRegionParams{T};
                   add::Bool = false) where {T}
    Nx, Ny, Nz = _dims(mesh); px, py, pz = _pbc(mesh)
    cx, cy, cz = mesh.cellsize
    _launch(_k_exchange!, Nx * Ny * Nz, B, m, p.Aex, p.Msat, p.Msat_region, Nx, Ny, Nz,
            T(1 / cx^2), T(1 / cy^2), T(1 / cz^2), px, py, pz, add)
    return B
end

function anisotropy!(B::CuField{T}, m::CuField{T}, mesh::Mesh, p::GpuRegionParams{T};
                     add::Bool = false) where {T}
    if !p.hasku
        add || fill!(B, zero(T)); return B
    end
    _launch(_k_anisotropy!, _ncells(m), B, m, p.Ku, p.Msat, p.Msat_region, p.ux, p.uy, p.uz,
            _ncells(m), add)
    return B
end

function dmi_interfacial!(B::CuField{T}, m::CuField{T}, mesh::Mesh, p::GpuRegionParams{T};
                          add::Bool = false, openbc::Bool = false) where {T}
    Nx, Ny, Nz = _dims(mesh); px, py, _ = _pbc(mesh)
    cx, cy, _ = mesh.cellsize
    _launch(_k_dmi_interfacial!, Nx * Ny * Nz, B, m, p.Dind, p.Aex, p.Msat,
            p.Msat_region, Nx, Ny, Nz, T(1 / (2cx)), T(1 / (2cy)), px, py, openbc, add)
    return B
end

function dmi_bulk!(B::CuField{T}, m::CuField{T}, mesh::Mesh, p::GpuRegionParams{T};
                   add::Bool = false, openbc::Bool = false) where {T}
    Nx, Ny, Nz = _dims(mesh); px, py, pz = _pbc(mesh)
    cx, cy, cz = mesh.cellsize
    _launch(_k_dmi_bulk!, Nx * Ny * Nz, B, m, p.Dbulk, p.Aex, p.Msat,
            p.Msat_region, Nx, Ny, Nz, T(1 / (2cx)), T(1 / (2cy)), T(1 / (2cz)), px, py, pz,
            openbc, add)
    return B
end

# dmi! dispatcher for the GPU region params (mirrors the CPU one).
function dmi!(B::CuField{T}, m::CuField{T}, mesh::Mesh, p::GpuRegionParams{T};
             add::Bool = false, openbc::Bool = false) where {T}
    if p.hasdind
        return dmi_interfacial!(B, m, mesh, p; add = add, openbc = openbc)
    elseif p.hasdbulk
        return dmi_bulk!(B, m, mesh, p; add = add, openbc = openbc)
    else
        add || fill!(B, zero(T)); return B
    end
end

# Region-aware demag on the GPU: the convolution acts on Msat[cell]·m (Msat·fill,
# padded by the kernel without a temporary) and the prefactor is μ0 alone — the
# CPU region path (src/demag_field.jl).
demagfield!(B::CuField{T}, m::CuField{T}, plan::GpuDemagPlan{T}, p::GpuRegionParams{T},
            mesh::Mesh; add::Bool = false) where {T} =
    _gpu_demag!(B, m, plan, p.Msat, T(μ0), add)

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
        w.mesh, gmat, gplan, w.Bext, Bbuf, w.current, w.openbc, w.drive)   # the SpinCurrent is plain numbers
end

# --- Simulation on the GPU ---------------------------------------------------

JuliaMag.deviceinfo(m::CuArray) = begin
    dev = CUDA.device()
    string("GPU ", CUDA.deviceid(dev), " (", CUDA.name(dev), ", compute capability ",
           CUDA.capability(dev), ", CUDA runtime ", CUDA.runtime_version(), ")")
end

"""
    togpu(sim::Simulation) -> Simulation

The same simulation on the GPU: the World and the magnetization are moved to the
device; the settings, the table and the output schedule are kept, and the CPU World
stays as the host World of the table columns that have no GPU method (energies).
"""
function JuliaMag.togpu(sim::JuliaMag.Simulation)
    sim.m isa CuArray && return sim
    return JuliaMag._rebuild(sim, togpu(sim.world), CuArray(sim.m), sim.world)
end

"""
    tocpu(sim::Simulation) -> Simulation

The simulation back on the CPU (its host World and a host copy of the state).
"""
function JuliaMag.tocpu(sim::JuliaMag.Simulation)
    sim.m isa Array && return sim
    hw = sim.hostworld
    hw.current = sim.world.current
    return JuliaMag._rebuild(sim, hw, Array(sim.m), hw)
end

end # module
