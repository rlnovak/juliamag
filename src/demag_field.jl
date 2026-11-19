# Demagnetization field via FFT convolution.
#
# The demag field is a convolution of the magnetization with the demag tensor,
#
#     B_d(r) = μ0 Msat Σ_r' K(r-r') · m(r')
#
# where K is the tensor built in demag_kernel.jl. (Our K already stores the
# field of a unit source, i.e. K = -N with N the usual demag tensor, so the
# sign works out to +K here.) A direct sum is O(Ncells²); the convolution
# theorem turns it into a handful of FFTs, O(N log N).
#
# The magnetization is zero-padded to `padsize(mesh)` so the circular
# convolution the FFT computes equals the linear convolution we want. The kernel
# is transformed once, at plan-construction time, and reused every timestep.
#
# Implementation (the mumax3 layout, engine/demag.go + cuda/kernmulrsymm3d.cu):
# - The kernel is even or odd under each axis reflection (reconstruct_symmetry!),
#   so its Fourier transform is purely real; only the real part is stored, which
#   halves the memory traffic of the Fourier-space product.
# - The transform of the kernel is done in Float64 whatever the plan precision,
#   then rounded to T (mumax3 computes the kernel in float64 too).
# - The inverse transform is the unnormalized backward rFFT; its 1/N is folded
#   into the stored kernel, so no separate scaling pass is needed.
# - The three components are transformed as one batch: the padded input, its
#   transform and the output are (P1,P2,P3,3) arrays, FFT over dims 1:3.
# - The zero padding of the input buffer is written once, at construction; each
#   call overwrites only the data region (the forward real-to-complex transform
#   does not touch its input).

using FFTW

"""
    DemagPlan{T}

Everything needed to evaluate the demag field of a given mesh: the transformed
kernel components, cached FFT plans, and scratch buffers. Build once, reuse
every timestep.
"""
# The array and FFT plan types are parameters rather than spelled out, so the same
# struct holds the CPU (Array/FFTW) plan; the CUDA extension has its own device plan.
struct DemagPlan{T<:AbstractFloat,A3<:AbstractArray{T,3},A4<:AbstractArray{T,4},
                 C4<:AbstractArray{Complex{T},4},PF,PI}
    padsize::NTuple{3,Int}
    dataregion::NTuple{3,UnitRange{Int}}
    prefactor::T                                  # μ0 * Msat
    # Kernel in Fourier space: real (see above), scaled by 1/prod(padsize), size
    # (P1÷2+1, P2, P3) since the rFFT halves the first axis.
    Kxx::A3
    Kyy::A3
    Kzz::A3
    Kxy::A3
    Kxz::A3
    Kyz::A3
    padm::A4     # (P1,P2,P3,3) zero-padded source; the padding is never written
    mhat::C4     # (P1÷2+1,P2,P3,3) its transform, overwritten by the field transform
    bout::A4     # (P1,P2,P3,3) inverse-transform output
    pfor::PF     # batched rFFT over dims 1:3
    pinv::PI     # batched unnormalized backward rFFT over dims 1:3
end

"""
    fourierkernel(kernel::DemagKernel) -> NTuple{6,Array{Float64,3}}

Real Fourier transform of the six kernel components (xx, yy, zz, xy, xz, yz),
computed in Float64 and divided by `prod(padsize)` (the normalization of the
inverse transform). Throws if a transform is not real, which would mean the
kernel lost the reflection symmetry the demag convolution relies on.
"""
function fourierkernel(kernel::DemagKernel)
    N = prod(kernel.padsize)
    comps = (kernel.Kxx, kernel.Kyy, kernel.Kzz, kernel.Kxy, kernel.Kxz, kernel.Kyz)
    names = ("xx", "yy", "zz", "xy", "xz", "yz")
    F = map(k -> rfft(Float64.(k)), comps)
    # Compare the imaginary parts with the scale of the whole kernel (the diagonal),
    # not of each component: an off-diagonal that vanishes by symmetry has no scale.
    scale = maximum(f -> maximum(abs ∘ real, f), F)
    for c in 1:6
        im_ = maximum(abs ∘ imag, F[c])
        im_ <= 1e-10 * scale || error("demag kernel component K$(names[c]) is not real in " *
                                      "Fourier space (max|Im| = $(im_ / scale) of the kernel " *
                                      "scale); the kernel lost its reflection symmetry")
    end
    return ntuple(c -> real.(F[c]) ./ N, 6)
end

"""
    DemagPlan(kernel::DemagKernel, mesh, Msat[, T]) -> DemagPlan{T}

Precompute FFT plans and the transformed demag kernel for `mesh`. `Msat` sets the
μ0·Msat prefactor (a scalar Material, or a representative value for a region map).
`T` is the precision of the plan (default: the kernel's); the kernel transform is
done in Float64 regardless.
"""
DemagPlan(kernel::DemagKernel{S}, mesh::Mesh, mat::Material) where {S} =
    DemagPlan(kernel, mesh, mat.Msat, S)

DemagPlan(kernel::DemagKernel{S}, mesh::Mesh, Msat::Real) where {S} =
    DemagPlan(kernel, mesh, Msat, S)

function DemagPlan(kernel::DemagKernel, mesh::Mesh, Msat::Real, ::Type{T}) where {T<:AbstractFloat}
    psize = kernel.padsize
    @assert psize == padsize(mesh)
    K = map(k -> Array{T,3}(k), fourierkernel(kernel))
    hatsize = (psize[1] ÷ 2 + 1, psize[2], psize[3])
    padm = zeros(T, psize..., 3)
    mhat = zeros(Complex{T}, hatsize..., 3)
    bout = zeros(T, psize..., 3)
    pfor = plan_rfft(padm, 1:3)
    pinv = plan_brfft(mhat, psize[1], 1:3)
    DemagPlan(psize, dataregion(mesh), T(μ0 * Msat), K..., padm, mhat, bout, pfor, pinv)
end

"""
    demagfield!(B, m, plan; add=false)

Add (or write) the demagnetization field of state `m` into `B` [T], using a
precomputed [`DemagPlan`](@ref). The plan's `prefactor = μ0·Msat` assumes a
uniform Msat; for a region-dependent Msat use the four-argument method below.
"""
function demagfield!(B::AbstractArray{T,4}, m::AbstractArray{T,4},
                     plan::DemagPlan{T}; add::Bool = false) where {T}
    _demagfield!(B, m, plan, plan.prefactor, nothing, add)
end

"""
    demagfield!(B, m, plan, params, mesh; add=false)

Region-aware demag field. The convolution acts on the physical magnetization
`M = Msat[cell]·m`, so a per-region Msat (a magnetic multilayer) is handled
correctly; the plan's kernel carries only μ0 in this path. Cells with Msat = 0
contribute nothing (unfilled geometry).
"""
function demagfield!(B::AbstractArray{T,4}, m::AbstractArray{T,4}, plan::DemagPlan{T},
                     params::AbstractParams, mesh::Mesh; add::Bool = false) where {T}
    # prefactor here is μ0 alone; Msat enters cell by cell when padding.
    _demagfield!(B, m, plan, T(μ0), params, add)
end

# Shared core. When `params === nothing` the input is m, scaled by the plan's
# uniform prefactor (μ0·Msat) at the end; otherwise the input is Msat[cell]·m and
# the prefactor is μ0.
function _demagfield!(B::AbstractArray{T,4}, m::AbstractArray{T,4},
                      plan::DemagPlan{T}, pref::T, params, add::Bool) where {T}
    rx, ry, rz = plan.dataregion
    P = plan.padm

    # Copy the (possibly Msat-weighted) magnetization into the data region of the
    # padded buffer; the padding stays zero.
    @inbounds for c in 1:3, k in rz, j in ry, i in rx
        P[i, j, k, c] = params === nothing ? m[c, i, j, k] : msat(params, i, j, k) * m[c, i, j, k]
    end
    mul!(plan.mhat, plan.pfor, P)

    # Pointwise tensor·vector product in Fourier space, in place: Ĥ = K̂ · M̂.
    H = plan.mhat
    @inbounds for I in CartesianIndices(plan.Kxx)
        Mx, My, Mz = H[I, 1], H[I, 2], H[I, 3]
        H[I, 1] = plan.Kxx[I]*Mx + plan.Kxy[I]*My + plan.Kxz[I]*Mz
        H[I, 2] = plan.Kxy[I]*Mx + plan.Kyy[I]*My + plan.Kyz[I]*Mz
        H[I, 3] = plan.Kxz[I]*Mx + plan.Kyz[I]*My + plan.Kzz[I]*Mz
    end

    # Unnormalized inverse transform (1/N is in the kernel), then copy out the data
    # region.
    mul!(plan.bout, plan.pinv, H)
    Bo = plan.bout
    @inbounds for k in rz, j in ry, i in rx, c in 1:3
        val = pref * Bo[i, j, k, c]
        if add
            B[c, i, j, k] += val
        else
            B[c, i, j, k] = val
        end
    end
    return B
end
