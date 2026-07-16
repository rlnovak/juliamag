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

using FFTW

"""
    DemagPlan{T}

Everything needed to evaluate the demag field of a given mesh: the transformed
kernel components, cached FFT plans, and scratch buffers. Build once, reuse
every timestep.
"""
# The forward/inverse FFT plan types are parameters (PF, PI) rather than spelled
# out: FFTW's concrete plan type signatures vary across versions, and pinning
# them here would break on a Julia/FFTW upgrade.
struct DemagPlan{T<:AbstractFloat,PF,PI}
    padsize::NTuple{3,Int}
    dataregion::NTuple{3,UnitRange{Int}}
    prefactor::T                                  # μ0 * Msat
    # Kernel in Fourier space (rfft halves the first axis).
    Kxx::Array{Complex{T},3}
    Kyy::Array{Complex{T},3}
    Kzz::Array{Complex{T},3}
    Kxy::Array{Complex{T},3}
    Kxz::Array{Complex{T},3}
    Kyz::Array{Complex{T},3}
    # Real-space padded scratch, one per component.
    padm::NTuple{3,Array{T,3}}
    # Fourier-space scratch: input mhat (per component) and output bhat.
    mhat::NTuple{3,Array{Complex{T},3}}
    bhat::NTuple{3,Array{Complex{T},3}}
    pfor::PF
    pinv::PI
end

"""
    DemagPlan(kernel::DemagKernel, mesh, Msat) -> DemagPlan

Precompute FFT plans and the transformed demag kernel for `mesh`. `Msat` sets the
μ0·Msat prefactor (a scalar Material, or a representative value for a region map).
"""
DemagPlan(kernel::DemagKernel, mesh::Mesh, mat::Material) = DemagPlan(kernel, mesh, mat.Msat)

function DemagPlan(kernel::DemagKernel{T}, mesh::Mesh, Msat::Real) where {T}
    psize = kernel.padsize
    @assert psize == padsize(mesh)

    scratch = zeros(T, psize...)
    pfor = plan_rfft(scratch)
    Kxx = pfor * kernel.Kxx
    Kyy = pfor * kernel.Kyy
    Kzz = pfor * kernel.Kzz
    Kxy = pfor * kernel.Kxy
    Kxz = pfor * kernel.Kxz
    Kyz = pfor * kernel.Kyz

    hatsize = size(Kxx)
    mhat = ntuple(_ -> zeros(Complex{T}, hatsize), 3)
    bhat = ntuple(_ -> zeros(Complex{T}, hatsize), 3)
    padm = ntuple(_ -> zeros(T, psize...), 3)
    pinv = plan_irfft(mhat[1], psize[1])

    DemagPlan(psize, dataregion(mesh), T(μ0 * Msat),
              Kxx, Kyy, Kzz, Kxy, Kxz, Kyz,
              padm, mhat, bhat, pfor, pinv)
end

"""
    demagfield!(B, m, plan; add=false)

Add (or write) the demagnetization field of state `m` into `B` [T], using a
precomputed [`DemagPlan`](@ref).
"""
function demagfield!(B::AbstractArray{T,4}, m::AbstractArray{T,4},
                     plan::DemagPlan{T}; add::Bool = false) where {T}
    rx, ry, rz = plan.dataregion

    # Zero-pad each component of m into the padded scratch, then forward-transform.
    for c in 1:3
        pc = plan.padm[c]
        fill!(pc, zero(T))
        @inbounds for (kk, k) in enumerate(rz), (jj, j) in enumerate(ry), (ii, i) in enumerate(rx)
            pc[ii, jj, kk] = m[c, i, j, k]
        end
        mul!(plan.mhat[c], plan.pfor, pc)
    end

    mx, my, mz = plan.mhat
    bx, by, bz = plan.bhat

    # Pointwise tensor·vector product in Fourier space: bhat = Khat · mhat.
    @inbounds @simd for I in eachindex(mx)
        Mx, My, Mz = mx[I], my[I], mz[I]
        bx[I] = plan.Kxx[I]*Mx + plan.Kxy[I]*My + plan.Kxz[I]*Mz
        by[I] = plan.Kxy[I]*Mx + plan.Kyy[I]*My + plan.Kyz[I]*Mz
        bz[I] = plan.Kxz[I]*Mx + plan.Kyz[I]*My + plan.Kzz[I]*Mz
    end

    # Inverse-transform each field component and copy out the data region,
    # scaled by μ0·Msat.
    pref = plan.prefactor
    for c in 1:3
        mul!(plan.padm[c], plan.pinv, plan.bhat[c])
        pc = plan.padm[c]
        @inbounds for (kk, k) in enumerate(rz), (jj, j) in enumerate(ry), (ii, i) in enumerate(rx)
            val = pref * pc[ii, jj, kk]
            if add
                B[c, i, j, k] += val
            else
                B[c, i, j, k] = val
            end
        end
    end
    return B
end
