# Energy minimizer (steepest conjugate descent, Barzilai-Borwein).
#
# Relaxing to equilibrium by integrating the LLG in time is hopelessly slow: the
# stiff exchange field drives the adaptive step to its floor, so reaching a ~ns
# equilibrium would take ~10¹² steps. mumax3 solves this with a dedicated
# minimizer instead of time integration, and so do we. This is a direct port of
# mumax3's Minimize() (engine/minimizer.go, cuda/minimize.cu; steepest descent
# per Exl et al., J. Appl. Phys. 115, 17D118 (2014)).
#
# Key details that must match mumax3 for the method to converge:
#
# 1. Torque scaling. The descent direction is the damping torque
#        τ = -γ m × (m × B)                          (units 1/time)
#    The γ factor makes the Barzilai-Borwein step `dt` a genuine time-like step,
#    as in mumax3, so it starts near 1e-13 s and BB grows it. (An earlier attempt
#    that reduced B by 1/(μ0 Msat) failed: the exchange field at a free boundary
#    is enormous, ~2.6e6 T per unit Δm, so the reduced torque was ~1e4 not O(1),
#    BB drove dt to ~1e-7, and the state froze.)
#
# 2. Operation order. Update m first with the *current* (τ, dt), THEN recompute
#    τ, THEN compute the BB step for the *next* iteration. dt computed this step
#    is used next step, exactly as in mumax3's Step().
#
# 3. Cayley update. m' = [(4 - t²) m + 4 dt τ] / (4 + t²), t² = dt²|τ|². An exact
#    rotation on the unit sphere — a plain `m += dt τ; normalize!` distorts the
#    geometry under the large BB steps that arise here.
#
# 4. Convergence on Δm. Stop when the recent max‖m - m_old‖ is small, not when
#    the torque is small: the BB torque oscillates and may never dip below a
#    torque threshold even at equilibrium.

"""
    Minimizer(world, m; dt=1e-13, stopdm=1e-6, dmsamples=10)

Energy minimizer state for `m` under `world`.

- `dt`: initial Barzilai-Borwein step (time-like, seconds).
- `stopdm`: convergence threshold on the recent max cell displacement ‖Δm‖.
- `dmsamples`: how many recent Δm values must all be below `stopdm` to stop.
"""
mutable struct Minimizer{T<:AbstractFloat,W<:World,A<:AbstractArray{T,4}}
    world::W
    m::A
    dt::T                    # BB step size (time-like, adaptive)
    γ::T                     # gyromagnetic ratio: scales torque to 1/time
    stopdm::T
    dmsamples::Int
    step::Int
    B::A
    τ::A                     # damping torque -γ m×(m×B)
    τlast::A
    mlast::A
    dmhist::Vector{T}        # recent max‖Δm‖ values (ring)
end

function Minimizer(world::World{T}, m::AbstractArray{T,4};
                   dt = 1e-13, stopdm = 1e-6, dmsamples = 10) where {T}
    # γ scales the torque to units of 1/time so the BB step `dt` is a genuine
    # (adaptive) time step, as in mumax3. The initial dt is time-like (~1e-13 s);
    # BB grows it from there.
    Minimizer{T,typeof(world),typeof(m)}(world, m, T(dt), T(γLL), T(stopdm), dmsamples, 0,
                               similar(m), similar(m), similar(m), similar(m),
                               fill(T(Inf), dmsamples))
end

# Damping torque τ = -γ m × (m × B), written into `τ` (units 1/time).
# The GPU (CuArray) methods of these three helpers live in the CUDA extension;
# they are factored out here so the minimizer step itself is array-type-agnostic.
function _damping_torque!(τ, m, B, γ)
    @inbounds for I in CartesianIndices(axes(m)[2:4])
        mx, my, mz = m[1, I], m[2, I], m[3, I]
        bx, by, bz = B[1, I], B[2, I], B[3, I]
        px = my*bz - mz*by               # m × B
        py = mz*bx - mx*bz
        pz = mx*by - my*bx
        τ[1, I] = -γ * (my*pz - mz*py)   # -γ m × (m × B)
        τ[2, I] = -γ * (mz*px - mx*pz)
        τ[3, I] = -γ * (mx*py - my*px)
    end
    return τ
end

# Cayley step m' = [(4-t²)m + 4dt τ]/(4+t²), t²=dt²|τ|², in place; returns max‖Δm‖².
function _cayley_step!(m::AbstractArray{T,4}, τ, dt) where {T}
    dmmax = zero(T)
    @inbounds for I in CartesianIndices(axes(m)[2:4])
        tx, ty, tz = τ[1, I], τ[2, I], τ[3, I]
        t2 = dt^2 * (tx^2 + ty^2 + tz^2)
        inv_d = 1 / (4 + t2)
        f = 4 - t2
        m0x, m0y, m0z = m[1, I], m[2, I], m[3, I]
        nx = (f * m0x + 4dt * tx) * inv_d
        ny = (f * m0y + 4dt * ty) * inv_d
        nz = (f * m0z + 4dt * tz) * inv_d
        m[1, I] = nx; m[2, I] = ny; m[3, I] = nz
        d2 = (nx - m0x)^2 + (ny - m0y)^2 + (nz - m0z)^2
        d2 > dmmax && (dmmax = d2)
    end
    return dmmax
end

# Barzilai-Borwein inner products (ss, sy, yy) with s = m - mlast, y = τlast - τ.
function _bb_sums(m::AbstractArray{T,4}, mlast, τlast, τ) where {T}
    ss = zero(T); sy = zero(T); yy = zero(T)
    @inbounds for I in eachindex(m)
        s = m[I] - mlast[I]
        y = τlast[I] - τ[I]
        ss += s*s; sy += s*y; yy += y*y
    end
    return (ss, sy, yy)
end

"""
    minimizestep!(min) -> max‖Δm‖

One Barzilai-Borwein minimizer step. Returns the max cell displacement this step.
"""
function minimizestep!(mn::Minimizer{T}) where {T}
    m, B, τ = mn.m, mn.B, mn.τ

    # Torque τ was computed at the end of the previous step (or step 0 below).
    # Save the current state, then move with the current (τ, dt).
    copyto!(mn.mlast, m)
    copyto!(mn.τlast, τ)

    if mn.step == 0
        effectivefield!(B, m, mn.world)
        _damping_torque!(τ, m, B, mn.γ)
        copyto!(mn.τlast, τ)
    end

    # Cayley step m' = [(4-t²)m + 4dt τ]/(4+t²), returning max‖Δm‖².
    dmmax = _cayley_step!(m, τ, mn.dt)

    # Recompute the torque at the new state.
    effectivefield!(B, m, mn.world)
    _damping_torque!(τ, m, B, mn.γ)

    # Barzilai-Borwein step size for the NEXT iteration.
    #   dm = m - m_old,  dk = τ_old - τ  (sign reversed, as in mumax3)
    ss, sy, yy = _bb_sums(m, mn.mlast, mn.τlast, τ)
    if sy != 0 && yy != 0
        mn.dt = iseven(mn.step) ? abs(ss / sy) : abs(sy / yy)
    end
    isfinite(mn.dt) && mn.dt > 0 || (mn.dt = T(1e-13))

    mn.dmhist[mod(mn.step, mn.dmsamples)+1] = sqrt(dmmax)
    mn.step += 1
    return sqrt(dmmax)
end

"""
    minimize!(min; maxsteps=100_000, verbose=false)

Run the minimizer until the recent `dmsamples` displacements are all below
`stopdm`, or `maxsteps` is reached.
"""
function minimize!(mn::Minimizer{T}; maxsteps = 100_000, verbose = false) where {T}
    for _ in 1:maxsteps
        dm = minimizestep!(mn)
        if verbose && mn.step % 500 == 0
            mx, my, mz = average(mn.m)
            println("  minimize step ", mn.step, "  max‖Δm‖ = ", dm,
                    "  ⟨m⟩ = (", round(mx, digits=4), ", ", round(my, digits=4),
                    ", ", round(mz, digits=4), ")")
            flush(stdout)
        end
        # Converged once every retained Δm is below the threshold.
        if mn.step >= mn.dmsamples && maximum(mn.dmhist) < mn.stopdm
            break
        end
    end
    return mn
end
