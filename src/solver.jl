# Adaptive Dormand-Prince RK45 time integrator.
#
# This is the RK45DP method mumax3 uses by default: a 5th-order step with an
# embedded 4th-order estimate. The difference between the two orders estimates
# the local error, which drives adaptive step-size control. After each accepted
# step |m| is renormalized, since the LLG torque conserves |m| only to the
# order of the integrator.
#
# The Butcher tableau (Dormand & Prince 1980):
#   c  = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
#   b5 (5th order) and b4 (4th order) give the solution and error estimate.

# Dormand-Prince coefficients.
const _DP_A = (
    (1/5,),
    (3/40, 9/40),
    (44/45, -56/15, 32/9),
    (19372/6561, -25360/2187, 64448/6561, -212/729),
    (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
    (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84),  # == b5 (FSAL)
)
# 5th-order weights (same as the last A row → First-Same-As-Last).
const _DP_B5 = (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0)
# 4th-order weights, for the embedded error estimate.
const _DP_B4 = (5179/57600, 0.0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40)
const _DP_C = (0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0)

"""
    Solver(world, m; dt=1e-15, maxerr=1e-5, mindt=1e-18, maxdt=1e-11)

Adaptive RK45 integrator state for magnetization `m` under `world`.

- `dt`: initial step [s]; adapted automatically.
- `maxerr`: per-step tolerance on the max cell error of |Δm|.
- `mindt`, `maxdt`: step-size bounds [s].
"""
mutable struct Solver{T<:AbstractFloat,W<:World}
    world::W
    m::Array{T,4}
    dt::T
    maxerr::T
    mindt::T
    maxdt::T
    t::T                          # elapsed time [s]
    step::Int                     # accepted steps
    nfail::Int                    # rejected steps
    # Scratch: 7 stage derivatives + temporaries.
    k::NTuple{7,Array{T,4}}
    mtmp::Array{T,4}
    B::Array{T,4}
    merr::Array{T,4}
end

function Solver(world::World{T}, m::Array{T,4};
               dt = 1e-15, maxerr = 1e-5, mindt = 1e-18, maxdt = 1e-11) where {T}
    k = ntuple(_ -> similar(m), 7)
    Solver{T,typeof(world)}(world, m, T(dt), T(maxerr), T(mindt), T(maxdt),
                            zero(T), 0, 0, k, similar(m), similar(m), similar(m))
end

# One derivative evaluation: dm = torque(effectivefield(m)).
function _rhs!(dm, m, s::Solver)
    effectivefield!(s.B, m, s.world)
    torque!(dm, m, s.B, damping(s.world.material))
    return dm
end

"""
    step!(solver) -> dt_taken

Take one adaptive RK45 step. On success advances `solver.m` and `solver.t`,
grows or shrinks `dt` for the next step, and returns the step actually taken;
on a rejected step, shrinks `dt` and retries.
"""
function step!(s::Solver{T}) where {T}
    m, k, mtmp = s.m, s.k, s.mtmp
    while true
        h = s.dt
        _rhs!(k[1], m, s)                             # stage 1

        for stage in 2:7
            @. mtmp = m
            a = _DP_A[stage-1]
            for l in 1:stage-1
                axpy_stage!(mtmp, T(h * a[l]), k[l])
            end
            _rhs!(k[stage], mtmp, s)
        end

        # 5th-order solution and embedded 4th-order error, into mtmp / merr.
        @. mtmp = m
        fill!(s.merr, zero(T))
        for l in 1:7
            b5 = T(h * _DP_B5[l]); b4 = T(h * _DP_B4[l])
            @inbounds @simd for I in eachindex(mtmp)
                mtmp[I] += b5 * k[l][I]
                s.merr[I] += (b5 - b4) * k[l][I]
            end
        end

        err = maxnorm3(s.merr)
        # A diverging trial can zero the error estimate (0·Inf) while the
        # solution itself is NaN, so gate acceptance on the solution being
        # finite too — otherwise `err == 0` would slip a NaN state through.
        ok = isfinite(err) && finite4(mtmp)
        if ok && (err <= s.maxerr || h <= s.mindt)
            # Accept: commit the step, renormalize, adapt dt.
            copyto!(m, mtmp)
            normalize!(m)
            s.t += h
            s.step += 1
            s.dt = clamp(adapt_dt(h, err, s.maxerr), s.mindt, s.maxdt)
            return h
        else
            # Reject: shrink and retry the same step.
            s.nfail += 1
            if !ok && h <= s.mindt
                error("RK45 diverged at the minimum step size ($(s.mindt) s): the " *
                      "state produced a non-finite torque. The initial state is " *
                      "likely too far from equilibrium for an explicit solver — " *
                      "relax from a smoother state or lower the cell size / field.")
            end
            s.dt = ok ? max(adapt_dt(h, err, s.maxerr), s.mindt) : max(h / 10, s.mindt)
        end
    end
end

# PI-free step adaptation: h_new = h · (tol/err)^(1/5), damped and bounded.
function adapt_dt(h::T, err::T, tol::T) where {T}
    err == 0 && return 2h
    fac = T(0.9) * (tol / err)^T(0.2)
    return h * clamp(fac, T(0.2), T(2.0))
end

# mtmp .+= a .* v  over a 4D array.
@inline function axpy_stage!(mtmp::AbstractArray{T,4}, a::T, v::AbstractArray{T,4}) where {T}
    @inbounds @simd for I in eachindex(mtmp)
        mtmp[I] += a * v[I]
    end
    return mtmp
end

# True if every entry is finite (no Inf/NaN).
function finite4(a::AbstractArray{T,4}) where {T}
    @inbounds for I in eachindex(a)
        isfinite(a[I]) || return false
    end
    return true
end

# Largest per-cell vector norm of a (3,Nx,Ny,Nz) array.
function maxnorm3(a::AbstractArray{T,4}) where {T}
    mx = zero(T)
    @inbounds for k in axes(a, 4), j in axes(a, 3), i in axes(a, 2)
        n = a[1,i,j,k]^2 + a[2,i,j,k]^2 + a[3,i,j,k]^2
        n > mx && (mx = n)
    end
    return sqrt(mx)
end

"""
    runtime!(solver, duration)

Integrate for `duration` seconds of simulated time, adapting the step so the
run ends exactly at `solver.t + duration`.
"""
function runtime!(s::Solver{T}, duration::Real) where {T}
    t_end = s.t + T(duration)
    while s.t < t_end
        s.dt = min(s.dt, t_end - s.t)
        step!(s)
    end
    return s
end

"""
    relax!(solver; stopmaxtorque=1e-4·γLL·|Bref|, maxsteps=100_000)

Integrate until the dynamics settle: stop when the maximum torque over all cells
drops below `stoptorque` [rad/s]. Damping is temporarily raised for speed.
"""
function relax!(s::Solver{T}; stoptorque = T(1e-2), maxsteps = 100_000) where {T}
    dm = s.k[1]
    for _ in 1:maxsteps
        step!(s)
        _rhs!(dm, s.m, s)
        maxtorque(dm) < stoptorque && break
    end
    return s
end
