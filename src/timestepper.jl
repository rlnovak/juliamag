# mumax3's time integrators, step for step.
#
# A TimeStepper advances the magnetization with one of mumax3's explicit solvers
# (engine/euler.go, heun.go, rk23.go, rk4.go, rk45dp.go), reproducing their stages,
# normalizations, error estimates and step-size control operation for operation:
#
#   :euler  Euler                          (mumax3 solver 1)
#   :heun   Heun, embedded Euler error     (2)
#   :rk23   Bogacki–Shampine, FSAL         (3)
#   :rk4    classic Runge–Kutta 4          (4)
#   :rk45   Dormand–Prince, FSAL — default (5)
#
# The only change of variables: mumax3's torque is in tesla and its internal step is
# h = Dt·γLL; JuliaMag's torque (llgrhs!) is in rad/s, already carrying γLL, so the
# same products h·k appear here as Dt·k. Error control (mumax3 engine/run.go):
# a step is accepted when err < MaxErr (or Dt ≤ MinDt, or the step is fixed), where
# err is the largest per-cell norm of the embedded error estimate of Δm; the next
# step is Dt·clamp(Headroom·(MaxErr/err)^(1/p), 1/2, 2), with p = order+1 after an
# accepted step and order+2 after a rejected one, then bounded by MinDt/MaxDt and
# clipped so as not to cross the end of the run (mumax3's alarm).
#
# Every operation is a broadcast or a per-term kernel (llgrhs!, normalize!, the
# error reductions below), so the stepper runs unchanged on a CuArray.
#
# Error reductions: the largest per-cell |Δm| estimate of a step is reduced into a
# two-slot accumulator (this step, peak), on the device for a GPU array. An adaptive
# step reads it back once (it needs err to accept the step and choose the next one,
# as mumax3 does); a fixed step never waits for it: the error stays on the device
# and is read only when lasterr/peakerr are asked for, and at the end of advanceto!,
# where a non-finite error stops the run. LastTorque is computed on demand from the
# stage that holds it (lasttorque(ts)) instead of after every step.
#
# Finite temperature: a fresh thermal field (thermalfield!) is drawn for every step
# and held through all of its stages; FSAL is disabled, as in mumax3. The noise
# variance scales as 1/Dt, so a thermal run needs a fixed step (fixdt > 0; mumax3
# allows an adaptive one, rescaling the noise after a rejected step, but the
# statistics of the resulting process are not those of the Langevin equation).

const SOLVERS = (:euler, :heun, :rk23, :rk4, :rk45)

# Order of the error estimate of each solver (mumax3's adaptDt exponents use
# 1/(order+1) after an accepted step, 1/(order+2) after a rejected one).
_order(s::Symbol) = s === :heun ? 1 : s === :rk23 ? 2 : s === :rk4 ? 3 : s === :rk45 ? 4 : 0
_nstage(s::Symbol) = s === :euler ? 1 : s === :heun ? 2 : s === :rk23 ? 4 : s === :rk4 ? 4 : 7

"""
    TimeStepper(world, m; solver=:rk45, maxerr=1e-5, fixdt=0, mindt=0, maxdt=0,
                headroom=0.8, dt=1e-15, temperature=0, rng=Random.default_rng())

State of one of mumax3's explicit time integrators (`solver` ∈ `:euler, :heun,
:rk23, :rk4, :rk45`) for magnetization `m` (integrated in place) under `world`.
Defaults are mumax3's: `MaxErr = 1e-5`, `Headroom = 0.8`, initial `Dt = 1e-15` s, no
`MinDt`/`MaxDt`; `fixdt > 0` fixes the step. The right-hand side is
[`llgrhs!`](@ref), so a current set on the world (`setcurrent!`) is included; a
`temperature > 0` [K] adds a thermal field (fixed step required).

Advance it with [`step!`](@ref) or [`advanceto!`](@ref). Counters: `nsteps`,
`nundone`, `nevals`; [`lasterr`](@ref)`(ts)`, [`peakerr`](@ref)`(ts)` (error of the
last accepted step and the largest one; with a fixed step they live on the device
during a run, so read them through these functions, not the fields), and
[`lasttorque`](@ref)`(ts)` [rad/s].
"""
mutable struct TimeStepper{T<:AbstractFloat,W,A<:AbstractArray{T,4},R,C}
    world::W
    m::A
    solver::Symbol
    t::Float64
    dt::Float64
    fixdt::Float64
    maxerr::Float64
    mindt::Float64
    maxdt::Float64
    headroom::Float64
    temperature::Float64
    rng::R
    nsteps::Int
    nundone::Int
    nevals::Int
    lasterr::Float64             # host copies (adaptive: per step; fixed: on demand)
    peakerr::Float64
    ltidx::Int                   # stage holding the last accepted torque (0: none)
    acc::C                       # error accumulator: (this step, peak), squared
    accfixed::Bool               # acc holds fixed-step (dt-scaled) values
    k::Vector{A}                 # stage torques
    m0::A                        # state at the start of the step
    err::A                       # error estimate / scratch
    Bth::Union{Nothing,A}        # thermal field of the current step
    fsal::Bool                   # k[1] holds the torque at the current state
    alarm::Float64               # end of the current run: no step crosses it
    precess::Bool                # false: relaxation torque only (mumax3 DoPrecess)
end

function TimeStepper(world, m::AbstractArray{T,4}; solver::Symbol = :rk45, maxerr = 1e-5,
                     fixdt = 0.0, mindt = 0.0, maxdt = 0.0, headroom = 0.8, dt = 1e-15,
                     temperature = 0.0, rng = Random.default_rng(), t = 0.0, precess::Bool = true) where {T}
    solver in SOLVERS ||
        throw(ArgumentError("unknown solver :$solver; choose one of $(join(SOLVERS, ", "))"))
    fixdt >= 0 && maxerr > 0 && headroom > 0 ||
        throw(ArgumentError("TimeStepper: need fixdt ≥ 0, maxerr > 0, headroom > 0"))
    temperature > 0 && fixdt == 0 &&
        throw(ArgumentError("a finite temperature needs a fixed time step: set fixdt > 0 " *
                            "(the thermal noise variance scales as 1/dt)"))
    solver === :euler && fixdt == 0 && temperature > 0 &&
        throw(ArgumentError("Euler with temperature needs fixdt > 0"))
    k = [similar(m) for _ in 1:_nstage(solver)]
    Bth = temperature > 0 ? similar(m) : nothing
    acc = _newacc(m)
    TimeStepper{T,typeof(world),typeof(m),typeof(rng),typeof(acc)}(
        world, m, solver, Float64(t), fixdt > 0 ? Float64(fixdt) : Float64(dt), Float64(fixdt),
        Float64(maxerr), Float64(mindt), Float64(maxdt), Float64(headroom), Float64(temperature),
        rng, 0, 0, 0, 0.0, 0.0, 0, acc, false, k, similar(m), similar(m), Bth, false, Inf, precess)
end

# One evaluation of the right-hand side (mumax3's torqueFn): LLG + spin torques, with
# the thermal field of the current step added to the effective field. `c`: the stage, at time
# t + c·dt (a time-dependent drive is set for it).
function _eval!(dm, m, ts::TimeStepper, c = 0.0)
    w = ts.world
    B = w._Bbuf
    w.drive === nothing || _drive!(w, ts.t + c * ts.dt)
    effectivefield!(B, m, w)
    ts.Bth === nothing || (B .+= ts.Bth)
    lltorque!(dm, m, B, w.material; precess = ts.precess)
    applycurrent!(dm, m, w)
    ts.nevals += 1
    return dm
end

# Largest per-cell norm |a| (mumax3 MaxVecNorm), in Float64.
_maxvecnorm(a) = Float64(maxtorque(a))

# --- Error accumulator -------------------------------------------------------
# Two running maxima of the squared per-cell norm of s·a or s·(a − b) (s = the step,
# so the values are squared errors |Δm|²): acc[1] for the current step (reset before
# it), acc[2] the peak since the stepper was built. A NaN is sticky in both. On the
# CPU a Vector{T}; the CUDA extension keeps the IEEE bit patterns in a device vector
# and updates them with atomic max (for non-negative floats, and NaN above them, the
# unsigned order of the bits is the order of the values), so nothing is copied to
# the host until _acc_read.
_newacc(m::AbstractArray{T}) where {T} = zeros(T, 2)
_acc_reset!(acc::Vector) = (acc[1] = zero(eltype(acc)); acc)
_acc_clear!(acc) = fill!(acc, zero(eltype(acc)))         # both slots (any backend)
_acc_read(acc::Vector) = (Float64(acc[1]), Float64(acc[2]))
function _acc_max!(acc::Vector{T}, a::AbstractArray{T,4}, b, scale) where {T}
    s = T(scale)
    mx = acc[1]
    @inbounds for k in axes(a, 4), j in axes(a, 3), i in axes(a, 2)
        x, y, z = a[1, i, j, k], a[2, i, j, k], a[3, i, j, k]
        if b !== nothing
            x -= b[1, i, j, k]; y -= b[2, i, j, k]; z -= b[3, i, j, k]
        end
        x *= s; y *= s; z *= s
        v = x * x + y * y + z * z
        (v > mx || isnan(v)) && (mx = v)
        isnan(mx) && break
    end
    acc[1] = mx
    (mx > acc[2] || isnan(mx)) && (acc[2] = mx)
    return acc
end

# Error of the step just computed, max|a|·dt or max|a − b|·dt over the cells. An
# adaptive step reads it (one device→host copy), with the arithmetic of mumax3 and of
# the earlier code (norm in T, then × dt in Float64), so the accepted step sequence
# is unchanged. A fixed step is accepted whatever its error (mumax3): its error,
# scaled by dt on the device, stays there and the function returns 0.
function _steperr!(ts::TimeStepper{T}, a, b = nothing) where {T}
    fixed = ts.fixdt != 0
    if fixed != ts.accfixed                  # the peak slot is only meaningful within a mode
        _acc_clear!(ts.acc)
        ts.accfixed = fixed
    end
    _acc_reset!(ts.acc)
    if fixed
        _acc_max!(ts.acc, a, b, ts.dt)
        return 0.0
    end
    _acc_max!(ts.acc, a, b, one(T))
    return Float64(sqrt(T(_acc_read(ts.acc)[1]))) * ts.dt
end

# Fixed step: copy the device-side errors to the host fields. A non-finite error
# means the magnetization diverged; the steps were accepted (as in mumax3), so the
# run stops here with an error instead of continuing on NaN.
function _syncerr!(ts::TimeStepper)
    (ts.fixdt == 0 || !ts.accfixed) && return ts
    e1, e2 = _acc_read(ts.acc)
    if !isfinite(e2)
        _acc_clear!(ts.acc)                  # do not poison later runs of this stepper
        ts.lasterr = ts.peakerr = 0.0
        error("the magnetization diverged during a fixed-step run (non-finite error " *
              "estimate by t = $(ts.t) s): reduce fixdt")
    end
    ts.lasterr = sqrt(e1)
    ts.peakerr = max(ts.peakerr, sqrt(e2))
    return ts
end

"""
    lasterr(ts), peakerr(ts)

Error estimate (largest per-cell |Δm|) of the last accepted step, and the largest
over all accepted steps of this stepper (mumax3's LastErr and PeakErr).
"""
lasterr(ts::TimeStepper) = (_syncerr!(ts); ts.lasterr)
peakerr(ts::TimeStepper) = (_syncerr!(ts); ts.peakerr)

"""
    lasttorque(ts)

Largest per-cell torque |dm/dt| [rad/s] at the end of the last accepted step
(mumax3's LastTorque), computed on demand; NaN before the first step. After a
rejected step it is the torque at the current (unchanged) state.
"""
lasttorque(ts::TimeStepper) = ts.ltidx == 0 ? NaN : _maxvecnorm(ts.k[ts.ltidx])

# mumax3 adaptDt (engine/run.go), followed by the alarm clip done by the caller.
function _adaptdt!(ts::TimeStepper, corr::Float64)
    if ts.fixdt != 0
        ts.dt = ts.fixdt
        return ts
    end
    isnan(corr) && (corr = 1.0)                  # err = 0: keep the step
    corr == 0 && error("time step too small: the error estimate is not finite")
    corr = clamp(corr * ts.headroom, 0.5, 2.0)
    ts.dt *= corr
    ts.mindt != 0 && ts.dt < ts.mindt && (ts.dt = ts.mindt)
    ts.maxdt != 0 && ts.dt > ts.maxdt && (ts.dt = ts.maxdt)
    return ts
end

_accept(ts::TimeStepper, err) = err < ts.maxerr || ts.dt <= ts.mindt || ts.fixdt != 0

function _setlasterr!(ts::TimeStepper, err)
    ts.lasterr = err
    err > ts.peakerr && (ts.peakerr = err)
end

# mumax3 computes corr = (MaxErr/err)^(1/p) in float64; a non-finite err (a
# diverging trial) gives corr = 0 there and aborts. Here it rejects the step and
# halves dt instead, and gives up only below 1e-20 s.
function _corr(ts::TimeStepper, err, p)
    isfinite(err) || return 0.5 / ts.headroom
    return (ts.maxerr / err)^(1 / p)
end

"""
    step!(ts::TimeStepper) -> Bool

Attempt one step of size `ts.dt`; returns `true` if it was accepted (then `ts.t`
and `ts.m` have advanced), `false` if it was undone (the step size has then been
reduced). The step size for the next attempt is adapted as in mumax3.
"""
function step!(ts::TimeStepper{T}) where {T}
    ts.fixdt != 0 && (ts.dt = ts.fixdt)
    ts.dt > 1e-20 || error("time step $(ts.dt) s too small: the integration diverges " *
                           "(state too far from equilibrium, or parameters not sensible)")
    if ts.Bth !== nothing
        thermalfield!(ts.Bth, ts.world.mesh, ts.world.material, ts.temperature, ts.dt;
                      rng = ts.rng)
        ts.fsal = false                          # the noise changed: k1 is stale
    end
    s = ts.solver
    ok = s === :rk45  ? _step_rk45!(ts) :
         s === :rk4   ? _step_rk4!(ts)  :
         s === :rk23  ? _step_rk23!(ts) :
         s === :heun  ? _step_heun!(ts) : _step_euler!(ts)
    ok || (ts.nundone += 1)
    return ok
end

# --- Euler (mumax3 euler.go) -------------------------------------------------
# Adaptive: the step is chosen so that the largest |Δm| equals MaxErr.
function _step_euler!(ts::TimeStepper{T}) where {T}
    m, dy0 = ts.m, ts.k[1]
    _eval!(dy0, m, ts)
    if ts.fixdt == 0
        τ = _maxvecnorm(dy0)
        τ > 0 || error("Euler: zero torque, cannot choose an adaptive step")
        ts.dt = ts.maxerr / τ
        ts.t + ts.dt > ts.alarm && (ts.dt = ts.alarm - ts.t)
        _setlasterr!(ts, ts.dt * τ)
    else
        _steperr!(ts, dy0)                       # stays on the device
    end
    h = T(ts.dt)
    ts.ltidx = 1
    @. m = m + h * dy0
    normalize!(m)
    ts.t += ts.dt
    ts.nsteps += 1
    return true
end

# --- Heun (mumax3 heun.go) -------------------------------------------------
# The predictor is NOT normalized before the second evaluation, as in mumax3.
function _step_heun!(ts::TimeStepper{T}) where {T}
    m, dy0, dy = ts.m, ts.k[1], ts.k[2]
    h = T(ts.dt)
    _eval!(dy0, m, ts)
    @. m = m + h * dy0
    _eval!(dy, m, ts, 1.0)
    err = _steperr!(ts, dy0, dy)
    if isfinite(err) && _accept(ts, err)
        @. m = m + T(0.5) * h * dy - T(0.5) * h * dy0
        normalize!(m)
        ts.t += ts.dt
        ts.nsteps += 1
        ts.fixdt == 0 && _setlasterr!(ts, err)
        _adaptdt!(ts, _corr(ts, err, 2))
        ts.ltidx = 2
        return true
    else
        @. m = m - h * dy0
        _adaptdt!(ts, _corr(ts, err, 3))
        ts.ltidx = 1                             # dy (k[2]) was overwritten; dy0 is at m
        return false
    end
end

# --- RK4 (mumax3 rk4.go) ---------------------------------------------------
# Error estimate: max |k1 − k4|·h.
function _step_rk4!(ts::TimeStepper{T}) where {T}
    m, m0 = ts.m, ts.m0
    k1, k2, k3, k4 = ts.k[1], ts.k[2], ts.k[3], ts.k[4]
    h = T(ts.dt)
    copyto!(m0, m)
    _eval!(k1, m, ts)
    @. m = m + T(1/2) * h * k1;  normalize!(m);  _eval!(k2, m, ts, 1/2)
    @. m = m0 + T(1/2) * h * k2; normalize!(m);  _eval!(k3, m, ts, 1/2)
    @. m = m0 + h * k3;          normalize!(m);  _eval!(k4, m, ts, 1.0)
    err = _steperr!(ts, k1, k4)
    if isfinite(err) && _accept(ts, err)
        @. m = m0 + T(1/6) * h * k1 + T(1/3) * h * k2 + T(1/3) * h * k3 + T(1/6) * h * k4
        normalize!(m)
        ts.t += ts.dt
        ts.nsteps += 1
        ts.fixdt == 0 && _setlasterr!(ts, err)
        _adaptdt!(ts, _corr(ts, err, 4))
        ts.ltidx = 4
        return true
    else
        copyto!(m, m0)
        _adaptdt!(ts, _corr(ts, err, 5))
        ts.ltidx = 1                             # k4 was overwritten; k1 is at m
        return false
    end
end

# --- Bogacki–Shampine RK23 (mumax3 rk23.go), FSAL ----------------------------
function _step_rk23!(ts::TimeStepper{T}) where {T}
    m, m0 = ts.m, ts.m0
    k1, k2, k3, k4 = ts.k[1], ts.k[2], ts.k[3], ts.k[4]
    ts.fsal || (_eval!(k1, m, ts); ts.fsal = true)
    h = T(ts.dt)
    copyto!(m0, m)
    @. m = m + T(1/2) * h * k1;  normalize!(m);  _eval!(k2, m, ts, 1/2)
    @. m = m0 + T(3/4) * h * k2; normalize!(m);  _eval!(k3, m, ts, 3/4)
    @. m = m0 + T(2/9) * h * k1 + T(1/3) * h * k2 + T(4/9) * h * k3   # 3rd order
    normalize!(m)
    _eval!(k4, m, ts, 1.0)
    e = ts.err
    @. e = T(7/24 - 2/9) * k1 + T(1/4 - 1/3) * k2 + T(1/3 - 4/9) * k3 + T(1/8) * k4
    err = _steperr!(ts, e)
    if isfinite(err) && _accept(ts, err)
        ts.fixdt == 0 && _setlasterr!(ts, err)
        ts.ltidx = 1
        ts.t += ts.dt
        ts.nsteps += 1
        _adaptdt!(ts, _corr(ts, err, 3))
        copyto!(k1, k4)                                                # FSAL
        return true
    else
        copyto!(m, m0)
        _adaptdt!(ts, _corr(ts, err, 4))
        return false
    end
end

# --- Dormand–Prince RK45 (mumax3 rk45dp.go), FSAL ----------------------------
function _step_rk45!(ts::TimeStepper{T}) where {T}
    m, m0 = ts.m, ts.m0
    k1, k2, k3, k4, k5, k6, k7 = ts.k[1], ts.k[2], ts.k[3], ts.k[4], ts.k[5], ts.k[6], ts.k[7]
    ts.fsal || (_eval!(k1, m, ts); ts.fsal = true)
    h = T(ts.dt)
    copyto!(m0, m)
    @. m = m + T(1/5) * h * k1
    normalize!(m); _eval!(k2, m, ts, 1/5)
    @. m = m0 + T(3/40) * h * k1 + T(9/40) * h * k2
    normalize!(m); _eval!(k3, m, ts, 3/10)
    @. m = m0 + T(44/45) * h * k1 + T(-56/15) * h * k2 + T(32/9) * h * k3
    normalize!(m); _eval!(k4, m, ts, 4/5)
    @. m = m0 + T(19372/6561) * h * k1 + T(-25360/2187) * h * k2 + T(64448/6561) * h * k3 +
           T(-212/729) * h * k4
    normalize!(m); _eval!(k5, m, ts, 8/9)
    @. m = m0 + T(9017/3168) * h * k1 + T(-355/33) * h * k2 + T(46732/5247) * h * k3 +
           T(49/176) * h * k4 + T(-5103/18656) * h * k5
    normalize!(m); _eval!(k6, m, ts, 1.0)
    @. m = m0 + T(35/384) * h * k1 + T(500/1113) * h * k3 + T(125/192) * h * k4 +
           T(-2187/6784) * h * k5 + T(11/84) * h * k6                    # 5th order
    normalize!(m); _eval!(k7, m, ts, 1.0)
    e = ts.err
    @. e = T(35/384 - 5179/57600) * k1 + T(500/1113 - 7571/16695) * k3 +
           T(125/192 - 393/640) * k4 + T(-2187/6784 + 92097/339200) * k5 +
           T(11/84 - 187/2100) * k6 + T(-1/40) * k7
    err = _steperr!(ts, e)
    if isfinite(err) && _accept(ts, err)
        ts.fixdt == 0 && _setlasterr!(ts, err)
        ts.ltidx = 1                                                   # k7, copied to k1
        ts.t += ts.dt
        ts.nsteps += 1
        _adaptdt!(ts, _corr(ts, err, 5))
        copyto!(k1, k7)                                                # FSAL
        return true
    else
        copyto!(m, m0)
        _adaptdt!(ts, _corr(ts, err, 6))
        return false
    end
end

"""
    advanceto!(ts, tend; onstep = ts -> nothing)

Step until `ts.t` reaches `tend` (mumax3's `Run`): an adaptive step is clipped so as
not to cross `tend`; a fixed step `fixdt` is shortened only for a final remainder
larger than 10⁻⁶·fixdt. `onstep(ts)` is called after every accepted step (the
output hook).
"""
function advanceto!(ts::TimeStepper, tend::Real; onstep = _ -> nothing)
    tend = Float64(tend)
    ts.alarm = tend
    ts.fsal = false              # the state or the world may have changed (mumax3 frees its stepper per Run)
    while ts.t < tend
        if ts.fixdt != 0
            rem = tend - ts.t
            rem <= 1e-6 * ts.fixdt && (ts.t = tend; break)
            if rem < ts.fixdt                      # final partial step
                fd = ts.fixdt
                ts.fixdt = rem
                try
                    step!(ts)
                finally
                    ts.fixdt = fd
                    ts.dt = fd
                end
                ts.t = tend
                onstep(ts)
                break
            end
        elseif ts.t + ts.dt > tend
            ts.dt = tend - ts.t                    # mumax3's alarm clip
        end
        step!(ts) && onstep(ts)
    end
    _syncerr!(ts)                                  # fixed step: errors to the host, NaN check
    return ts
end
