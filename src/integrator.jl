# LLG time integration via OrdinaryDiffEq.jl.
#
# The hand-rolled Dormand-Prince solver in solver.jl is kept for reference and
# for the Larmor tests, but the production integrator uses OrdinaryDiffEq's
# community-validated, PI-controlled adaptive methods. The magnetization is
# integrated as a plain ODE du/dt = f(u) with u === the (3,Nx,Ny,Nz) array; a
# per-step callback renormalizes |m| = 1, which the LLG torque only conserves to
# the order of the integrator.
#
# Default method is Tsit5 — a 5th-order explicit Runge-Kutta (Tsitouras 2011),
# the efficient modern replacement for Dormand-Prince RK45 (mumax3's method) for
# non-stiff problems, which micromagnetic LLG dynamics are.

using OrdinaryDiffEqTsit5: ODEProblem, Tsit5, init, solve, step!, DiscreteCallback,
                           CallbackSet, terminate!, u_modified!, savevalues!
import OrdinaryDiffEqTsit5 as ODE

# RHS of the LLG equation as an in-place ODE function. `p` carries the World.
function _llg_rhs!(du, u, world::World, t)
    B = world._Bbuf                       # reuse the World's field buffer
    effectivefield!(B, u, world)
    torque!(du, u, B, world.material.alpha)
    return nothing
end

# Callback that renormalizes |m| after every accepted step.
function _normalize_callback()
    condition(u, t, integrator) = true            # fire every step
    function affect!(integrator)
        normalize!(integrator.u)
        u_modified!(integrator, false)            # normalization doesn't affect error control
    end
    DiscreteCallback(condition, affect!; save_positions = (false, false))
end

"""
    Integrator(world, m; alg=Tsit5(), abstol=1e-6, reltol=1e-5, dtmax=1e-11)

Wrap an OrdinaryDiffEq integrator around the LLG equation for state `m` under
`world`. Advance it with [`advance!`](@ref) / [`relaxate!`](@ref). `m` is
integrated in place; `integrator.u` aliases it.
"""
mutable struct Integrator{T,I}
    world::World{T}
    integrator::I
end

function Integrator(world::World{T}, m::Array{T,4};
                    alg = Tsit5(), abstol = 1e-6, reltol = 1e-5,
                    dt = 1e-15, dtmax = 1e-11, tend = 1.0) where {T}
    prob = ODEProblem(_llg_rhs!, m, (0.0, tend), world)
    integ = init(prob, alg; abstol = abstol, reltol = reltol, dt = dt,
                 dtmax = dtmax, save_everystep = false, save_start = false,
                 callback = _normalize_callback())
    Integrator{T,typeof(integ)}(world, integ)
end

"Current simulated time [s]."
currenttime(it::Integrator) = it.integrator.t

"Current magnetization state (aliases the integrated array)."
state(it::Integrator) = it.integrator.u

"""
    advance!(it, duration)

Integrate for `duration` seconds of simulated time.
"""
function advance!(it::Integrator, duration::Real)
    t_target = it.integrator.t + duration
    ODE.step!(it.integrator, duration, true)      # advance exactly `duration`
    return it
end

"""
    relaxate!(it; stoptorque, maxtime=1e-6, checkevery=1e-11)

Integrate the (damped) LLG until the maximum torque falls below `stoptorque`
[rad/s], or `maxtime` is reached. Use a heavily damped material for speed.
"""
function relaxate!(it::Integrator{T}; stoptorque = T(1e-3), maxtime = 1e-7,
                   checkevery = 1e-11) where {T}
    B = it.world._Bbuf
    dm = similar(state(it))
    t_end = it.integrator.t + maxtime
    while it.integrator.t < t_end
        ODE.step!(it.integrator, checkevery, true)
        effectivefield!(B, state(it), it.world)
        torque!(dm, state(it), B, it.world.material.alpha)
        maxtorque(dm) < stoptorque && break
    end
    return it
end
