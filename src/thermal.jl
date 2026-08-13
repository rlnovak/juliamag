# Finite-temperature (thermal / Langevin) dynamics.
#
# Thermal fluctuations are modelled by Brown's approach: a random field added to
# the effective field, whose statistics reproduce the fluctuation-dissipation
# theorem. Ported from mumax3 (engine/temperature.go, cuda/temperature2.cu). Per
# cell and per Cartesian component the thermal field is
#
#     B_therm = η · sqrt( 2 α kB T / (γ Msat V Δt) )
#
# where η is a standard normal (mean 0, variance 1) drawn independently for each
# component every step, α the Gilbert damping, T the temperature [K], V the cell
# volume, Δt the integration step, and γ = γLL. This matches mumax3's kernel
#     B = noise * sqrt( (2 kB / (γ V Δt)) · α · T / Msat ).
#
# The 1/Δt makes the field a white-noise increment: its variance scales as 1/Δt
# so that the magnetization increment has the correct, step-size-independent
# variance. A finite-temperature run therefore needs a FIXED step (or the noise
# must be rescaled by sqrt(Δt_old/Δt_new) on a step change), so we integrate with
# a fixed-step stochastic Heun scheme (the standard choice for the LLG-Langevin
# equation, converging in the Stratonovich sense) rather than the adaptive solver.

using Random

"""
    thermalfield!(Btherm, mesh, mat, T, dt; rng=Random.default_rng())

Fill `Btherm` (a `(3,Nx,Ny,Nz)` array) with a fresh thermal field for temperature
`T` [K] and step `dt` [s]: each component is an independent normal scaled by
`sqrt(2 α kB T / (γLL Msat V dt))`. Empty cells (Msat = 0) get zero field.
"""
function thermalfield!(Btherm::AbstractArray{T2,4}, mesh::Mesh, params::AbstractParams,
                       temp::Real, dt::Real; rng = Random.default_rng()) where {T2}
    Nx, Ny, Nz = mesh.size
    V = cellvolume(mesh)
    randn!(rng, Btherm)                       # unit normals in place
    kfac = T2(2 * kB * temp / (γLL * V * dt)) # 2 kB T / (γ V Δt)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Msc = msat(params, i, j, k)
        α = alphaof(params, i, j, k)
        s = (Msc == 0 || temp <= 0) ? zero(T2) : sqrt(kfac * α / Msc)
        Btherm[1, i, j, k] *= s
        Btherm[2, i, j, k] *= s
        Btherm[3, i, j, k] *= s
    end
    return Btherm
end

"""
    runthermal!(sim, duration, temperature; dt=1e-14, every=nothing, rng)

Integrate the LLG–Langevin equation at finite `temperature` [K] for `duration`
seconds with a fixed step `dt`, appending a table row every `every` seconds (or
never, if `every === nothing`). Uses a stochastic Heun scheme with the same
thermal field in the predictor and corrector stage (Stratonovich).

The step must be small enough that the thermal kick per step stays modest; `dt`
around 1e-14–1e-13 s is typical. A strongly damped material equilibrates faster.
"""
function runthermal!(sim::Simulation{T}, duration::Real, temperature::Real;
                     dt = 1e-14, every = nothing, rng = Random.default_rng()) where {T}
    world, mesh, params = sim.world, sim.world.mesh, sim.world.material
    m = sim.m
    B  = world._Bbuf
    Bth = similar(m)
    k1 = similar(m); k2 = similar(m); mp = similar(m)
    α = damping(params)
    dtT = T(dt); tempT = T(temperature)
    nsteps = max(1, round(Int, duration / dt))
    saveevery = every === nothing ? typemax(Int) : max(1, round(Int, every / dt))

    _thermrhs!(dm, mm) = begin           # dm = LLG torque at effective+thermal field
        effectivefield!(B, mm, world)
        B .+= Bth
        torque!(dm, mm, B, α)
        dm
    end

    every === nothing || tablesave!(sim.table, world, m, sim.t)
    for step in 1:nsteps
        thermalfield!(Bth, mesh, params, tempT, dtT; rng = rng)   # one draw per step
        _thermrhs!(k1, m)
        @. mp = m + dtT * k1; normalize!(mp)                      # Heun predictor
        _thermrhs!(k2, mp)                                        # same Bth (Stratonovich)
        @. m = m + (dtT/2) * (k1 + k2); normalize!(m)             # corrector
        sim.t += dtT
        (step % saveevery == 0) && tablesave!(sim.table, world, m, sim.t)
    end
    return sim
end
