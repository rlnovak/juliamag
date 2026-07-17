# User-facing simulation wrapper.
#
# Bundles a World, the current magnetization, a data table, and its save
# interval, and drives the run loop — so a typical script reads like the mumax3
# workflow (set geometry/material, choose what to save, relax, run) rather than
# wiring up a World, an Integrator, and a table by hand.
#
#   sim = Simulation(mesh, material("Permalloy"))
#   setmag!(sim, VortexConfig(mesh))
#   savequantities!(sim, q_time(), q_m(), q_vortexcore(); every = 10e-12)
#   relax!(sim)
#   run!(sim, 1e-9)
#   writetable(sim.table, "out.txt")

"""
    Simulation(mesh, params; demag=true, Bext=(0,0,0))

A ready-to-run simulation: geometry, material parameters, magnetization, and an
output table. Use [`setmag!`](@ref) to set the initial state,
[`savequantities!`](@ref) to choose table columns and the save interval,
[`relax!`](@ref)/[`minimize!`](@ref) to reach equilibrium, and [`run!`](@ref) to
integrate in time.
"""
mutable struct Simulation{T<:AbstractFloat}
    world::World{T}
    m::Array{T,4}
    table::DataTable
    t::T
end

function Simulation(mesh::Mesh, params::AbstractParams; demag = true, Bext = (0, 0, 0))
    T = eltype(params)
    world = World(mesh, params; demag = demag, Bext = Bext)
    m = uniform(T, mesh, (1, 0, 0))
    clearempty!(m, params)
    Simulation{T}(world, m, DataTable(), zero(T))
end

"The mesh of a simulation."
mesh(sim::Simulation) = sim.world.mesh

"""
    setmag!(sim, config)   /   setmag!(sim, m)

Set the initial magnetization from a Config (sampled on the mesh) or a ready
`(3,Nx,Ny,Nz)` array. Empty cells are cleared.
"""
function setmag!(sim::Simulation, config::Config)
    setconfig!(sim.m, sim.world.mesh, config)
    clearempty!(sim.m, sim.world.material)
    return sim
end
function setmag!(sim::Simulation{T}, m::AbstractArray{T,4}) where {T}
    copyto!(sim.m, m)
    clearempty!(sim.m, sim.world.material)
    return sim
end

"Set the uniform applied field [T]."
setfield!(sim::Simulation, Bext) = (setexternalfield!(sim.world, Bext); sim)

"""
    savequantities!(sim, quantities...; every=0.0)

Set the output-table columns and the auto-save interval `every` [s]. Time and
⟨m⟩ are always included as the first columns.
"""
function savequantities!(sim::Simulation, quantities::Quantity...; every = 0.0)
    tbl = DataTable(autosave = every)
    for q in quantities
        tableadd!(tbl, q)
    end
    sim.table = tbl
    return sim
end

"Append one row to the table now."
savenow!(sim::Simulation) = (tablesave!(sim.table, sim.world, sim.m, sim.t); sim)

"""
    relax!(sim; kwargs...)

Relax the magnetization to equilibrium with the energy minimizer, in place.
"""
function relax!(sim::Simulation; kwargs...)
    mn = Minimizer(sim.world, sim.m; kwargs...)
    minimize!(mn)
    copyto!(sim.m, mn.m)
    return sim
end

"""
    run!(sim, duration; abstol=1e-6, reltol=1e-5)

Integrate the LLG for `duration` seconds, appending a table row every
`sim.table.autosave` seconds (if set) and once at the end. Uses OrdinaryDiffEq.
"""
function run!(sim::Simulation{T}, duration::Real; abstol = 1e-6, reltol = 1e-5) where {T}
    it = Integrator(sim.world, sim.m; abstol = abstol, reltol = reltol, tend = sim.t + duration)
    tbl = sim.table
    dt = tbl.autosave
    t_end = sim.t + T(duration)

    # Save the initial row.
    tbl.nextsave = sim.t
    tablesave!(tbl, sim.world, sim.m, sim.t)

    if dt > 0
        while sim.t < t_end
            target = min(sim.t + T(dt), t_end)
            advance!(it, target - sim.t)
            sim.t = currenttime(it)
            copyto!(sim.m, state(it))
            tablesave!(tbl, sim.world, sim.m, sim.t)
        end
    else
        advance!(it, duration)
        sim.t = currenttime(it)
        copyto!(sim.m, state(it))
        tablesave!(tbl, sim.world, sim.m, sim.t)
    end
    return sim
end

"""
    runcurrent!(sim, J, duration; every=20e-12, dt=5e-14)

Integrate the LLG with an in-plane Zhang-Li spin-transfer current `J = (Jx,Jy,Jz)`
[A/m²] for `duration` seconds, appending a table row every `every` seconds. Uses
a fixed-step fourth-order Runge-Kutta with the current torque added to the LLG
torque (the STT is a torque, not a field, so it is not set through
`setexternalfield!`). The material's `pol` and `xi` set the polarization and
non-adiabaticity.

Use this to drive a domain wall or skyrmion with a current (see the manual's
skyrmion tutorial and µMAG standard problem 5).
"""
function runcurrent!(sim::Simulation{T}, J, duration::Real;
                     every = 20e-12, dt = 5e-14) where {T}
    world, msh, mat = sim.world, sim.world.mesh, sim.world.material
    tbl = sim.table
    Jt = NTuple{3,T}(J)
    nchunks = max(1, round(Int, duration / every))
    substeps = max(1, round(Int, every / dt))
    tablesave!(tbl, world, sim.m, sim.t)
    for _ in 1:nchunks
        _rk4_current!(sim.m, world, msh, mat, Jt, T(dt), substeps)
        sim.t += T(every)
        tablesave!(tbl, world, sim.m, sim.t)
    end
    return sim
end

# Fixed-step RK4 of dm/dt = LLG torque + Zhang-Li STT, `nsteps` of size `dt`.
function _rk4_current!(m::AbstractArray{T,4}, world, mesh::Mesh, mat, J, dt, nsteps) where {T}
    k1 = similar(m); k2 = similar(m); k3 = similar(m); k4 = similar(m); tmp = similar(m); B = world._Bbuf
    rhs!(dm, mm) = begin
        effectivefield!(B, mm, world)
        torque!(dm, mm, B, damping(mat))
        zhanglitorque!(dm, mm, mesh, mat, J; add = true)
        dm
    end
    for _ in 1:nsteps
        rhs!(k1, m)
        @. tmp = m + (dt/2)*k1; normalize!(tmp); rhs!(k2, tmp)
        @. tmp = m + (dt/2)*k2; normalize!(tmp); rhs!(k3, tmp)
        @. tmp = m + dt*k3;     normalize!(tmp); rhs!(k4, tmp)
        @. m = m + (dt/6)*(k1 + 2k2 + 2k3 + k4)
        normalize!(m)
    end
    return m
end

"Average magnetization ⟨mx,my,mz⟩ of the current state."
average(sim::Simulation) = average(sim.m)

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation)
    print(io, "Simulation\n")
    show(io, MIME"text/plain"(), sim.world.mesh); println(io)
    print(io, "  t = ", sim.t * 1e9, " ns, table columns: ",
          join([q.name for q in sim.table.quantities], ", "))
end
