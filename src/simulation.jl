# User-facing simulation wrapper.
#
# Bundles a World, the current magnetization, a data table, the solver settings and
# the output schedule, and drives the run loop — so a problem script reads like a
# mumax3 input file rather than wiring up a World, an integrator and a table by hand:
#
#   sim = Simulation(mesh, material("Permalloy"); outdir = "run.out")
#   setmag!(sim, VortexConfig(mesh))
#   tableadd!(sim, q_maxtorque()); tableautosave!(sim, 10e-12)   # TableAdd, TableAutosave
#   autosave!(sim, 100e-12)                                      # AutoSave(m, …)
#   setcurrent!(sim, (0, 0, 3e12); zhangli = false, slonczewski = true,
#               fixedlayer = (1, 0, 0))                          # J, FixedLayer
#   relax!(sim)                                                  # Minimize()
#   run!(sim, 1e-9)                                              # Run(1e-9), RK45 by default
#
# The solvers are mumax3's (timestepper.jl), selected by name (setsolver! or the
# `solver` keyword of run!), plus OrdinaryDiffEq's Tsit5 (:tsit5). A Simulation moved
# to the GPU with togpu(sim) runs the same script there; table columns that have no
# GPU method are evaluated on a host copy of the state.

"""
    Simulation(mesh, params; demag=true, Bext=(0,0,0), outdir=nothing, verbose=true)

A ready-to-run simulation: geometry, material parameters, magnetization, solver
settings and output. `outdir` (created if needed) receives the streamed table
(`table.txt`) and the OVF snapshots; without it the table is kept in memory only.

Main verbs (mumax3 equivalents in brackets):
[`setmag!`](@ref) [m = …], [`setexternalfield!`](@ref) [B_ext], [`setcurrent!`](@ref)
[J, FixedLayer, …], [`settemperature!`](@ref) [Temp], [`setsolver!`](@ref)
[SetSolver, MaxErr, FixDt, …], [`tableadd!`](@ref)/[`tableautosave!`](@ref)/
[`tablesave!`](@ref) [TableAdd/TableAutosave/TableSave], [`save!`](@ref)/
[`autosave!`](@ref) [Save/AutoSave of m], [`relax!`](@ref) [Minimize],
[`run!`](@ref) [Run].

The time `sim.t` is kept in Float64 whatever the precision of the run.

`togpu(sim)` returns a GPU simulation that shares the table, the output schedule
and the RNG with `sim` (use it as `sim = togpu(sim)`); on the GPU the thermal
field is drawn with CUDA's generator, so `rng` does not seed it.
"""
mutable struct Simulation{T<:AbstractFloat,W<:World{T},A<:AbstractArray{T,4},H<:World{T}}
    world::W                     # the World the dynamics runs on (CPU or GPU)
    m::A
    table::DataTable
    t::Float64
    hostworld::H                 # CPU World for host-side diagnostics (=== world on the CPU)
    # Solver settings (mumax3 names and defaults).
    solver::Symbol               # :rk45 (default), :rk23, :rk4, :heun, :euler, :tsit5
    maxerr::Float64              # MaxErr
    fixdt::Float64               # FixDt (0: adaptive)
    mindt::Float64               # MinDt (0: none)
    maxdt::Float64               # MaxDt (0: none)
    headroom::Float64            # Headroom
    dt::Float64                  # current step (carried over between runs, as Dt_si)
    abstol::Float64              # Tsit5 tolerances
    reltol::Float64
    temperature::Float64         # Temp [K]
    rng::Any
    # Counters over the whole simulation (NSteps, NUndone, NEvals, LastErr, PeakErr).
    nsteps::Int
    nundone::Int
    nevals::Int
    lasterr::Float64
    peakerr::Float64
    # Output.
    outdir::Union{Nothing,String}
    ovf::Vector{Any}             # OVF autosaves: (name, period, start, count, field) per quantity
    autonum::Dict{String,Int}    # next file number per OVF name
    verbose::Bool
    log::IO
    _stepper::Any                # cached TimeStepper (buffers), or nothing
    precess::Bool                # mumax3 DoPrecess
end

function Simulation(mesh::Mesh, params::AbstractParams; demag = true, Bext = (0, 0, 0),
                    openbc::Bool = false, outdir = nothing, verbose::Bool = true, log::IO = stdout)
    T = eltype(params)
    world = World(mesh, params; demag = demag, Bext = Bext, openbc = openbc)
    m = uniform(T, mesh, (1, 0, 0))
    clearempty!(m, params)
    sim = _newsim(world, m, world; verbose = verbose, log = log)
    outdir === nothing || setoutdir!(sim, outdir)
    return sim
end

function _newsim(world::World{T}, m, hostworld; verbose = true, log = stdout) where {T}
    Simulation{T,typeof(world),typeof(m),typeof(hostworld)}(
        world, m, DataTable(), 0.0, hostworld,
        :rk45, 1e-5, 0.0, 0.0, 0.0, 0.8, 1e-15, 1e-6, 1e-5, 0.0, Random.default_rng(),
        0, 0, 0, 0.0, 0.0,
        nothing, Any[], Dict{String,Int}(), verbose, log, nothing, true)
end

# Copy of `sim` with another world / state (togpu, tocpu); settings and output kept.
function _rebuild(sim::Simulation, world::World{T}, m, hostworld) where {T}
    s = Simulation{T,typeof(world),typeof(m),typeof(hostworld)}(
        world, m, sim.table, sim.t, hostworld,
        sim.solver, sim.maxerr, sim.fixdt, sim.mindt, sim.maxdt, sim.headroom, sim.dt,
        sim.abstol, sim.reltol, sim.temperature, sim.rng,
        sim.nsteps, sim.nundone, sim.nevals, sim.lasterr, sim.peakerr,
        sim.outdir, sim.ovf, sim.autonum, sim.verbose, sim.log, nothing, sim.precess)
    return s
end

"The mesh of a simulation."
mesh(sim::Simulation) = sim.world.mesh

ondevice(sim::Simulation) = !(sim.m isa Array)

# The state on the host (a copy if the simulation runs on the GPU).
_hoststate(sim::Simulation) = sim.m isa Array ? sim.m : Array(sim.m)

# --- Setting up --------------------------------------------------------------

"""
    setmag!(sim, config)   /   setmag!(sim, m)

Set the magnetization from a Config (sampled on the mesh) or a ready
`(3,Nx,Ny,Nz)` array. Empty cells are cleared.
"""
function setmag!(sim::Simulation{T}, config::Config) where {T}
    mh = sim.m isa Array ? sim.m : zeros(T, size(sim.m))
    setconfig!(mh, sim.world.mesh, config)
    clearempty!(mh, sim.hostworld.material)
    mh === sim.m || copyto!(sim.m, mh)
    return sim
end
function setmag!(sim::Simulation{T}, m::AbstractArray{T,4}) where {T}
    mh = Array(m)
    clearempty!(mh, sim.hostworld.material)
    copyto!(sim.m, mh)
    return sim
end

"""
    setfield!(sim, Bext)   /   setexternalfield!(sim, Bext)

Set the uniform applied field [T] (mumax3 `B_ext`).
"""
function setexternalfield!(sim::Simulation, Bext)
    setexternalfield!(sim.world, Bext)
    sim.hostworld === sim.world || setexternalfield!(sim.hostworld, Bext)
    return sim
end
setfield!(sim::Simulation, Bext) = setexternalfield!(sim, Bext)

"""
    setcurrent!(sim, J; zhangli=true, slonczewski=false, fixedlayer=(0,0,1),
                thickness=<mesh thickness>, params=sttparams(material))
    setcurrent!(sim, nothing)

Set a uniform charge-current density `J` [A/m²] and choose its spin torques (mumax3
`J`, `FixedLayer`, `FreeLayerThickness`, `Pol`, `Lambda`, `EpsilonPrime`, `xi`).
Every solver then integrates the LLG plus these torques. Unlike mumax3, which adds
the Zhang-Li torque whenever `J` is set (unless `DisableZhangLiTorque`), each torque
is enabled explicitly. `nothing` removes the current.
"""
function setcurrent!(sim::Simulation, J; kwargs...)
    setcurrent!(sim.hostworld, J; kwargs...)
    sim.hostworld === sim.world || (sim.world.current = sim.hostworld.current)
    return sim
end
function setcurrent!(sim::Simulation, ::Nothing)
    setcurrent!(sim.hostworld, nothing)
    sim.world.current = nothing
    return sim
end

setdrive!(sim::Simulation, f) = (setdrive!(sim.world, f); setdrive!(sim.hostworld, f); sim)
function addfield!(sim::Simulation, map, f = 1.0)
    addfield!(sim.world, map, f; t = sim.t)
    sim.hostworld === sim.world || addfield!(sim.hostworld, map, f; t = sim.t)
    return sim
end
function addcurrent!(sim::Simulation, map, f = 1.0)
    addcurrent!(sim.world, map, f; t = sim.t)
    sim.hostworld === sim.world || addcurrent!(sim.hostworld, map, f; t = sim.t)
    return sim
end
clearfields!(sim::Simulation) = (clearfields!(sim.world); clearfields!(sim.hostworld); sim)
clearcurrents!(sim::Simulation) = (clearcurrents!(sim.world); clearcurrents!(sim.hostworld); sim)
"The vector `v` in region `id` of the simulation's regions, zero elsewhere ([`regionmap`](@ref))."
regionmap(sim::Simulation, id::Integer, v) = regionmap(sim.hostworld.material, id, v)

"""
    settemperature!(sim, T; rng=Random.default_rng())

Set the temperature [K] (mumax3 `Temp`). A finite temperature adds a thermal field,
redrawn every step; it requires a fixed time step (`setsolver!(sim, …; fixdt=…)`).
"""
function settemperature!(sim::Simulation, temperature::Real; rng = nothing)
    temperature >= 0 || throw(ArgumentError("temperature must be ≥ 0"))
    sim.temperature = Float64(temperature)
    rng === nothing || (sim.rng = rng)
    sim._stepper = nothing
    return sim
end

"""
    setsolver!(sim, solver=sim.solver; maxerr, fixdt, mindt, maxdt, headroom,
               abstol, reltol, precess)

Choose the time integrator by name — mumax3's `:euler`, `:heun`, `:rk23`, `:rk4`,
`:rk45` (Dormand–Prince, the default), or OrdinaryDiffEq's `:tsit5` — and its
settings, with mumax3's names and defaults: `maxerr = 1e-5` (MaxErr), `fixdt = 0`
(FixDt; > 0 fixes the step), `mindt`, `maxdt` (0 = no bound), `headroom = 0.8`.
`abstol`, `reltol` apply to `:tsit5` only. `precess = false` integrates the relaxation torque
−γ m×(m×B) alone (mumax3 `DoPrecess = false`; not with `:tsit5`).
"""
function setsolver!(sim::Simulation, solver::Symbol = sim.solver;
                    maxerr = sim.maxerr, fixdt = sim.fixdt, mindt = sim.mindt,
                    maxdt = sim.maxdt, headroom = sim.headroom,
                    abstol = sim.abstol, reltol = sim.reltol, precess::Bool = sim.precess)
    solver in SOLVERS || solver === :tsit5 ||
        throw(ArgumentError("unknown solver :$solver; choose one of " *
                            join(map(string, (SOLVERS..., :tsit5)), ", ")))
    sim.solver = solver
    sim.maxerr = maxerr; sim.fixdt = fixdt; sim.mindt = mindt; sim.maxdt = maxdt
    sim.headroom = headroom; sim.abstol = abstol; sim.reltol = reltol; sim.precess = precess
    sim._stepper = nothing
    return sim
end

# --- Output ------------------------------------------------------------------

"""
    setoutdir!(sim, dir)

Write the output into `dir` (created if needed): the table is streamed to
`dir/table.txt` as rows are saved, OVF snapshots go to `dir/<name>NNNNNN.ovf`.
"""
function setoutdir!(sim::Simulation, dir::AbstractString)
    mkpath(dir)
    sim.outdir = String(dir)
    sim.table.path = joinpath(dir, "table.txt")
    sim.table.written = false                  # the next row starts the file afresh
    return sim
end

"""
    savequantities!(sim, quantities...; every=0.0)

Set the output-table columns and the auto-save interval `every` [s]. Time and
⟨m⟩ are always included as the first columns.
"""
function savequantities!(sim::Simulation, quantities::Quantity...; every = 0.0)
    tbl = DataTable(autosave = every, start = sim.t,
                    path = sim.outdir === nothing ? nothing : joinpath(sim.outdir, "table.txt"))
    for q in quantities
        q.name in ("t", "m") && q.ncomp == (q.name == "t" ? 1 : 3) && continue
        tableadd!(tbl, q)
    end
    sim.table = tbl
    return sim
end

"""
    tableadd!(sim, quantity)

Add a column to the table (mumax3 `TableAdd`). The table starts with t and ⟨m⟩
(the magnet average, `average_magnet`).
"""
tableadd!(sim::Simulation, q::Quantity) = (tableadd!(sim.table, q); sim)

"""
    tableautosave!(sim, period)

Append a table row every `period` seconds of simulated time during [`run!`](@ref),
starting with the current state (mumax3 `TableAutosave`). `0` stops it.
"""
function tableautosave!(sim::Simulation, period::Real)
    sim.table.autosave = Float64(period)
    sim.table.start = sim.t
    sim.table.count = -1
    return sim
end

"""
    tablesave!(sim)   /   savenow!(sim)

Append one table row now (mumax3 `TableSave`). On the GPU, the columns without a
GPU method are evaluated on a host copy of the state.
"""
function tablesave!(sim::Simulation)
    _drive!(sim.world, sim.t); _drive!(sim.hostworld, sim.t)
    row = Float64[]
    mh = nothing
    for q in sim.table.quantities
        if q.device || !ondevice(sim)
            append!(row, q(sim.world, sim.m, sim.t))
        else
            mh === nothing && (mh = Array(sim.m))
            append!(row, q(sim.hostworld, mh, sim.t))
        end
    end
    _pushrow!(sim.table, row)
    return sim
end
savenow!(sim::Simulation) = tablesave!(sim)

"Write the in-memory table to `path` (tab-separated, with a header)."
writetable(sim::Simulation, path::AbstractString) = writetable(sim.table, path)

"""
    save!(sim; name="m", field=nothing)

Write the magnetization to `outdir/<name>NNNNNN.ovf` now, numbering the files per
name from 0 (mumax3 `Save(m)`). Requires an output directory ([`setoutdir!`](@ref)). Another
field is written with `field`, a function of the simulation that returns a host array
(1 or 3, Nx, Ny, Nz), or that and its mesh.
"""
function save!(sim::Simulation; name::AbstractString = "m", field = nothing)
    sim.outdir === nothing &&
        throw(ArgumentError("save!: set an output directory first (setoutdir! or outdir=)"))
    n = get(sim.autonum, name, 0)
    path = joinpath(sim.outdir, @sprintf("%s%06d.ovf", name, n))
    _drive!(sim.world, sim.t); _drive!(sim.hostworld, sim.t)
    a = field === nothing ? _hoststate(sim) : field(sim)
    f, msh = a isa Tuple ? a : (a, sim.world.mesh)
    saveovf(path, f, msh;
            title = @sprintf("%s t=%.6e s", name, sim.t))
    sim.autonum[name] = n + 1
    return path
end

"""
    autosave!(sim, period; name="m", field=nothing)

Save the magnetization as OVF every `period` seconds of simulated time during
[`run!`](@ref), starting with the current state (mumax3 `AutoSave(m, period)`).
`0` stops it. `field`: as in [`save!`](@ref).
"""
function autosave!(sim::Simulation, period::Real; name::AbstractString = "m", field = nothing)
    filter!(a -> a[1] != name, sim.ovf)
    if period != 0
        sim.outdir === nothing &&
            throw(ArgumentError("autosave!: set an output directory first (setoutdir! or outdir=)"))
        push!(sim.ovf, Any[String(name), Float64(period), sim.t, -1, field])
    end
    return sim
end

# mumax3 DoOutput: every autosave that is due.
# Whether some autosave (OVF or table) is due at sim.t.
_outputdue(sim::Simulation) =
    any(a -> _needsave(a[2], a[3], a[4], sim.t), sim.ovf) ||
    _needsave(sim.table.autosave, sim.table.start, sim.table.count, sim.t)

function _dooutput!(sim::Simulation)
    for a in sim.ovf
        if _needsave(a[2], a[3], a[4], sim.t)
            save!(sim; name = a[1], field = a[5])
            a[4] += 1
        end
    end
    tbl = sim.table
    if _needsave(tbl.autosave, tbl.start, tbl.count, sim.t)
        tablesave!(sim)
        tbl.count += 1
    end
    return sim
end

# --- Logging -----------------------------------------------------------------

"Description of the device an array lives on (CPU model, or GPU name)."
deviceinfo(::Array) = string("CPU (", strip(Sys.cpu_info()[1].model), ", ",
                             Threads.nthreads(), Threads.nthreads() == 1 ? " thread)" : " threads)")

_timestamp() = Libc.strftime("%Y-%m-%d %H:%M:%S", time())

function _logstart(sim::Simulation{T}, duration, solver) where {T}
    sim.verbose || return
    msh = sim.world.mesh
    stt = sim.world.current
    sv = solver === :tsit5 ? "tsit5 (abstol $(sim.abstol), reltol $(sim.reltol))" :
         sim.fixdt > 0 ? "$solver (fixdt $(sim.fixdt) s)" : "$solver (maxerr $(sim.maxerr))"
    extras = String[]
    stt === nothing || push!(extras, "current J = $(stt.J) A/m²" *
        (stt.zhangli ? ", Zhang-Li" : "") * (stt.slonczewski ? ", Slonczewski" : ""))
    sim.temperature > 0 && push!(extras, "T = $(sim.temperature) K")
    println(sim.log, "[JuliaMag] ", _timestamp(), "  run ", @sprintf("%.4g", duration),
            " s from t = ", @sprintf("%.6g", sim.t), " s")
    println(sim.log, "[JuliaMag]   device: ", deviceinfo(sim.m), ", precision ", T)
    println(sim.log, "[JuliaMag]   mesh ", join(msh.size, "×"), ", solver ", sv,
            isempty(extras) ? "" : ", " * join(extras, ", "))
    flush(sim.log)
end

function _logend(sim::Simulation, wall, steps, undone, evals)
    sim.verbose || return
    println(sim.log, "[JuliaMag] ", _timestamp(), "  done at t = ", @sprintf("%.6g", sim.t),
            " s: ", steps, " steps (", undone, " undone), ", evals, " evaluations, wall ",
            @sprintf("%.1f", wall), " s")
    flush(sim.log)
end

# --- Running -----------------------------------------------------------------

"""
    relax!(sim; kwargs...)

Relax the magnetization to equilibrium with the energy minimizer, in place
(mumax3 `Minimize`). Keyword arguments go to [`Minimizer`](@ref). See [`relaxllg!`](@ref) for
mumax3's `Relax`.
"""
function relax!(sim::Simulation; kwargs...)
    _drive!(sim.world, sim.t)
    mn = Minimizer(sim.world, sim.m; kwargs...)
    minimize!(mn)
    copyto!(sim.m, mn.m)
    return sim
end

"""
    run!(sim, duration; solver=sim.solver, kwargs...)

Integrate the LLG (with the spin torques of [`setcurrent!`](@ref) and the thermal
field of [`settemperature!`](@ref)) for `duration` seconds (mumax3 `Run`). The
solver is chosen by name, `:rk45` by default; other keywords (`maxerr`, `fixdt`,
`mindt`, `maxdt`, `headroom`, `abstol`, `reltol`) override the simulation's
settings for this run only (see [`setsolver!`](@ref)).

Output: table rows and OVF snapshots as scheduled by [`tableautosave!`](@ref) /
[`autosave!`](@ref), with mumax3's rule. If the table has no autosave interval, a
row is saved at the start and at the end of the run. The wall-clock start and end
time, the device and the solver are printed (`verbose = false` to silence).
"""
function run!(sim::Simulation, duration::Real; solver = nothing, kwargs...)
    duration >= 0 || throw(ArgumentError("duration must be ≥ 0"))
    # Only tolerances given (the pre-2026-11 call run!(sim, t; abstol, reltol)): Tsit5.
    if solver === nothing
        solver = !isempty(kwargs) && all(k -> k in (:abstol, :reltol), keys(kwargs)) ?
                 :tsit5 : sim.solver
    end
    if !isempty(kwargs) || solver !== sim.solver
        saved = (sim.solver, sim.maxerr, sim.fixdt, sim.mindt, sim.maxdt, sim.headroom,
                 sim.abstol, sim.reltol, sim.precess)
        try
            setsolver!(sim, solver; kwargs...)
            return _run!(sim, duration)
        finally
            sim.solver, sim.maxerr, sim.fixdt, sim.mindt, sim.maxdt, sim.headroom,
                sim.abstol, sim.reltol, sim.precess = saved
            sim._stepper = nothing
        end
    end
    return _run!(sim, duration)
end

function _run!(sim::Simulation, duration::Real)
    t_end = sim.t + Float64(duration)
    solver = sim.solver
    _logstart(sim, duration, solver)
    wall0 = time()
    s0, u0, e0 = sim.nsteps, sim.nundone, sim.nevals
    legacy = sim.table.autosave == 0           # no table schedule: rows at start and end
    legacy && tablesave!(sim)
    _dooutput!(sim)                            # every autosave that is due (OVF, table)
    if solver === :tsit5
        sim.precess || throw(ArgumentError(":tsit5 does not support precess = false; use one of $(join(SOLVERS, ", "))"))
        _run_tsit5!(sim, t_end)
    else
        _stepping!(sim, ts -> advanceto!(ts, t_end; onstep = ts -> _stepoutput!(sim, ts)))
    end
    legacy && tablesave!(sim)
    _logend(sim, time() - wall0, sim.nsteps - s0, sim.nundone - u0, sim.nevals - e0)
    return sim
end

# The time, the step and the counters of the stepper, taken to the simulation.
function _sync!(sim::Simulation, ts::TimeStepper)
    sim.t = ts.t
    sim.dt = ts.dt
    sim.nsteps += ts.nsteps; sim.nundone += ts.nundone; sim.nevals += ts.nevals
    ts.nsteps = ts.nundone = ts.nevals = 0
    sim.lasterr = ts.lasterr; sim.peakerr = max(sim.peakerr, ts.peakerr)
    return sim
end

# After an accepted step: the state of the stepper and the output that is due.
function _stepoutput!(sim::Simulation, ts::TimeStepper)
    _sync!(sim, ts)
    # a fixed step keeps its error on the device: check it (NaN) before
    # writing output, not only at the end of the run
    _outputdue(sim) && (_syncerr!(ts); _sync!(sim, ts))
    _dooutput!(sim)
end

# `f(ts)` on the stepper of the simulation; the time, step and counters are taken back also when
# it stops with an error.
function _stepping!(sim::Simulation, f)
    ts = _stepper!(sim)
    try
        f(ts)
    finally
        _sync!(sim, ts)
    end
    return sim
end

"""
    runwhile!(sim, cond)

Integrate while `cond(sim)` is true, evaluated before every step (mumax3 `RunWhile`), with the
scheduled output. `cond` sees the current `sim.t`, `sim.m` and counters.
"""
function runwhile!(sim::Simulation, cond)
    sim.solver === :tsit5 && throw(ArgumentError("runwhile!: not available with :tsit5"))
    _dooutput!(sim)
    wall0 = time()
    s0, u0, e0 = sim.nsteps, sim.nundone, sim.nevals
    sim.verbose && println(sim.log, "[JuliaMag] ", _timestamp(), "  run while a condition holds, from t = ", sim.t, " s")
    while cond(sim)
        _stepping!(sim, ts -> (ts.fsal = false; _stepwhile!(sim, ts, cond)))
    end
    _logend(sim, time() - wall0, sim.nsteps - s0, sim.nundone - u0, sim.nevals - e0)
    return sim
end
function _stepwhile!(sim::Simulation, ts::TimeStepper, cond)
    ts.alarm = Inf
    while true
        ok = step!(ts)                         # an undone step counts for the condition too (mumax3 NEval)
        _syncerr!(ts)                          # (a fixed step: the errors, for a condition on them)
        ok ? _stepoutput!(sim, ts) : _sync!(sim, ts)
        cond(sim) || break
    end
    _syncerr!(ts)
end

"""
    steps!(sim, n)

Take `n` accepted steps (mumax3 `Steps`), with the scheduled output.
"""
function steps!(sim::Simulation, n::Integer)
    stop = sim.nsteps + n
    return runwhile!(sim, s -> s.nsteps < stop)
end

"""
    relaxllg!(sim; threshold=-1) -> Bool

mumax3's `Relax`: integrate the relaxation torque −γ m×(m×B), plus the spin torques of a current,
with the adaptive Bogacki–Shampine solver, without precession, thermal field or advance of
time. First while the energy decreases (checked every 3 steps), then while the torque does,
tightening the error tolerance by √2 down to 1e-9; with `threshold` > 0 [T], until the largest
torque is below it at every tolerance. Returns whether it converged.
"""
function relaxllg!(sim::Simulation; threshold::Real = -1.0)
    _drive!(sim.world, sim.t)
    w = sim.world
    ts = TimeStepper(w, sim.m; solver = :rk23, maxerr = sim.maxerr, mindt = sim.mindt, maxdt = sim.maxdt,
                     headroom = sim.headroom, dt = sim.dt, t = sim.t, precess = false,
                     timed = false)            # the time stands still
    function steps3()
        _dooutput!(sim)
        n = 0
        while n < 3
            step!(ts) && (n += 1)
        end
        ts.t = sim.t                           # the time stands still
        _sync!(sim, ts)
    end
    energy() = energy!(w._Bbuf, sim.m, w)
    try
        steps3(); E0 = energy()
        steps3(); E1 = energy()
        while E1 < E0
            steps3(); E0, E1 = E1, energy()
        end
        # now the torque, less noisy than the energy; k[1] is the torque at the state (FSAL)
        avg() = Float64(_gdot(ts.k[1], ts.k[1]))
        if threshold > 0
            while true
                while _maxvecnorm(ts.k[1]) > threshold * γLL
                    steps3()
                end
                ts.maxerr /= sqrt(2)
                ts.maxerr < 1e-9 && break
            end
        else
            T1 = avg()
            while ts.maxerr > 1e-9
                ts.maxerr /= sqrt(2)
                steps3(); T0, T1 = T1, avg()
                while T1 < T0
                    steps3(); T0, T1 = T1, avg()
                end
            end
        end
    finally
        ts.t = sim.t
        _sync!(sim, ts)
        sim._stepper = nothing
    end
    return true
end

# The cached stepper for the current settings (buffers reused between runs).
function _stepper!(sim::Simulation)
    ts = sim._stepper
    if !(ts isa TimeStepper) || ts.m !== sim.m || ts.world !== sim.world ||
       ts.solver !== sim.solver || (ts.Bth === nothing) != (sim.temperature == 0) || ts.precess != sim.precess
        ts = TimeStepper(sim.world, sim.m; solver = sim.solver, maxerr = sim.maxerr,
                         fixdt = sim.fixdt, mindt = sim.mindt, maxdt = sim.maxdt,
                         headroom = sim.headroom, dt = sim.dt,
                         temperature = sim.temperature, rng = sim.rng, t = sim.t, precess = sim.precess)
        sim._stepper = ts
    end
    ts.t = sim.t; ts.dt = sim.fixdt > 0 ? sim.fixdt : sim.dt
    ts.maxerr = sim.maxerr; ts.fixdt = sim.fixdt; ts.mindt = sim.mindt
    ts.maxdt = sim.maxdt; ts.headroom = sim.headroom; ts.temperature = sim.temperature
    ts.rng = sim.rng
    return ts
end

# Tsit5 (OrdinaryDiffEq): integrate to each scheduled output time exactly.
function _run_tsit5!(sim::Simulation, t_end::Float64)
    sim.temperature > 0 &&
        throw(ArgumentError(":tsit5 cannot integrate a finite temperature; use a fixed-step solver"))
    t0 = sim.t
    it = Integrator(sim.world, sim.m; abstol = sim.abstol, reltol = sim.reltol,
                    tend = t_end - t0)
    tbl = sim.table
    scheduled = tbl.autosave > 0 || !isempty(sim.ovf)
    # next scheduled time strictly after t for an autosave (period, start, count)
    function nextsave(period, start, count, t)
        n = count + 1
        while start + n * period <= t + 1e-9 * period
            n += 1
        end
        return start + n * period
    end
    while sim.t < t_end
        target = t_end
        if scheduled
            tbl.autosave > 0 && (target = min(target, nextsave(tbl.autosave, tbl.start, tbl.count, sim.t)))
            for a in sim.ovf
                target = min(target, nextsave(a[2], a[3], a[4], sim.t))
            end
        end
        advance!(it, target - (t0 + currenttime(it)))
        sim.t = target
        copyto!(sim.m, state(it))
        scheduled && _dooutput!(sim)
    end
    st = it.integrator.stats
    sim.nsteps += st.naccept; sim.nundone += st.nreject; sim.nevals += st.nf
    return sim
end

"""
    runcurrent!(sim, J, duration; every=20e-12, dt=5e-14)

Integrate the LLG with an in-plane Zhang-Li spin-transfer current `J = (Jx,Jy,Jz)`
[A/m²] for `duration` seconds, appending a table row every `every` seconds. Uses
a fixed-step fourth-order Runge-Kutta with the current torque added to the LLG
torque. The material's `pol` and `xi` set the polarization and non-adiabaticity.

Kept for existing scripts; the general route is `setcurrent!(sim, J)` with
[`run!`](@ref), which works with every solver (adaptive ones included) and with the
Slonczewski torque. This function ignores a current set with `setcurrent!`.
"""
function runcurrent!(sim::Simulation{T}, J, duration::Real;
                     every = 20e-12, dt = 5e-14) where {T}
    _nodrive(sim.world, "runcurrent!")
    sim.precess || throw(ArgumentError("runcurrent! does not support precess = false"))
    world, msh, mat = sim.world, sim.world.mesh, sim.world.material
    tbl = sim.table
    Jt = NTuple{3,T}(J)
    nchunks = max(1, round(Int, duration / every))
    substeps = max(1, round(Int, every / dt))
    tablesave!(sim)
    for _ in 1:nchunks
        _rk4_current!(sim.m, world, msh, mat, Jt, T(dt), substeps)
        sim.t += Float64(every)
        tablesave!(sim)
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

"""
Average magnetization ⟨mx,my,mz⟩ of the current state over the magnet — mumax3's
table `m` ([`average_magnet`](@ref): fill-weighted, empty cells excluded). Equal to
`average(sim.m)` when every cell is magnetic.
"""
average(sim::Simulation) = average_magnet(sim.m, sim.world.material)

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation{T}) where {T}
    print(io, "Simulation (", T, ", ", deviceinfo(sim.m), ")\n")
    show(io, MIME"text/plain"(), sim.world.mesh); println(io)
    print(io, "  t = ", sim.t * 1e9, " ns, solver ", sim.solver,
          sim.fixdt > 0 ? " (fixdt $(sim.fixdt) s)" : " (maxerr $(sim.maxerr))",
          ", table columns: ", join([q.name for q in sim.table.quantities], ", "))
end

"""
    eigenmodes(sim; kwargs...)

The linear eigenmodes of the state of a (relaxed) simulation; see [`eigenmodes`](@ref).
"""
eigenmodes(sim::Simulation; kwargs...) = eigenmodes(sim.world, sim.m; kwargs...)
