# The part of the GUI that needs no window: the jobs that run on a worker thread (build, load a
# script, relax, run, set the field), and the snapshots they publish for the window to draw.
#
# Threads. The window (Qt's event loop) runs on the main thread. Every job that touches the
# simulation runs on one worker task, on another thread, so the window stays responsive. The
# worker never touches Makie or QML: it publishes copies (a snapshot, and the new points of the
# curves) under a lock, and the main thread takes them in `takeupdates!`, called by a QML timer.
# The main thread touches the simulation only through a job. Requests from the window while a
# job runs (pause, stop, a new applied field) are read by the worker between steps.
#
# Cost. During a run the worker publishes at most `rate[]` times per second of wall time:
# ⟨m⟩ (a reduction) and the maximum torque of the last step (no extra evaluation); the layer
# shown at most `srate[]` times per second (only that layer is copied from the GPU). The total
# energy needs a field evaluation, on the host for a GPU simulation, so it is computed at most
# `erate[]` times per second (1 on a GPU), and only when the energy curve is shown.

using JuliaMag
using JuliaMag: Mesh, Cylinder, Cuboid, Ellipsoid
using Printf
using Colors

# --- What the worker publishes ---------------------------------------------

mutable struct Snapshot
    phase::String                    # "idle", "building", "loading", "relaxing", "running", "paused"
    t::Float64; nsteps::Int; nundone::Int; dt::Float64
    torque::Float64                  # maxTorque [T] (NaN: not known)
    m::NTuple{3,Float64}
    E::Float64                       # total energy [J] (NaN: not computed)
    Bext::NTuple{3,Float64}
    wall::Float64                    # wall time of the job [s]
    srate::Float64                   # accepted steps per second of wall time, pauses excluded
    device::String
    mesh::Union{Nothing,NTuple{6,Float64}}   # x0, x1, y0, y1 [nm], Nz, layer shown
    slice::Union{Nothing,Array{Float32,3}}   # (3, Nx, Ny), the layer shown; 0 in empty cells
    message::String
end
Snapshot() = Snapshot("idle", 0.0, 0, 0, 0.0, NaN, (NaN, NaN, NaN), NaN, (0.0, 0.0, 0.0), 0.0, NaN, "",
                      nothing, nothing, "Ready.")

const lk = ReentrantLock()                    # guards everything below that both threads use
const snap = Ref(Snapshot())
const mpoints = Vector{NTuple{4,Float64}}()   # (t, mx, my, mz) not yet drawn
const epoints = Vector{NTuple{2,Float64}}()   # (t, E) not yet drawn
const newslice = Ref(false)
const reset_curves = Ref(false)
const pending_field = Ref{Any}(nothing)       # (Bx, By, Bz) [T] for the worker to apply

# --- State of the jobs ------------------------------------------------------

const sim_ref = Ref{Any}(nothing)             # set by the worker only
const job = Ref{Union{Nothing,Task}}(nothing) # set by the main thread only
const stopflag = Threads.Atomic{Bool}(false)
const pauseflag = Threads.Atomic{Bool}(false)
const layer = Ref(1)                          # the layer shown (an Int: atomic in practice)
const want_energy = Ref(true)
const rate = Ref(20.0)                        # publications per second of wall time
const srate = Ref(10.0)                       # slices per second of wall time
const erate = Ref(4.0)                        # energies per second of wall time (CPU)
const steps0 = Ref(0)                         # sim.nsteps when the job started (worker)
const paused = Ref(0.0)                       # wall time spent paused in this job (worker)

busy() = (j = job[]; j !== nothing && !istaskdone(j))

# Wait on the worker thread. Not `sleep`: Julia's event loop (timers, I/O) is run by the main
# thread, which sits in Qt's event loop, so a timer on the worker may never fire. Not one long
# `Libc.systemsleep` either: a thread blocked in C never reaches a safepoint, and a garbage
# collection asked for by the main thread would wait for it. Short naps with a safepoint between.
function nap(seconds)
    t1 = time() + seconds
    while time() < t1
        Libc.systemsleep(0.005)
        GC.safepoint()
    end
end

# --- Building a simulation from the inputs of the window -------------------

# Region 0 empty, region 1 the shape (diameter or side in nm), tall enough to cross every layer.
function _apply_geometry(mesh, mat, geom, s)
    geom == "Full mesh" && return mat
    tall = 10 * mesh.cellsize[3] * mesh.size[3] + s
    rp = RegionParams(mesh, mat)
    setregion!(rp, 0; Msat = 0.0)
    shape = geom == "Cylinder"  ? Cylinder(s, tall) :
            geom == "Rectangle" ? Cuboid(s, s, tall) :
                                  Ellipsoid(s, s / 2, tall)      # "Ellipse 2:1", long axis along x
    defregion!(rp, 1, shape)
    return rp
end

function make_sim(nx, ny, nz, cell_nm, matname, alpha, state, geom, size_nm, pbcx)
    c = cell_nm * 1e-9
    mesh = Mesh((nx, ny, nz), (c, c, c); pbc = (pbcx ? 1 : 0, 0, 0))
    mat = material(matname; alpha = alpha)
    sim = Simulation(mesh, _apply_geometry(mesh, mat, geom, size_nm * 1e-9); demag = true, verbose = false)
    cfg = state == "Vortex"         ? VortexConfig(mesh; circ = 1, pol = 1) :
          state == "Néel skyrmion"  ? NeelSkyrmionConfig(mesh; charge = 1, pol = -1) :
          state == "Bloch skyrmion" ? BlochSkyrmionConfig(mesh; charge = 1, pol = -1) :
          state == "Random"         ? RandomConfig() : UniformConfig(1, 0, 0)
    setmag!(sim, cfg)
    return sim
end

# --- Snapshots (worker) ------------------------------------------------------

_maxtorque_now(sim) = q_maxtorque()(sim.world, sim.m, sim.t)[1]
# The maximum torque of the stage of the last accepted step that mumax3 uses for maxTorque.
function _maxtorque_last(sim)
    ts = sim._stepper
    ts isa TimeStepper || return NaN
    τ = lasttorque(ts)
    return isnan(τ) ? NaN : τ / JuliaMag.γLL
end
function _energy(sim)
    # the terms that depend on time at the time of the state, as the table does
    JuliaMag._drive!(sim.world, sim.t); JuliaMag._drive!(sim.hostworld, sim.t)
    JuliaMag.ondevice(sim) ? JuliaMag.totalenergy(Array(sim.m), sim.hostworld) :
                             JuliaMag.totalenergy(sim.m, sim.world)
end

# The layer shown, copied to the host (only that layer leaves the GPU).
function _slice(sim)
    k = clamp(layer[], 1, size(sim.m, 4))
    return Array{Float32}(Array(view(sim.m, :, :, :, k))), k
end

function _meshbox(sim, k)
    msh = sim.world.mesh
    Lx = msh.size[1] * msh.cellsize[1] * 1e9; Ly = msh.size[2] * msh.cellsize[2] * 1e9
    return (-Lx / 2, Lx / 2, -Ly / 2, Ly / 2, Float64(msh.size[3]), Float64(k))
end

"""
    publish!(sim, phase, wall; torque, energy, slice, point, message)

Copy what the window shows from `sim` (worker only), then hand it over under the lock.
`torque = :last` takes the maximum torque of the last step (free), `:eval` computes it. `point`
appends ⟨m⟩ (and E, if computed) to the curves.
"""
function publish!(sim, phase, wall; torque = :last, energy = false, slice = true, point = true,
                  message = nothing)
    m = Tuple(Float64.(average(sim)))
    τ = torque === :eval ? _maxtorque_now(sim) : _maxtorque_last(sim)
    E = energy ? Float64(_energy(sim)) : NaN
    sl, k = slice ? _slice(sim) : (nothing, 0)
    box = slice ? _meshbox(sim, k) : nothing
    B = Tuple(Float64.(sim.world.Bext))
    active = wall - paused[]
    @lock lk begin
        s = snap[]
        s.phase = phase; s.t = sim.t; s.nsteps = sim.nsteps; s.nundone = sim.nundone; s.dt = sim.dt
        isnan(τ) || (s.torque = τ)
        s.m = m; isnan(E) || (s.E = E); s.Bext = B; s.wall = wall
        s.srate = active > 0 && sim.nsteps > steps0[] ? (sim.nsteps - steps0[]) / active : NaN
        sl === nothing || (s.slice = sl; s.mesh = box; newslice[] = true)
        message === nothing || (s.message = message)
        if point
            push!(mpoints, (sim.t, m...))
            isnan(E) || push!(epoints, (sim.t, E))
        end
    end
    return nothing
end

setphase!(phase, msg) = @lock lk (snap[].phase = phase; snap[].message = msg; nothing)
"A message for the window; the phase stays."
setstatus!(msg) = @lock lk (snap[].message = msg; nothing)

"""
    takeupdates!() -> (snapshot copy, new ⟨m⟩ points, new E points, reset curves?)

Called on the main thread: take what the worker published since the last call. The slice of the
copy is `nothing` unless a new one came.
"""
function takeupdates!()
    @lock lk begin
        s = snap[]
        c = Snapshot(s.phase, s.t, s.nsteps, s.nundone, s.dt, s.torque, s.m, s.E, s.Bext, s.wall, s.srate,
                     s.device, s.mesh, newslice[] ? s.slice : nothing, s.message)
        mp = copy(mpoints); empty!(mpoints)
        ep = copy(epoints); empty!(epoints)
        r = reset_curves[]; reset_curves[] = false; newslice[] = false
        return c, mp, ep, r
    end
end

# --- Jobs (started by the main thread, run by the worker) ----------------------

# Apply a field requested by the window (worker). A new field invalidates the stored torque of
# the first-same-as-last solvers.
function _applyfield!(sim)
    B = @lock lk (b = pending_field[]; pending_field[] = nothing; b)
    B === nothing && return false
    JuliaMag.setexternalfield!(sim, B)
    ts = sim._stepper
    ts isa TimeStepper && (ts.fsal = false)
    return true
end

_erate(sim) = JuliaMag.ondevice(sim) ? min(erate[], 1.0) : erate[]

# Run `f()` on a worker thread unless a job is running. Errors end up in the message. A field
# requested during the job is applied at its end.
function startjob(f, name)
    busy() && return false
    stopflag[] = false; pauseflag[] = false
    job[] = Threads.@spawn :default begin
        steps0[] = (s = sim_ref[]; s === nothing ? 0 : s.nsteps); paused[] = 0.0
        try
            f()
        catch e
            showerror(stderr, e, catch_backtrace()); println(stderr)
            setphase!("idle", "$name failed: " * sprint(showerror, e))
        end
        s = sim_ref[]
        if s !== nothing && _applyfield!(s)
            Base.invokelatest(publish!, s, "idle", 0.0; torque = :eval, energy = want_energy[], point = false)
        end
    end
    return true
end

function _adopt!(sim, msg)
    sim_ref[] = sim
    steps0[] = sim.nsteps
    @lock lk begin
        snap[] = Snapshot(); snap[].device = string(eltype(sim.m), ", ", deviceinfo(sim.m))
        empty!(mpoints); empty!(epoints); reset_curves[] = true
    end
    publish!(sim, "idle", 0.0; torque = :eval, energy = want_energy[], message = msg)
end

job_build(args...) = startjob("build") do
    setphase!("building", "Building the simulation…")
    _adopt!(make_sim(args...), "Built. Relax or run.")
end

function job_load(path)
    startjob("loading") do
        setphase!("loading", "Loading " * basename(path) * "…")
        mod = Module(:GUIScript)
        Core.eval(mod, :(using JuliaMag))
        Base.include(mod, path)
        # the script defined new methods and globals: look at them, and call everything that may
        # use them (functions of time, quantities), in the newest world
        s = Base.invokelatest(() -> isdefined(mod, :sim) ? getglobal(mod, :sim) : nothing)
        s isa Simulation || error("the script defined no `sim::Simulation`")
        Base.invokelatest(_adopt!, s, "Loaded `sim` from " * basename(path) * ".")
    end
end

function job_relax()
    sim = sim_ref[]
    sim === nothing && (setstatus!("Build or load a simulation first."); return false)
    startjob("relax") do
        wall0 = time()
        _applyfield!(sim)
        setphase!("relaxing", "Relaxing…")
        relax!(sim)
        publish!(sim, "idle", time() - wall0; torque = :eval, energy = want_energy[],
                 message = @sprintf("Relaxed in %.1f s.", time() - wall0))
    end
end

# A new applied field [T]: in a short job if nothing runs, else between two steps (or at the end
# of a relaxation).
function job_field(B)
    @lock lk (pending_field[] = B)
    sim = sim_ref[]
    sim === nothing && return true
    busy() && return true
    startjob("field") do
        _applyfield!(sim)
        publish!(sim, "idle", 0.0; torque = :eval, energy = want_energy[], point = false,
                 message = @sprintf("B_ext = (%.4g, %.4g, %.4g) mT.", (B .* 1e3)...))
    end
end

# Show another layer: a new slice in a short job if nothing runs, else at the next publication.
function job_layer(k)
    layer[] = k
    sim = sim_ref[]
    (sim === nothing || busy()) && return true
    startjob("layer") do
        publish!(sim, "idle", 0.0; point = false)
    end
end

struct GUIStop <: Exception end

"""
    job_run(duration)

Integrate for `duration` seconds (`Inf`: until Stop) with the solver of the simulation and its
scheduled output (table, OVF), publishing while it runs. The steps end exactly at the end time,
as in `run!`. Pause and Stop act between steps.
"""
function job_run(duration)
    sim = sim_ref[]
    sim === nothing && (setstatus!("Build or load a simulation first."); return false)
    startjob("run") do
        wall0 = time(); tend = sim.t + duration
        last = -Inf; lasts = -Inf; laste = -Inf
        _applyfield!(sim)
        setphase!("running", "Running…")
        publish!(sim, "running", 0.0; torque = :eval, energy = want_energy[])
        # between two steps: requests of the window, and the publications that are due
        function tick(s)
            _applyfield!(s)
            now = time()
            if now - last >= 1 / rate[]
                e = want_energy[] && now - laste >= 1 / _erate(s)
                sl = now - lasts >= 1 / srate[]
                e && (laste = now); sl && (lasts = now)
                publish!(s, "running", now - wall0; energy = e, slice = sl)
                last = now
            end
            if pauseflag[] && !stopflag[]
                p0 = time()
                publish!(s, "paused", p0 - wall0; energy = want_energy[], point = false, message = "Paused.")
                while pauseflag[] && !stopflag[]
                    nap(0.05)
                    _applyfield!(s) && publish!(s, "paused", p0 - wall0; torque = :eval, point = false)
                end
                paused[] += time() - p0
                setphase!("running", "Running…")
            end
            stopflag[] && throw(GUIStop())
            return nothing
        end
        # as run!: without a table schedule, a row at the start and at the end of the run
        legacy = sim.table.autosave == 0 && sim.solver !== :tsit5
        legacy && tablesave!(sim)
        try
            if sim.solver === :tsit5
                # the OrdinaryDiffEq solver has no step hook: run it in pieces
                piece = isfinite(duration) ? duration / 200 : 10e-12
                while sim.t < tend
                    run!(sim, min(piece, tend - sim.t))
                    tick(sim)
                end
            else
                JuliaMag._dooutput!(sim)                 # the output due at the start
                JuliaMag._stepping!(sim, ts -> advanceto!(ts, tend;
                    onstep = ts -> (JuliaMag._stepoutput!(sim, ts); tick(sim))))
            end
        catch e
            e isa GUIStop || rethrow()
            ts = sim._stepper
            ts isa TimeStepper && (JuliaMag._syncerr!(ts); JuliaMag._sync!(sim, ts))
        end
        legacy && tablesave!(sim)
        msg = stopflag[] ? @sprintf("Stopped at t = %.6g ns.", sim.t * 1e9) :
                           @sprintf("Done: t = %.6g ns.", sim.t * 1e9)
        publish!(sim, "idle", time() - wall0; energy = want_energy[], message = msg)
    end
end

# --- Colours and text (main thread) ------------------------------------------

"""
    wheelcolour(mx, my, mz)

mumax3's colour of a direction: the hue is the in-plane angle, the lightness 0.5 + 0.5 m_z
(white along +z, black along −z). Empty cells (m = 0) are grey.
"""
function wheelcolour(mx, my, mz)
    n2 = mx^2 + my^2 + mz^2
    n2 == 0 && return RGB{Float32}(0.75, 0.75, 0.75)
    h = mod(atand(my, mx), 360)
    return RGB{Float32}(convert(RGB, HSL(h, 1.0, 0.5 + 0.5 * mz / sqrt(n2))))
end

wheelimage(sl) = [wheelcolour(sl[1, i, j], sl[2, i, j], sl[3, i, j]) for i in axes(sl, 2), j in axes(sl, 3)]

function componentimage(sl, c)
    out = Array{Float32}(undef, size(sl, 2), size(sl, 3))
    @inbounds for j in axes(sl, 3), i in axes(sl, 2)
        empty = sl[1, i, j] == 0 && sl[2, i, j] == 0 && sl[3, i, j] == 0
        out[i, j] = empty ? NaN32 : sl[c, i, j]
    end
    return out
end

function infotext(s::Snapshot)
    f(x, fmt) = isnan(x) ? "—" : Printf.format(Printf.Format(fmt), x)
    join([
        "state      " * s.phase,
        "t          " * f(s.t * 1e9, "%.6g") * " ns",
        "steps      $(s.nsteps)" * (s.nundone > 0 ? "  ($(s.nundone) undone)" : ""),
        "dt         " * f(s.dt, "%.3e") * " s",
        "maxTorque  " * f(s.torque, "%.4e") * " T",
        "⟨m⟩        " * @sprintf("(%.5f, %.5f, %.5f)", s.m...),
        "E_total    " * f(s.E, "%.6e") * " J",
        "B_ext      " * @sprintf("(%.4g, %.4g, %.4g) mT", (s.Bext .* 1e3)...),
        "wall       " * @sprintf("%.1f s", s.wall) * (isnan(s.srate) ? "" : @sprintf("   (%.0f steps/s)", s.srate)),
        "device     " * s.device,
    ], "\n")
end

"The local path of a file URL from QML (percent-decoded; Windows drives and UNC shares)."
function urlpath(url::AbstractString)
    p = String(url)
    if startswith(p, "file:///")
        p = p[8:end]                                         # "/C:/…" or "/home/…"
        Sys.iswindows() && occursin(r"^/[A-Za-z]:", p) && (p = p[2:end])
    elseif startswith(p, "file://")
        p = "//" * p[8:end]                                  # a UNC share
    end
    io = IOBuffer(); i = 1; b = codeunits(p)
    while i <= length(b)
        if b[i] == UInt8('%') && i + 2 <= length(b)
            write(io, parse(UInt8, String(b[i+1:i+2]); base = 16)); i += 3
        else
            write(io, b[i]); i += 1
        end
    end
    return String(take!(io))
end
