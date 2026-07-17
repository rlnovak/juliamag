# JuliaMag GUI application.
#
# Opens a dedicated Qt6/QML window (via QML.jl) with two embedded Makie viewports
# (via QMLMakie/GLMakie): a time-series panel (a chosen scalar vs. time) and a
# field panel (an in-plane (mx,my) quiver over an mz — or other — colour map of
# the current state). The control column builds a Simulation from the inputs or
# from a dropped Julia script, and drives it with Run/Pause/Step/Stop. All heavy
# work stays in Julia; QML is only the UI.
#
# Run:  julia --project=gui gui/app.jl   (after gui/setup.jl once)

ENV["QSG_RENDER_LOOP"] = "basic"   # recommended for embedded Makie/JuliaCanvas

using JuliaMag
using QML
using QMLMakie
using GLMakie
using Makie
using Observables
using Printf

# Makie/GeometryBasics also export Mesh, Cylinder, Rect, Circle, Cone; import the
# JuliaMag geometry names explicitly so an unqualified `Mesh`/`Cylinder` here is
# unambiguously JuliaMag's (an explicit import wins over the `using` glob).
using JuliaMag: Mesh, Cylinder, Cuboid, Ellipsoid

# --- Shared state ----------------------------------------------------------

const status   = Observable("Ready.")
const sim_ref  = Ref{Union{Nothing,Simulation}}(nothing)
const stopflag = Ref(false)
const pauseflag = Ref(false)
const running  = Ref(false)
const materials = String[m for m in materialnames()]

# Field-panel display options (set from QML dropdowns/checkbox).
const field_on      = Ref(true)       # draw the field panel at all
const field_quantity = Ref("mz")      # scalar shown as the colour map
const field_colormap = Ref("balance") # Makie colormap name
const field_layer   = Ref(1)          # z-layer to show
const field_every   = Ref(4)          # redraw the field every N chunks (throttle)

# Time-series option (which scalar is on the y-axis).
const ycurve = Ref("mx")

# --- Live plot data --------------------------------------------------------

const ts   = Observable(Float64[])    # time [s]
const ys   = Observable(Float64[])    # the selected scalar vs. time
const ylabel = Observable("mx")

# Field-panel observables. The quiver arrays start with a single degenerate arrow
# (not empty): arrows2d! cannot build its mesh from empty vectors.
const fieldscalar = Observable(zeros(Float32, 2, 2))   # colour-map array
const qx = Observable(Float32[1]); const qy = Observable(Float32[1])
const qu = Observable(Float32[0]); const qv = Observable(Float32[0])
const field_title = Observable("mz")
const field_cmap  = Observable(:balance)   # colormap symbol for the field panel

# --- Makie figures (one per QML viewport) ----------------------------------

function build_curve_figure()
    fig = Figure(size = (560, 360))
    ax  = Axis(fig[1, 1], xlabel = "t (ns)", ylabel = "value")
    ax.ylabel = "mx"
    lines!(ax, lift(t -> t .* 1e9, ts), ys, color = :dodgerblue)
    on(ylabel) do lab; ax.ylabel = lab; end
    return fig
end

function build_field_figure()
    fig = Figure(size = (560, 420))
    ax  = Axis(fig[1, 1], aspect = DataAspect(), xlabel = "x (nm)", ylabel = "y (nm)")
    hm  = heatmap!(ax, fieldscalar, colormap = field_cmap, colorrange = (-1, 1))
    Colorbar(fig[1, 2], hm)
    arrows2d!(ax, qx, qy, qu, qv; color = :black)
    on(field_title) do ttl; ax.title = ttl; end
    return fig
end

const plot_curve = build_curve_figure()
const plot_field = build_field_figure()

# --- Building a simulation from GUI inputs ---------------------------------

# Map a geometry choice + size to region params (region 0 empty background,
# region 1 = the shape, keeping the chosen material) or a plain single-region
# material for "Full mesh". The size is a diameter/side in nm; the shape is made
# tall along z so it spans every layer.
function _apply_geometry(mesh, mat, geom, sizestr)
    geom = String(geom)
    geom == "Full mesh" && return mat
    s = parse(Float64, sizestr) * 1e-9
    tall = 10 * mesh.cellsize[3] * mesh.size[3] + s      # spans all z layers
    rp = RegionParams(mesh, mat)
    setregion!(rp, 0; Msat = 0.0)                        # empty background
    shape = geom == "Cylinder"  ? Cylinder(s, tall) :
            geom == "Rectangle" ? Cuboid(s, s, tall)   :
                                  Ellipsoid(s, s, tall)  # "Ellipse"
    defregion!(rp, 1, shape)                             # region 1 keeps `mat`
    return rp
end

function make_sim(nx, ny, nz, cellstr, matname, alphastr, statename, geom, sizestr, pbcx)
    c = parse(Float64, cellstr) * 1e-9
    mesh = Mesh((Int(nx), Int(ny), Int(nz)), (c, c, c); pbc = (pbcx ? 1 : 0, 0, 0))
    mat  = material(String(matname); alpha = parse(Float64, alphastr))
    params = _apply_geometry(mesh, mat, geom, sizestr)
    sim = Simulation(mesh, params; demag = true)

    state = String(statename)
    cfg = if state == "Vortex"
        VortexConfig(mesh; circ = 1, pol = 1)
    elseif state == "Néel skyrmion"
        NeelSkyrmionConfig(mesh; charge = 1, pol = -1)
    elseif state == "Bloch skyrmion"
        BlochSkyrmionConfig(mesh; charge = 1, pol = -1)
    elseif state == "Random"
        RandomConfig()
    else
        UniformConfig(1, 0, 0)
    end
    setmag!(sim, cfg)
    return sim
end

# --- Extracting display data from the current state ------------------------

# The scalar shown as the colour map, for the selected quantity and layer.
function _scalar_field(sim, quantity, layer)
    m = sim.m
    Nx, Ny, Nz = sim.world.mesh.size
    k = clamp(layer, 1, Nz)
    comp = quantity == "mx" ? 1 : quantity == "my" ? 2 : 3
    img = Array{Float32}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        img[i, j] = Float32(m[comp, i, j, k])
    end
    return img
end

# Refresh the field panel (colour map + subsampled in-plane quiver).
function update_field!(sim)
    field_on[] || return
    Nx, Ny, Nz = sim.world.mesh.size
    cx, cy, _ = sim.world.mesh.cellsize
    k = clamp(field_layer[], 1, Nz)
    fieldscalar[] = _scalar_field(sim, field_quantity[], k)
    field_title[] = @sprintf("%s (layer %d)", field_quantity[], k)

    # In-plane quiver, subsampled to ~20 arrows across x.
    st = max(1, Nx ÷ 20)
    xs = Float32[]; ys_ = Float32[]; us = Float32[]; vs = Float32[]
    scale = Float32(0.8 * st)
    @inbounds for j in 1:st:Ny, i in 1:st:Nx
        push!(xs, Float32(i)); push!(ys_, Float32(j))
        push!(us, Float32(sim.m[1, i, j, k]) * scale)
        push!(vs, Float32(sim.m[2, i, j, k]) * scale)
    end
    qx[] = xs; qy[] = ys_; qu[] = us; qv[] = vs
end

# The scalar plotted on the time-series y-axis for the current state.
function _ycurve_value(sim)
    q = ycurve[]
    if q == "mx"; return average(sim.m)[1]
    elseif q == "my"; return average(sim.m)[2]
    elseif q == "mz"; return average(sim.m)[3]
    elseif q == "vortex x"; return vortexcore(sim.m, sim.world.mesh)[1]
    elseif q == "vortex y"; return vortexcore(sim.m, sim.world.mesh)[2]
    elseif q == "skyrmion x"; return skyrmionpos(sim.m, sim.world.mesh)[1]
    elseif q == "skyrmion y"; return skyrmionpos(sim.m, sim.world.mesh)[2]
    elseif q == "topological charge"; return topologicalcharge(sim.m, sim.world.mesh)
    else; return average(sim.m)[1]
    end
end

_push_point!(sim) = (push!(ts[], sim.t); push!(ys[], _ycurve_value(sim));
                     notify(ts); notify(ys))

# --- QML-callable functions ------------------------------------------------

function gui_build(nx, ny, nz, cellstr, matname, alphastr, statename, geom, sizestr, pbcx)
    status[] = "Building simulation…"
    try
        sim = make_sim(nx, ny, nz, cellstr, matname, alphastr, statename, geom, sizestr, pbcx)
        sim_ref[] = sim
        ts[] = Float64[]; ys[] = Float64[]
        update_field!(sim)
        status[] = "Built. Relax or Run."
    catch e
        status[] = "Build error: " * sprint(showerror, e)
    end
    nothing
end

function gui_relax()
    sim = sim_ref[]
    sim === nothing && (status[] = "Build first."; return nothing)
    status[] = "Relaxing…"
    relax!(sim; stopdm = 1e-6)
    update_field!(sim)
    mx, my, mz = average(sim)
    status[] = @sprintf("Relaxed. ⟨m⟩ = (%.3f, %.3f, %.3f)", mx, my, mz)
    nothing
end

# Load a dropped Julia script: include it and adopt the `sim` it defines.
function gui_loadscript(path)
    p = String(path)
    startswith(p, "file://") && (p = p[8:end])
    Sys.iswindows() && startswith(p, "/") && (p = p[2:end])   # file:///C:/… → C:/…
    status[] = "Loading " * basename(p) * "…"
    try
        mod = Module(:DroppedScript)
        Core.eval(mod, :(using JuliaMag))
        Base.include(mod, p)
        if isdefined(mod, :sim) && getfield(mod, :sim) isa Simulation
            sim_ref[] = getfield(mod, :sim)
            ts[] = Float64[]; ys[] = Float64[]
            update_field!(sim_ref[])
            status[] = "Loaded sim from " * basename(p)
        else
            status[] = "Script ran but defined no `sim::Simulation`."
        end
    catch e
        status[] = "Script error: " * sprint(showerror, e)
    end
    nothing
end

# Run in chunks of `savedt`, streaming the curve and (throttled) the field.
function gui_run(durstr, savedtstr, bxstr, bystr, bzstr)
    sim = sim_ref[]
    sim === nothing && (status[] = "Build first."; return nothing)
    running[] && (status[] = "Already running."; return nothing)
    duration = parse(Float64, durstr) * 1e-9
    savedt   = parse(Float64, savedtstr) * 1e-12
    Bext = (parse(Float64, bxstr), parse(Float64, bystr), parse(Float64, bzstr)) .* 1e-3
    JuliaMag.setexternalfield!(sim.world, Bext)

    stopflag[] = false; pauseflag[] = false; running[] = true
    nchunks = max(1, round(Int, duration / savedt))
    it = JuliaMag.Integrator(sim.world, sim.m; tend = sim.t + duration)
    status[] = "Running…"
    _push_point!(sim)
    @async begin
        i = 0
        while i < nchunks
            stopflag[] && break
            if pauseflag[]
                status[] = "Paused."
                while pauseflag[] && !stopflag[]; sleep(0.05); end
                stopflag[] && break
                status[] = "Running…"
            end
            JuliaMag.advance!(it, savedt)
            sim.t = JuliaMag.currenttime(it)
            copyto!(sim.m, JuliaMag.state(it))
            _push_point!(sim)
            (i % field_every[] == 0) && update_field!(sim)
            i += 1
            yield()
        end
        update_field!(sim)
        running[] = false
        mx, my, mz = average(sim.m)
        status[] = @sprintf("Done. t = %.3f ns, ⟨m⟩ = (%.3f, %.3f, %.3f)",
                            sim.t * 1e9, mx, my, mz)
    end
    nothing
end

# Advance exactly one chunk of `savedt` (when not in a running loop).
function gui_step(savedtstr, bxstr, bystr, bzstr)
    sim = sim_ref[]
    sim === nothing && (status[] = "Build first."; return nothing)
    running[] && (status[] = "Pause before stepping."; return nothing)
    savedt = parse(Float64, savedtstr) * 1e-12
    Bext = (parse(Float64, bxstr), parse(Float64, bystr), parse(Float64, bzstr)) .* 1e-3
    JuliaMag.setexternalfield!(sim.world, Bext)
    it = JuliaMag.Integrator(sim.world, sim.m; tend = sim.t + savedt)
    JuliaMag.advance!(it, savedt)
    sim.t = JuliaMag.currenttime(it)
    copyto!(sim.m, JuliaMag.state(it))
    _push_point!(sim)
    update_field!(sim)
    status[] = @sprintf("Stepped to t = %.4f ns", sim.t * 1e9)
    nothing
end

gui_pause()  = (pauseflag[] = !pauseflag[];
                status[] = pauseflag[] ? "Pausing…" : "Resuming…"; nothing)
gui_stop()   = (stopflag[] = true; pauseflag[] = false;
                status[] = "Stopping…"; nothing)

# Display-option setters from QML.
function gui_set_ycurve(name)
    ycurve[] = String(name); ylabel[] = String(name)
    sim = sim_ref[]
    # Recompute the whole curve for the new quantity is not stored; just relabel
    # and let subsequent points use it. Reset so the axis is clean.
    ts[] = Float64[]; ys[] = Float64[]
    sim !== nothing && _push_point!(sim)
    nothing
end
function gui_set_field(quantity, colormap, layer, on)
    field_quantity[] = String(quantity)
    field_colormap[] = String(colormap)
    field_cmap[]     = Symbol(String(colormap))   # live-updates the heatmap
    field_layer[]    = Int(layer)
    field_on[]       = Bool(on)
    sim = sim_ref[]
    sim !== nothing && field_on[] && update_field!(sim)
    nothing
end

@qmlfunction gui_build gui_relax gui_loadscript gui_run gui_step gui_pause gui_stop gui_set_ycurve gui_set_field

# --- Launch ----------------------------------------------------------------
# The two Makie Figures and the observables are exposed to QML as context
# properties. (The QML module for the Julia-provided components — including
# QMLMakie's MakieViewport — is `jlqml` in QML.jl ≥ 0.11, imported in main.qml.)

loadqml(joinpath(@__DIR__, "main.qml");
        status = status, materials = materials,
        plot_curve = plot_curve, plot_field = plot_field)
exec()
