# JuliaMag GUI application.
#
# Opens a dedicated Qt6/QML window (via QML.jl) with an embedded Makie viewport
# (via QMLMakie/GLMakie). The control column builds a Simulation from the user's
# inputs; Run integrates it, streaming ⟨m⟩(t) into a live plot and the
# magnetization into a heatmap. All heavy work stays in Julia; QML is only the UI.
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

# --- Shared state ----------------------------------------------------------

const status = Observable("Ready.")
const sim_ref = Ref{Union{Nothing,Simulation}}(nothing)
const stopflag = Ref(false)
const materials = String[m for m in materialnames()]

# Live plot data.
const ts  = Observable(Float64[])
const mxs = Observable(Float64[])
const mys = Observable(Float64[])
const mzs = Observable(Float64[])
const mzimage = Observable(zeros(Float32, 2, 2))   # magnetization mz heatmap

# --- Makie figure ----------------------------------------------------------

function build_figure()
    fig = Figure(size = (760, 700))
    ax1 = Axis(fig[1, 1], title = "⟨m⟩(t)", xlabel = "t (ns)", ylabel = "⟨m⟩")
    lines!(ax1, lift(t -> t .* 1e9, ts), mxs, color = :red,   label = "mx")
    lines!(ax1, lift(t -> t .* 1e9, ts), mys, color = :green, label = "my")
    lines!(ax1, lift(t -> t .* 1e9, ts), mzs, color = :blue,  label = "mz")
    axislegend(ax1; position = :rb)

    ax2 = Axis(fig[2, 1], title = "mz (top layer)", aspect = DataAspect())
    heatmap!(ax2, mzimage, colormap = :balance, colorrange = (-1, 1))
    return fig
end

const plot = build_figure()

# --- Building a simulation from GUI inputs ---------------------------------

function make_sim(nx, ny, nz, cellstr, matname, alphastr, statename)
    c = parse(Float64, cellstr) * 1e-9
    mesh = Mesh((Int(nx), Int(ny), Int(nz)), (c, c, c))
    mat = material(String(matname); alpha = parse(Float64, alphastr))
    sim = Simulation(mesh, mat; demag = true)

    state = String(statename)
    cfg = if state == "Vortex"
        VortexConfig(mesh; circ = 1, pol = 1)
    elseif state == "Néel skyrmion"
        NeelSkyrmionConfig(mesh; charge = 1, pol = 1)
    elseif state == "Bloch skyrmion"
        BlochSkyrmionConfig(mesh; charge = 1, pol = 1)
    elseif state == "Random"
        RandomConfig()
    else
        UniformConfig(1, 0, 0)
    end
    setmag!(sim, cfg)
    return sim
end

# Push the current magnetization's top-layer mz into the heatmap observable.
function update_image!(sim)
    m = sim.m
    Nx, Ny, Nz = mesh(sim).size
    img = Array{Float32}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        img[i, j] = Float32(m[3, i, j, Nz])
    end
    mzimage[] = img
end

# --- QML-callable functions ------------------------------------------------

function gui_relax(nx, ny, nz, cellstr, matname, alphastr, statename)
    status[] = "Building simulation…"
    sim = make_sim(nx, ny, nz, cellstr, matname, alphastr, statename)
    sim_ref[] = sim
    status[] = "Relaxing…"
    relax!(sim; stopdm = 1e-6)
    update_image!(sim)
    mx, my, mz = average(sim)
    status[] = @sprintf("Relaxed. ⟨m⟩ = (%.3f, %.3f, %.3f)", mx, my, mz)
    nothing
end

function gui_run(durstr, savedtstr, bxstr, bystr, bzstr)
    sim = sim_ref[]
    if sim === nothing
        status[] = "Relax first."
        return nothing
    end
    duration = parse(Float64, durstr) * 1e-9
    savedt = parse(Float64, savedtstr) * 1e-12
    Bext = (parse(Float64, bxstr), parse(Float64, bystr), parse(Float64, bzstr)) .* 1e-3
    JuliaMag.setexternalfield!(sim.world, Bext)

    # Reset live curves.
    ts[] = Float64[]; mxs[] = Float64[]; mys[] = Float64[]; mzs[] = Float64[]
    stopflag[] = false

    # Step in chunks so the UI updates; stop if requested. Drive the integrator
    # directly and collect ⟨m⟩ per chunk (rather than the Simulation table, which
    # would re-save a start row each call).
    nchunks = max(1, round(Int, duration / savedt))
    status[] = "Running…"
    it = JuliaMag.Integrator(sim.world, sim.m; tend = sim.t + duration)
    @async begin
        for _ in 1:nchunks
            stopflag[] && break
            JuliaMag.advance!(it, savedt)
            sim.t = JuliaMag.currenttime(it)
            copyto!(sim.m, JuliaMag.state(it))
            mx, my, mz = average(sim.m)
            push!(ts[], sim.t); push!(mxs[], mx); push!(mys[], my); push!(mzs[], mz)
            notify(ts); notify(mxs); notify(mys); notify(mzs)
            update_image!(sim)
            yield()
        end
        mx, my, mz = average(sim.m)
        status[] = @sprintf("Done. t = %.3f ns, ⟨m⟩ = (%.3f, %.3f, %.3f)",
                            sim.t * 1e9, mx, my, mz)
    end
    nothing
end

gui_stop() = (stopflag[] = true; status[] = "Stopping…"; nothing)

@qmlfunction gui_relax gui_run gui_stop

# --- Launch ----------------------------------------------------------------
# The Makie Figure and the observables/arrays are exposed to QML as context
# properties. (The QML module for the Julia-provided components — including
# QMLMakie's MakieViewport — is `jlqml` in QML.jl ≥ 0.11, imported in main.qml.)

loadqml(joinpath(@__DIR__, "main.qml");
        status = status, materials = materials, plot = plot)
exec()
