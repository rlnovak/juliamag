# JuliaMag GUI: a Qt6/QML window (QML.jl) with two Makie viewports (QMLMakie/GLMakie).
#
#   julia -t auto --project=gui gui/app.jl        (after gui/setup.jl once)
#
# The left column builds a simulation from a few inputs, or loads a Julia script that defines
# `sim`, and drives it: Relax, Run, Pause, Stop, and the applied field. The right column shows
# ⟨m⟩(t) and the total energy, the state of the run (t, steps, dt, maxTorque, wall time,
# device), and one layer of m as mumax3's colour wheel or as one component.
#
# The jobs run on a worker thread (core.jl); this file only draws. A QML timer calls `gui_poll`
# about 30 times per second on the main thread, which takes what the worker published and
# updates the plots. Start Julia with at least two threads (`-t auto`).

ENV["QSG_RENDER_LOOP"] = "basic"   # recommended for embedded Makie

include(joinpath(@__DIR__, "core.jl"))

using QML
using QMLMakie
using GLMakie
using Makie
using Observables

if Threads.nthreads(:default) < 1 || (Threads.nthreads(:default) == 1 && Threads.threadpool() === :default)
    error("the GUI needs a worker thread besides the main one: start Julia with `-t auto` (or `-t 2`)")
end

const materials = String[m for m in materialnames()]
const props = JuliaPropertyMap("status" => "Ready.", "info" => infotext(Snapshot()), "busy" => false,
                               "running" => false, "paused" => false, "fps" => "", "nz" => 1, "shot" => false)

# --- Figures -------------------------------------------------------------------

const MAXPTS = 20_000                     # points per curve kept on screen (thinned when exceeded)
const mx_pts = Observable(Point2f[]); const my_pts = Observable(Point2f[]); const mz_pts = Observable(Point2f[])
const e_pts = Observable(Point2d[])       # Float64: relative changes of E near equilibrium are small
const e_unit = Ref(0.0)                   # energies are shown in units of 10^k J (0: not chosen yet)

const curves = Figure(size = (620, 420))
const ax_m = Axis(curves[1, 1], ylabel = "⟨m⟩", xticklabelsvisible = false)
const ax_e = Axis(curves[2, 1], xlabel = "t (ns)", ylabel = "E (J)")
lines!(ax_m, mx_pts, color = :red, label = "mx")
lines!(ax_m, my_pts, color = :green, label = "my")
lines!(ax_m, mz_pts, color = :blue, label = "mz")
axislegend(ax_m, position = :rc, framevisible = false, orientation = :horizontal)
lines!(ax_e, e_pts, color = :black)
linkxaxes!(ax_m, ax_e)

const wheel_img = Observable(fill(RGB{Float32}(0.75, 0.75, 0.75), 2, 2))
const comp_img = Observable(fill(NaN32, 2, 2))
const xr = Observable((-1.0, 1.0)); const yr = Observable((-1.0, 1.0))
const view_mode = Ref("wheel")
const last_slice = Ref{Any}(nothing)

const field = Figure(size = (620, 480))
const ax_f = Axis(field[1, 1], aspect = DataAspect(), xlabel = "x (nm)", ylabel = "y (nm)")
const im_wheel = image!(ax_f, xr, yr, wheel_img; interpolate = false)
const im_comp = image!(ax_f, xr, yr, comp_img; colormap = :RdBu, colorrange = (-1, 1),
                       nan_color = RGBf(0.75, 0.75, 0.75), interpolate = false, visible = false)
const cbar = Colorbar(field[1, 2], im_comp; height = Relative(0.6))
cbar.blockscene.visible[] = false

# Append without notifying; past MAXPTS keep every other point, and always the newest.
function _append!(obs, pts)
    isempty(pts) && return false
    v = obs.val
    append!(v, pts)
    length(v) > MAXPTS && (obs.val = push!(v[1:2:end-1], v[end]))
    return true
end

function _drawslice!(sl, box)
    last_slice[] = sl
    box === nothing || (xr.val = (box[1], box[2]); yr.val = (box[3], box[4]))
    if view_mode[] == "wheel"
        wheel_img[] = wheelimage(sl)
    else
        comp_img[] = componentimage(sl, view_mode[] == "mx" ? 1 : view_mode[] == "my" ? 2 : 3)
    end
    box === nothing || (notify(xr); notify(yr); reset_limits!(ax_f))
end

# --- The poll of the window ------------------------------------------------------

const frames = Ref(0); const fps_t0 = Ref(time()); const fps_text = Ref("")
const shown = Dict{String,Any}()          # the values last given to QML

_setprop!(k, v) = get(shown, k, nothing) == v || (props[k] = v; shown[k] = v)

function gui_poll()
    mute[] && return 0
    s, mp, ep, reset = takeupdates!()
    if reset
        mx_pts.val = Point2f[]; my_pts.val = Point2f[]; mz_pts.val = Point2f[]; e_pts.val = Point2d[]
        e_unit[] = 0.0; ax_e.ylabel = "E (J)"
    end
    if !isempty(ep) && e_unit[] == 0
        x = maximum(p -> abs(p[2]), ep)
        k = x > 0 ? floor(Int, log10(x)) : 0
        e_unit[] = 10.0^k; ax_e.ylabel = rich("E (10", superscript(string(k)), " J)")
    end
    a = _append!(mx_pts, [Point2f(p[1] * 1e9, p[2]) for p in mp])
    _append!(my_pts, [Point2f(p[1] * 1e9, p[3]) for p in mp])
    _append!(mz_pts, [Point2f(p[1] * 1e9, p[4]) for p in mp])
    b = _append!(e_pts, [Point2d(p[1] * 1e9, p[2] / e_unit[]) for p in ep])
    redraw = 0                            # which viewports QML must update: 1 curves, 2 the map
    if a || b || reset
        notify(mx_pts); notify(my_pts); notify(mz_pts); notify(e_pts)
        autolimits!(ax_m); autolimits!(ax_e)
        redraw |= 1
    end
    if s.slice !== nothing
        _drawslice!(s.slice, s.mesh)
        s.mesh === nothing || _setprop!("nz", Int(s.mesh[5]))
        redraw |= 2
    end
    _setprop!("info", infotext(s))
    _setprop!("status", s.message)
    _setprop!("busy", busy())
    _setprop!("running", busy() && s.phase in ("running", "paused"))
    _setprop!("paused", s.phase == "paused")
    shoot[] && _setprop!("shot", true)
    frames[] += 1
    if time() - fps_t0[] >= 2
        fps_text[] = @sprintf("%.0f updates/s", frames[] / (time() - fps_t0[]))
        _setprop!("fps", fps_text[])
        frames[] = 0; fps_t0[] = time()
    end
    return redraw
end

# --- Calls from QML ----------------------------------------------------------------

_num(x) = parse(Float64, String(x))
_busymsg(ok) = ok || setstatus!("A job is running.")

function gui_build(nx, ny, nz, cell, mat, alpha, state, geom, sz, pbcx)
    try
        _busymsg(job_build(Int(nx), Int(ny), Int(nz), _num(cell), String(mat), _num(alpha), String(state),
                           String(geom), _num(sz), Bool(pbcx)))
    catch e
        setstatus!("Input error: " * sprint(showerror, e))
    end
    nothing
end

gui_loadscript(url) = (_busymsg(job_load(urlpath(String(url)))); nothing)
gui_relax() = (_busymsg(job_relax()); nothing)

function gui_run(dur)
    d = strip(String(dur))
    try
        duration = isempty(d) || lowercase(d) in ("inf", "∞") ? Inf : _num(d) * 1e-9
        duration > 0 || error("the duration must be positive")
        _busymsg(job_run(duration))
    catch e
        setstatus!("Input error: " * sprint(showerror, e))
    end
    nothing
end

gui_pause() = (pauseflag[] = !pauseflag[]; nothing)
gui_stop() = (stopflag[] = true; pauseflag[] = false; nothing)

function gui_field(bx, by, bz)
    try
        job_field((_num(bx), _num(by), _num(bz)) .* 1e-3)
    catch e
        setstatus!("Input error: " * sprint(showerror, e))
    end
    nothing
end

function gui_view(mode, lay, energy)
    view_mode[] = String(mode)
    w = view_mode[] == "wheel"
    im_wheel.visible[] = w; im_comp.visible[] = !w; cbar.blockscene.visible[] = !w
    want_energy[] = Bool(energy)
    if Int(lay) != layer[]
        job_layer(Int(lay))
    elseif last_slice[] !== nothing
        _drawslice!(last_slice[], nothing)
    end
    return 2
end

@qmlfunction gui_poll gui_build gui_loadscript gui_relax gui_run gui_pause gui_stop gui_field gui_view

# A check without a person at the window: JULIAMAG_GUI_SMOKE=out.png waits for the window, builds
# a vortex disc, relaxes it, runs it in a field for 30 s while it logs the steps per second and the
# updates per second to out.png.log, then has the window save a picture of itself and quit.
const smoke = get(ENV, "JULIAMAG_GUI_SMOKE", "")
const smokelog = Ref{Any}(nothing)
const shoot = Ref(false)
const mute = Ref(false)                   # smoke: the poll does nothing (A/B of the window's cost)
function slog(x...)
    io = smokelog[]
    io === nothing && return
    println(io, round(time(), digits = 2), " ", x...); flush(io)
end
if !isempty(smoke)
    smokelog[] = open(smoke * ".log", "w")
    slog("threads ", Threads.nthreads(:default), "+", Threads.nthreads(:interactive), ", main thread in pool ", Threads.threadpool())
    Threads.@spawn :default begin
        try
            t0 = time(); dumped = false
            while fps_text[] == ""                     # wait for the window to poll
                nap(0.2)
                if !dumped && time() - t0 > 300
                    slog("no poll after 300 s: task backtraces to stderr"); dumped = true
                    ccall(:jl_print_task_backtraces, Cvoid, ())
                end
            end
            slog("window polling: ", fps_text[])
            job_build(64, 64, 1, 4.0, "Permalloy", 0.02, "Vortex", "Cylinder", 256.0, false); wait(job[])
            job_relax(); wait(job[])
            job_field((10e-3, 0.0, 0.0)); wait(job[])
            slog("run")
            job_run(Inf)
            nap(4)                                     # past the start of the run
            for i in 1:8                               # window on / muted, alternately
                mute[] = iseven(i)
                n0 = (@lock lk snap[].nsteps); w0 = time()
                nap(8)
                n1 = (@lock lk snap[].nsteps); w1 = time()
                slog(mute[] ? "muted " : "window", ": ", round((n1 - n0) / (w1 - w0), digits = 1), " steps/s, ", fps_text[])
            end
            mute[] = false; nap(1)
            shoot[] = true
        catch e
            slog(sprint(showerror, e, catch_backtrace()))
        end
    end
end

# updates of the window per second (JULIAMAG_GUI_HZ, default 30)
const pollms = round(Int, 1000 / parse(Float64, get(ENV, "JULIAMAG_GUI_HZ", "30")))

loadqml(joinpath(@__DIR__, "main.qml"); props = props, materials = materials, pollms = pollms,
        plot_curve = curves, plot_field = field, smoke = smoke)
exec()
stopflag[] = true
# The window has closed with its OpenGL context: forget GLMakie's screens without the GL calls
# that its exit hook would make on them.
empty!(GLMakie.ALL_SCREENS); empty!(GLMakie.SINGLETON_SCREEN)
for ((atlas, ctx), (tex, func)) in GLMakie.atlas_texture_cache
    Makie.remove_font_render_callback!(atlas, func)
end
empty!(GLMakie.atlas_texture_cache)
