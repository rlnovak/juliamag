# JuliaMag GUI

A desktop window (Qt6/QML via [QML.jl](https://github.com/JuliaGraphics/QML.jl)) with two
embedded [Makie](https://makie.org) viewports. It shows the essentials of a running simulation and
drives it; everything else is done in a Julia script.

## Setup (once)

QML, QMLMakie and GLMakie are heavy binary dependencies, so the GUI has its own environment:

```
julia --project=gui gui/setup.jl
```

This dev-installs JuliaMag from the parent directory and adds the GUI packages.

## Run

```
julia -t auto --project=gui gui/app.jl
```

The GUI needs at least two threads: the simulation runs on a worker thread, the window on the
main one.

**Getting a simulation.**
- **Open script…**, or drop a `.jl` file on the window: the script is run in a fresh module and
  must leave a variable `sim::Simulation`. It should set the simulation up and not run it; the GUI
  runs it. Geometry, regions, materials, currents, AMR contacts, fields in space and time, the
  table and the OVF autosaves of the script all apply.
- **Build** makes a simple one from the inputs: a mesh, a shape (full mesh, cylinder, rectangle,
  ellipse), a material of the library, α and an initial state.

**Driving it.**
- **Relax**: the energy minimizer (`relax!`).
- **Run**: integrate for the duration given (empty: until Stop), with the solver of the
  simulation and its scheduled output. The run ends exactly at the end time.
- **Pause / Resume** and **Stop** act between two steps.
- **Applied field → Set**: the uniform applied field, applied between two steps if a run is going.
  Running does not change the field otherwise.

**What it shows.**
- ⟨m⟩(t) and the total energy E(t). The energy costs a field evaluation (on the host for a GPU
  simulation), so it is computed at most 4 times per second of wall time; untick "energy curve"
  to skip it.
- The state of the run: t, accepted (and undone) steps, dt, maxTorque, ⟨m⟩, E, B_ext, wall time
  and steps per second, device and precision.
- One layer of m, as mumax3's colour wheel (hue: in-plane angle; white +z, black −z) or one
  component; empty cells grey.

## How it works

- `core.jl`: the jobs (build, load, relax, run, set the field) on a worker thread, and the
  snapshots they publish. It needs no window, and `test_core.jl` checks it headlessly:
  `julia -t 2 --project=gui gui/test_core.jl`.
- `app.jl`: the Makie figures and the functions QML calls. A QML timer calls `gui_poll` 30 times
  per second (`JULIAMAG_GUI_HZ` to change it) on the main thread; it takes what the worker
  published, updates the plots, and tells QML which viewport to repaint (a `MakieViewport` is
  redrawn only when QML calls its `update()`).
- `main.qml`: the layout.

**Threads.** Qt's event loop holds the main thread, so:
- the main thread touches the simulation only through a job, and the worker never touches Makie or
  Qt;
- the worker must not wait with `sleep`: Julia's timers are run by the main thread's event loop,
  which is Qt's here, and the wait may never end. A long `Libc.systemsleep` would block garbage
  collection instead. Waits are short naps with a safepoint between them (`nap`).

**What a run through the GUI is.** `advanceto!` on the stepper of the simulation, as `run!`, with a
hook after every accepted step for the window's requests and the publications. It ends exactly at
the end time, and its scheduled output (table, OVF) is that of `run!`. `test_core.jl` checks that
the state after a GUI run is bit for bit that of `run!`.

**Publications.** During a run the worker publishes at most 20 times per second of wall time ⟨m⟩
and the maximum torque of the last step (no extra evaluation); 10 times per second the layer shown
(only that layer is copied from the GPU); 4 times per second (1 on a GPU) the total energy.

## Measurements (2026-11-27, Intel Core Ultra 5 125H, CPU, Float64)

64 × 64 × 1 Permalloy disc, 256 nm, 4 nm cells, adaptive RK45, 10 mT:
- **Publishing:** the same 0.2 ns run with and without the worker's publications, bit-identical
  results, wall time ratio 1.00 (0.96–1.05 over several runs).
- **The window:** in the same process, 8 s with the window updating at 30 per second, alternating
  with 8 s with the poll muted: 182–275 steps/s against 240–274 steps/s, about 15 % slower
  with the window, with a scatter of the same size. The plots kept 30 updates per second.
- **Start-up:** loading the GUI packages takes about 90 s cold. The first build, relax and run
  compile for about a minute more.

`JULIAMAG_GUI_SMOKE=out.png julia -t auto --project=gui gui/app.jl` repeats this check without a
person at the window: it builds and runs the disc, logs the steps per second with the window on
and muted to `out.png.log`, and saves a picture of the window.

## Notes

- The QML module of the Julia-provided components (with QMLMakie's `MakieViewport`) is `jlqml`
  for QML.jl ≥ 0.11.
- `ENV["QSG_RENDER_LOOP"] = "basic"` is set for stable embedded Makie rendering.
- On exit, GLMakie's screens are dropped without OpenGL calls: the context has already gone with
  the window.
- A script runs in a fresh module; its functions (of time, quantities) are called in the newest
  world (`invokelatest`).
- `:tsit5` has no step hook: the GUI runs it in pieces of 1/200 of the duration.
