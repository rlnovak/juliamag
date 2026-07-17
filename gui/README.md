# JuliaMag GUI

A dedicated desktop window (Qt6/QML via [QML.jl](https://github.com/JuliaGraphics/QML.jl))
with an embedded [Makie](https://makie.org) viewport for live plots and
magnetization visualization — **not** a browser interface. The initial layout
mirrors mumax3 (a control column on the left; visualization on the right) and is
meant to be restyled.

## Setup (once)

QML, QMLMakie, and GLMakie are heavy binary dependencies, so the GUI has its own
environment separate from the core package:

```
julia --project=gui gui/setup.jl
```

This dev-installs JuliaMag from the parent directory and adds the GUI packages.

## Run

```
julia --project=gui gui/app.jl
```

A window opens. Set the mesh, pick a material and initial state, then **Relax**
to reach equilibrium and **Run** to integrate under the applied field. The
⟨m⟩(t) curves and the top-layer `mz` heatmap update live; **Stop** interrupts a
run.

## Structure

- `main.qml` — the window layout (Qt6 QML): control groups and the `MakieViewport`.
- `app.jl` — the Julia driver: builds a `Simulation` from the inputs, exposes
  `gui_relax`/`gui_run`/`gui_stop` to QML via `@qmlfunction`, and streams results
  into Makie `Observable`s. All simulation work is plain JuliaMag; QML is only UI.

## Notes

- `ENV["QSG_RENDER_LOOP"] = "basic"` is set for stable embedded Makie rendering.
- The GUI logic (building/relaxing/running a simulation, extracting the image)
  is exercised headlessly, but the **window itself must be run on a machine with
  a display and Qt** — it cannot be verified in a headless CI. If a control
  doesn't behave as expected, the wiring is in `app.jl`'s QML-callable functions.
