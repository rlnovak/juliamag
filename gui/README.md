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

A window opens. Set the mesh (with optional periodic-x and an optional geometry
shape), pick a material and initial state, then:

- **Build** assembles the `Simulation` from the inputs.
- **Relax** minimizes to equilibrium.
- **Run** integrates under the applied field, streaming results live.
- **Pause** suspends/resumes a run; **Step** advances one `Δt` chunk (when not
  running); **Stop** ends the run.

Two live panels on the right:

1. **Time series** — a chosen scalar vs. time. The dropdown below selects the
   y-quantity: `mx/my/mz`, the vortex or skyrmion `x/y` position, or the
   topological charge. (Switching quantity clears the curve, since the units
   differ.)
2. **Field** — an in-plane `(mx,my)` quiver over a colour map of the current
   state. The dropdowns below choose the mapped component (`mz/mx/my`), the
   colormap, and the z-layer (for multilayer meshes); the **show** checkbox turns
   the panel off to save rendering cost. During a run the field is redrawn every
   few chunks (throttled); it always redraws on pause, step, and completion.

**Drag and drop** a `.jl` script defining a variable `sim::Simulation` onto the
window to load and visualize it — e.g. any of the `examples/` scripts adapted to
leave a `sim` in scope. This is the GUI's on-ramp for problems too complex for
the built-in controls (multi-region geometry, custom materials, currents).

## Structure

- `main.qml` — the window layout (Qt6 QML): the control column, a `DropArea`, and
  the two `MakieViewport`s with their option dropdowns.
- `app.jl` — the Julia driver: builds a `Simulation` from the inputs (or a dropped
  script), exposes `gui_build`/`gui_relax`/`gui_run`/`gui_pause`/`gui_step`/
  `gui_stop`/`gui_loadscript`/`gui_set_ycurve`/`gui_set_field` to QML via
  `@qmlfunction`, and streams results into Makie `Observable`s. All simulation
  work is plain JuliaMag; QML is only UI. The GUI never modifies the core package
  — it is an optional front end over the same file→CLI workflow.

## Notes

- The QML module for the Julia-provided components (including QMLMakie's
  `MakieViewport`) is `jlqml` — `main.qml` does `import jlqml`. Earlier QML.jl
  versions used `org.julialang`; if you see *module "org.julialang" is not
  installed*, your QML.jl is ≥ 0.11 and the import must be `jlqml` (already fixed
  here).
- The Makie `Figure` is handed to QML as the context property `plot`, bound in
  the QML by `MakieViewport { scene: plot }`. If a QML.jl version rejects a
  `Figure` as a `loadqml` keyword, pass it instead with
  `set_context_property(qmlcontext(), "plot", plot)` after `loadqml`.
- `ENV["QSG_RENDER_LOOP"] = "basic"` is set for stable embedded Makie rendering.
- The GUI logic (building/relaxing/running a simulation, extracting the image)
  is exercised headlessly, but the **window itself must be run on a machine with
  a display and Qt** — it cannot be verified in a headless CI. If a control
  doesn't behave as expected, the wiring is in `app.jl`'s QML-callable functions.
