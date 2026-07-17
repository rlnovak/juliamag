# One-time setup for the JuliaMag GUI environment.
#
# QML, QMLMakie, and GLMakie are heavy dependencies with binary artifacts, so the
# GUI lives in its own environment (gui/) rather than in the core package. This
# script dev-installs JuliaMag from the parent directory and adds the GUI
# dependencies by name (letting Pkg resolve their UUIDs).
#
# Run once:  julia --project=gui gui/setup.jl
# Then:      julia --project=gui gui/app.jl

using Pkg
Pkg.develop(PackageSpec(path = dirname(@__DIR__)))   # JuliaMag from ../
for pkg in ("QML", "QMLMakie", "GLMakie", "Makie", "Observables", "Colors", "Printf")
    Pkg.add(pkg)
end
Pkg.instantiate()
println("GUI environment ready. Run:  julia --project=gui gui/app.jl")
