# µMAG Standard Problem 4
# https://www.ctcms.nist.gov/~rdm/std4/spec4.html
#
# Permalloy film 500 × 125 × 3 nm, discretized 160 × 40 × 1 (cells 3.125 × 3.125
# × 3 nm). Relax to the equilibrium S-state, then apply field 1 —
# (-24.6, 4.3, 0) mT, ~25 mT at 170° from +x — and record the spatially averaged
# magnetization ⟨m⟩(t) for 1 ns.
#
# Outputs:
#   stdproblem4.txt — table of t[s], mx, my, mz
#   stdproblem4.png — ⟨m⟩ components vs. time
#
# Run:  julia --project=examples examples/stdproblem4.jl

using JuliaMag
using Printf
include(joinpath(@__DIR__, "makie_shim.jl"))

# --- Phase 1: relax to the S-state -----------------------------------------
function relax_sstate(mesh, mat)
    # The µMAG spec defines the initial state as "the equilibrium S-state obtained
    # after applying and slowly reducing a saturating field along [1,1,1] to
    # zero." We reproduce that by saturating along (1,1,1) and minimizing the
    # energy at zero applied field: the demag field then curls the ends into the
    # S while the +y and +z memory of the tilt sets the S's sense.
    m = uniform(mesh, (1, 1, 1))

    # Minimize the energy — much faster than integrating the LLG in time.
    world = World(mesh, mat; demag = true)
    println("\nRelaxing to the S-state (energy minimizer)…"); flush(stdout)
    mn = Minimizer(world, m; stopdm = 1e-6)
    minimize!(mn; maxsteps = 20_000, verbose = true)
    mx, my, mz = average(mn.m)
    @printf("  relaxed after %d steps, ⟨m⟩ = (%.4f, %.4f, %.4f)\n", mn.step, mx, my, mz)
    flush(stdout)
    return mn.m
end

# --- Phase 2: apply field 1 and record the switching -----------------------
function run_switching(mesh, mat, m0; t_total = 1e-9, n_samples = 400)
    world = World(mesh, mat; demag = true, Bext = (-24.6e-3, 4.3e-3, 0.0))
    it = Integrator(world, m0; abstol = 1e-6, reltol = 1e-5, dtmax = 5e-13, tend = t_total)
    Δt = t_total / n_samples

    ts = Float64[]; mxs = Float64[]; mys = Float64[]; mzs = Float64[]
    mx, my, mz = average(state(it))
    push!(ts, currenttime(it)); push!(mxs, mx); push!(mys, my); push!(mzs, mz)

    println("\nIntegrating the switching for 1 ns (OrdinaryDiffEq/Tsit5)…"); flush(stdout)
    for s in 1:n_samples
        advance!(it, Δt)
        mx, my, mz = average(state(it))
        push!(ts, currenttime(it)); push!(mxs, mx); push!(mys, my); push!(mzs, mz)
        if s % 50 == 0
            @printf("  t = %.2f ns  ⟨m⟩ = (%.4f, %.4f, %.4f)\n",
                    currenttime(it) * 1e9, mx, my, mz)
            flush(stdout)
        end
    end
    return ts, mxs, mys, mzs
end

function main()
    mesh = Mesh((160, 40, 1), (3.125e-9, 3.125e-9, 3e-9))
    mat  = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)

    println("Standard Problem 4")
    show(stdout, MIME"text/plain"(), mesh); println()
    println("  exchange length = ", round(exchangelength(mat) * 1e9, digits = 2), " nm ",
            "(cell 3.125 nm is safely below it)")
    flush(stdout)

    m0 = relax_sstate(mesh, mat)
    ts, mxs, mys, mzs = run_switching(mesh, mat, m0)

    # --- Save the table ----------------------------------------------------
    tablepath = joinpath(@__DIR__, "stdproblem4.txt")
    open(tablepath, "w") do io
        println(io, "# muMAG Standard Problem 4 — JuliaMag")
        println(io, "# t[s]\tmx\tmy\tmz")
        for i in eachindex(ts)
            @printf(io, "%.6e\t%.6f\t%.6f\t%.6f\n", ts[i], mxs[i], mys[i], mzs[i])
        end
    end
    println("\nWrote table  → ", tablepath); flush(stdout)

    # --- Plot --------------------------------------------------------------
    gr()
    tns = ts .* 1e9
    plt = plot(tns, mxs; label = "⟨mx⟩", lw = 2, color = :red,
               xlabel = "time (ns)", ylabel = "⟨m⟩",
               title = "µMAG Standard Problem 4 (field 1)", legend = :right)
    plot!(plt, tns, mys; label = "⟨my⟩", lw = 2, color = :green)
    plot!(plt, tns, mzs; label = "⟨mz⟩", lw = 2, color = :blue)
    hline!(plt, [0]; color = :gray, ls = :dash, label = "")

    pngpath = joinpath(@__DIR__, "stdproblem4.png")
    savefig(plt, pngpath)
    println("Wrote figure → ", pngpath); flush(stdout)
end

main()
