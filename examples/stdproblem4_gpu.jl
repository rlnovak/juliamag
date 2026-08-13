# µMAG Standard Problem 4 on the GPU — full simulation (relax + switching)
# entirely on a CUDA device, validated against the CPU reference table.
#
# The same JuliaMag source runs on the GPU: the state and the World are moved to
# the device with togpu, after which the energy minimizer (relax to the S-state)
# and the LLG time integrator (the 1 ns switching) dispatch every field, torque,
# and solver kernel to a GPU method. This reproduces the CPU ⟨m⟩(t) curve to
# integrator tolerance.
#
# Setup:  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.add("CUDA")'
# Run:    julia --project=. examples/stdproblem4_gpu.jl

using JuliaMag
using CUDA
using Printf
using DelimitedFiles

# Plotting is optional: the table is always written; the figure is drawn only if
# Plots is available in the active environment (it is not a dependency of the
# core project, which carries CUDA). Run under an env that has both to get the plot.
const HAVE_PLOTS = try; @eval using Plots; true; catch; false; end

CUDA.functional() || error("No functional CUDA GPU found.")
println("CUDA device: ", CUDA.name(CUDA.device()))

function main()
    mesh = Mesh((160, 40, 1), (3.125e-9, 3.125e-9, 3e-9))
    mat  = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)

    # --- Phase 1: relax to the S-state on the GPU ---
    wg = togpu(World(mesh, mat; demag = true))
    mg = togpu(uniform(mesh, (1, 1, 1)))
    println("Relaxing to the S-state on the GPU…"); flush(stdout)
    mn = Minimizer(wg, mg; stopdm = 1e-6); minimize!(mn; maxsteps = 20_000)
    sx, sy, sz = average(mn.m)
    @printf("  relaxed after %d steps, ⟨m⟩ = (%.4f, %.4f, %.4f)\n", mn.step, sx, sy, sz)

    # --- Phase 2: switching under field 1, on the GPU ---
    wg2 = togpu(World(mesh, mat; demag = true, Bext = (-24.6e-3, 4.3e-3, 0.0)))
    it  = Integrator(wg2, mn.m; abstol = 1e-6, reltol = 1e-5, dtmax = 5e-13, tend = 1e-9)
    n = 400; Δt = 1e-9 / n
    ts = Float64[]; mxs = Float64[]; mys = Float64[]; mzs = Float64[]
    a = average(state(it)); push!(ts, currenttime(it))
    push!(mxs, a[1]); push!(mys, a[2]); push!(mzs, a[3])
    println("Integrating the 1 ns switching on the GPU…"); flush(stdout)
    for s in 1:n
        advance!(it, Δt)
        a = average(state(it))
        push!(ts, currenttime(it)); push!(mxs, a[1]); push!(mys, a[2]); push!(mzs, a[3])
        s % 100 == 0 && (@printf("  t = %.2f ns  ⟨m⟩ = (%.4f, %.4f, %.4f)\n",
                                 currenttime(it)*1e9, a...); flush(stdout))
    end

    # --- Save the GPU table ---
    tbl = joinpath(@__DIR__, "stdproblem4_gpu.txt")
    open(tbl, "w") do io
        println(io, "# muMAG Standard Problem 4 — JuliaMag on GPU (", CUDA.name(CUDA.device()), ")")
        println(io, "# t[s]\tmx\tmy\tmz")
        for i in eachindex(ts)
            @printf(io, "%.6e\t%.6f\t%.6f\t%.6f\n", ts[i], mxs[i], mys[i], mzs[i])
        end
    end
    println("Wrote table → ", tbl)

    # --- Compare with the CPU reference table if present ---
    ref = joinpath(@__DIR__, "stdproblem4.txt")
    if isfile(ref)
        d = readdlm(ref, '\t'; comments = true, comment_char = '#')
        n2 = min(size(d, 1), length(mxs))
        dmx = maximum(abs.(d[1:n2,2] .- mxs[1:n2]))
        dmy = maximum(abs.(d[1:n2,3] .- mys[1:n2]))
        dmz = maximum(abs.(d[1:n2,4] .- mzs[1:n2]))
        @printf("Max |GPU - CPU| over the run: mx=%.2e my=%.2e mz=%.2e\n", dmx, dmy, dmz)
    end

    # --- Plot (only if Plots is available) ---
    if HAVE_PLOTS
        gr(); tns = ts .* 1e9
        plt = Plots.plot(tns, mxs; label = "⟨mx⟩", lw = 2, color = :red, xlabel = "time (ns)",
                   ylabel = "⟨m⟩", title = "µMAG Standard Problem 4 — GPU", legend = :right)
        Plots.plot!(plt, tns, mys; label = "⟨my⟩", lw = 2, color = :green)
        Plots.plot!(plt, tns, mzs; label = "⟨mz⟩", lw = 2, color = :blue)
        Plots.hline!(plt, [0]; color = :gray, ls = :dash, label = "")
        png = joinpath(@__DIR__, "stdproblem4_gpu.png")
        Plots.savefig(plt, png); println("Wrote figure → ", png)
    else
        println("(Plots not in this environment — table written, figure skipped.)")
    end
end

main()
