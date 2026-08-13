# Finite-temperature demagnetization: the average magnetization of a nanodot
# decreases as the temperature rises, as thermal fluctuations tilt the moments
# away from the easy axis.
#
# A small uniaxial nanodot is initialized along its easy axis (+z) and driven by
# the LLG–Langevin equation (a random thermal field added to the effective field,
# with fluctuation-dissipation statistics) for a range of temperatures. At each
# temperature we let it thermalize and then time-average ⟨mz⟩. The equilibrium
# ⟨mz⟩(T) falls from 1 toward 0 as T grows — the thermal analogue of a
# magnetization curve. A strongly damped material reaches equilibrium quickly.
#
# Run:  julia --project=examples examples/thermal_demagnetization.jl

using JuliaMag
using Printf
using Statistics
using Plots

function mz_at(mesh, mat, TK; dt = 2e-15, warmup = 3e-10, nsamp = 2000, gap = 40)
    sim = Simulation(mesh, mat; demag = false)
    setmag!(sim, UniformConfig(0, 0, 1))
    runthermal!(sim, warmup, TK; dt = dt)          # thermalize
    acc = Float64[]
    for _ in 1:nsamp
        runthermal!(sim, gap * dt, TK; dt = dt)
        push!(acc, average(sim.m)[3])
    end
    return mean(acc), std(acc)
end

function main()
    mesh = Mesh((1, 1, 1), (5e-9, 5e-9, 5e-9))     # a single-domain nanodot
    Ku   = 4e5
    mat  = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5, Ku = Ku, anisU = (0, 0, 1))

    Ts = [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0]
    mzs = Float64[]
    println(" T (K)   ⟨mz⟩    (KuV/kBT)")
    KuV = Ku * JuliaMag.cellvolume(mesh)
    for TK in Ts
        mz, _ = TK == 0 ? (1.0, 0.0) : mz_at(mesh, mat, TK)
        push!(mzs, mz)
        ratio = TK == 0 ? Inf : KuV / (JuliaMag.kB * TK)
        @printf("  %5.0f   %.4f   %.2f\n", TK, mz, ratio); flush(stdout)
    end

    gr()
    plt = plot(Ts, mzs; marker = :circle, lw = 2, legend = false,
               xlabel = "temperature (K)", ylabel = "⟨mz⟩",
               title = "Thermal demagnetization of a uniaxial nanodot",
               titlefontsize = 10, ylims = (0, 1.05))
    out = joinpath(@__DIR__, "thermal_demagnetization.png")
    savefig(plt, out)
    println("Wrote figure → ", out)
end

main()
