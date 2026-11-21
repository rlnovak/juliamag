# Is the Langevin stage-2 problem (examples/stage2_local.jl) under-thermalized at
# small ξ? The script thermalizes for 2 ns from a random state, but the relaxation
# time of a free macrospin in the applied field, 1/(α γ' μ0 H), is 3.5 ns at ξ = 1
# (μ0H = 4.1 mT). Same ensemble (4096 uncoupled macrospins, 10 nm cells, Ms = 1e6 A/m,
# α = 0.5, T = 300 K), thermalized for 20 ns; ⟨mx⟩ averaged over the ensemble and 40
# snapshots, as in the script.
#
# Run:  julia --project=. gpu_validation/revalidation/langevin_thermalization.jl [DT [TTHERM]]
using JuliaMag, Printf, Statistics

dt     = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1e-14
ttherm = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 20e-9
μ0, kB = JuliaMag.μ0, JuliaMag.kB
mesh = Mesh((16, 16, 16), (10e-9, 10e-9, 10e-9))
Ms = 1e6
mat = Material(Msat = Ms, Aex = 0.0, alpha = 0.5)
V = JuliaMag.cellvolume(mesh); T = 300.0
langevin(ξ) = coth(ξ) - 1 / ξ
@printf("dt = %.1e s, thermalization %.0f ns\n", dt, ttherm * 1e9)
for ξ in (1.0, 2.0, 5.0)
    H = ξ * kB * T / (μ0 * Ms * V)
    τ = 1 / (0.5 * γLL / 1.25 * μ0 * H)
    sim = Simulation(mesh, mat; demag = false, verbose = false)
    setmag!(sim, RandomConfig())
    setexternalfield!(sim, (μ0 * H, 0.0, 0.0))
    runthermal!(sim, ttherm, T; dt = dt)
    acc = Float64[]
    for _ in 1:40
        runthermal!(sim, 40 * dt, T; dt = dt)
        push!(acc, average(sim.m)[1])
    end
    @printf("ξ = %4.1f  τ = %.2f ns  ⟨mx⟩ = %.4f ± %.4f   L(ξ) = %.4f\n", ξ, τ * 1e9,
            mean(acc), std(acc) / sqrt(length(acc)), langevin(ξ))
    flush(stdout)
end
