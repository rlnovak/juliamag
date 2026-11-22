# Vortex hysteresis of the 500 nm × 20 nm Permalloy disc at finite temperature.
#
# Same disc, mesh and field sweep as examples/disc_hysteresis.jl (0 → +150 → −150 →
# +150 mT along x, 5 mT steps), but at each field step the magnetization evolves
# under the LLG–Langevin equation for a fixed dwell time instead of being relaxed
# with the energy minimizer. The thermal field breaks the mirror symmetry of the
# problem physically, so the field at which the vortex renucleates is sampled from
# its thermal distribution instead of being set by round-off (see the T = 0
# discussion in gpu_validation/revalidation/README.md). Run an ensemble of seeds
# per temperature and compare the switching-field distributions.
#
# Quasi-static protocol: the damping is raised to α = 0.5 (default) so that each
# field step relaxes within the dwell time; α sets the relaxation rate, not the
# equilibrium, and the thermal field uses the same α (fluctuation–dissipation).
# Ms and A keep their room-temperature values at every T. The sweep rate,
# 5 mT per dwell (2.5e6 T/s for 2 ns), is many orders of magnitude above an
# experiment's, so the switching fields are upper bounds for thermally activated
# switching. The 2 ns dwell comes from a convergence check on the virgin branch at
# 50 K (gpu_validation/thermal_hysteresis/phase1): 0.5 and 1 ns lag the T = 0 branch
# at low field and delay the annihilation by 5 mT; dt = 1e-13 s is converged at 50
# and 300 K.
#
# Integration: Heun with a fixed step (mumax3's choice for thermal runs). Each field
# step logs ⟨m⟩ at the end of the dwell and its average over the last quarter.
#
# Run:  julia --project=<env with JuliaMag (+ CUDA)> disc_hysteresis_thermal.jl \
#           T SEED [OUT [DWELL [DT [ALPHA [DEVICE [PREC [NSTEPS]]]]]]]
#   T       temperature [K]
#   SEED    random seed (CUDA.seed! on the GPU, Random.seed! on the CPU)
#   OUT     table path (default disc_hysteresis_T<T>_seed<SEED>.txt next to this file)
#   DWELL   time per field step [s] (default 2e-9)
#   DT      fixed time step [s] (default 1e-13)
#   ALPHA   damping (default 0.5)
#   DEVICE  gpu | cpu (default gpu)
#   PREC    Float32 | Float64 (default Float32)
#   NSTEPS  only the first NSTEPS field steps (default: the whole sweep; for tests)

using JuliaMag, Printf, Random, Statistics

TK     = parse(Float64, ARGS[1])
SEED   = parse(Int, ARGS[2])
OUT    = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, @sprintf("disc_hysteresis_T%g_seed%d.txt", TK, SEED))
DWELL  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 2e-9
DT     = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1e-13
ALPHA  = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 0.5
DEVICE = length(ARGS) >= 7 ? ARGS[7] : "gpu"
PREC   = length(ARGS) >= 8 ? Dict("Float32" => Float32, "Float64" => Float64)[ARGS[8]] : Float32
NSTEPS = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : typemax(Int)

DEVICE == "gpu" && @eval using CUDA

function main()
    mesh = Mesh((100, 100, 4), (5e-9, 5e-9, 5e-9))           # 500 × 500 × 20 nm
    rp   = RegionParams(mesh, material("Permalloy"; alpha = ALPHA, precision = PREC))
    setregion!(rp, 0; Msat = 0.0)                             # empty background
    defregion!(rp, 1, Cylinder(500e-9, 1e6))                  # the 500 nm disc
    sim = Simulation(mesh, rp; demag = true, verbose = false)
    setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))
    Random.seed!(SEED)
    sim.rng = Random.default_rng()
    if DEVICE == "gpu"
        CUDA.functional() || error("no functional GPU")
        CUDA.seed!(SEED)
        sim = togpu(sim)
    end
    relax!(sim; stopdm = 1e-6)                                # T = 0 vortex to start from
    settemperature!(sim, TK)
    setsolver!(sim, :heun; fixdt = DT)

    Bmax, dB = 0.15, 0.005
    Bs = vcat(0:dB:Bmax, Bmax:-dB:-Bmax, -Bmax:dB:Bmax)
    nsub = 5                                                  # samples over the last quarter
    io = open(OUT, "w")
    println(io, "# Disc vortex hysteresis at T = $TK K, seed $SEED, α = $ALPHA, dwell $DWELL s, ",
            "Heun dt = $DT s, $(PREC), $(deviceinfo(sim.m))")
    println(io, "# B_x (T)\tmx_end\tmy_end\tmz_end\tmx_mean\tmy_mean\tmz_mean\tt (s)")
    @printf("T = %g K, seed %d, %s\n", TK, SEED, deviceinfo(sim.m)); flush(stdout)
    wall0 = time()
    for (n, B) in enumerate(Bs)
        n > NSTEPS && break
        setexternalfield!(sim, (B, 0.0, 0.0))
        run!(sim, 0.75 * DWELL)
        acc = zeros(3)
        for _ in 1:nsub
            run!(sim, 0.25 * DWELL / nsub)
            acc .+= average(sim)
        end
        a = average(sim); acc ./= nsub
        @printf(io, "%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6e\n", B, a..., acc..., sim.t)
        flush(io)
        n % 10 == 0 && (@printf("  step %3d/%d  B = %+6.1f mT  ⟨mx⟩ = %+.3f  wall %.0f s\n",
                                n, length(Bs), B * 1e3, acc[1], time() - wall0); flush(stdout))
    end
    close(io)
    @printf("done: %s, %d steps, %d evaluations, wall %.0f s\n", OUT, sim.nsteps, sim.nevals,
            time() - wall0)
end

main()
