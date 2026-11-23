# Re-run of one finite-temperature vortex-hysteresis loop (disc_hysteresis_thermal.jl)
# with snapshots in a window of field steps: OVF of m every SNAPDT and the vortex core
# position every CORE_DT, to see where the vortex nucleates and how it moves.
#
# The integration is the same as in disc_hysteresis_thermal.jl, call for call (the same
# run! calls, the same seeds): the extra output goes through the autosave hooks, which
# only read m. So the trajectory, and the loop table written to OUT, must reproduce the
# original run's rows bit for bit (on the same code commit and GPU type). That is the
# check that the snapshots belong to the original loop.
#
# Run:  julia --project=<env> disc_hysteresis_thermal_snap.jl T SEED OUT SNAPDIR NFIRST NLAST
#           [NSTOP [DWELL=2e-9 [DT=1e-13 [ALPHA=0.5 [SNAPDT=1e-10 [CORE_DT=1e-11]]]]]]
#   NFIRST..NLAST  field steps (1-based, as in the loop table) with snapshots
#   NSTOP          stop after this field step (default NLAST)
# The device is the GPU and the precision Float32, as in phase 2.

using JuliaMag, Printf, Random, Statistics, CUDA

TK     = parse(Float64, ARGS[1])
SEED   = parse(Int, ARGS[2])
OUT    = ARGS[3]
SNAP   = ARGS[4]
NFIRST = parse(Int, ARGS[5])
NLAST  = parse(Int, ARGS[6])
NSTOP  = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : NLAST
DWELL  = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 2e-9
DT     = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : 1e-13
ALPHA  = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 0.5
SNAPDT = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : 1e-10
COREDT = length(ARGS) >= 12 ? parse(Float64, ARGS[12]) : 1e-11

function main()
    mesh = Mesh((100, 100, 4), (5e-9, 5e-9, 5e-9))
    rp   = RegionParams(mesh, material("Permalloy"; alpha = ALPHA, precision = Float32))
    setregion!(rp, 0; Msat = 0.0)
    defregion!(rp, 1, Cylinder(500e-9, 1e6))
    sim = Simulation(mesh, rp; demag = true, verbose = false)
    setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))
    Random.seed!(SEED)
    sim.rng = Random.default_rng()
    CUDA.functional() || error("no functional GPU")
    CUDA.seed!(SEED)
    sim = togpu(sim)
    relax!(sim; stopdm = 1e-6)
    settemperature!(sim, TK)
    setsolver!(sim, :heun; fixdt = DT)
    setoutdir!(sim, SNAP)                     # table.txt (core track) + m*.ovf
    tableadd!(sim, q_vortexcore())

    Bmax, dB = 0.15, 0.005
    Bs = vcat(0:dB:Bmax, Bmax:-dB:-Bmax, -Bmax:dB:Bmax)
    nsub = 5
    io = open(OUT, "w")
    println(io, "# Disc vortex hysteresis at T = $TK K, seed $SEED, α = $ALPHA, dwell $DWELL s, ",
            "Heun dt = $DT s, Float32, $(deviceinfo(sim.m)); snapshots for steps $NFIRST-$NLAST")
    println(io, "# B_x (T)\tmx_end\tmy_end\tmz_end\tmx_mean\tmy_mean\tmz_mean\tt (s)")
    wall0 = time()
    for (n, B) in enumerate(Bs)
        n > NSTOP && break
        if n == NFIRST
            autosave!(sim, SNAPDT)
            tableautosave!(sim, COREDT)
        end
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
        if n == NLAST
            autosave!(sim, 0)
            tableautosave!(sim, 0)
        end
        @printf("  step %3d  B = %+6.1f mT  ⟨mx⟩ = %+.3f  wall %.0f s\n", n, B * 1e3, acc[1],
                time() - wall0); flush(stdout)
    end
    close(io)
    @printf("done: %s, wall %.0f s\n", OUT, time() - wall0)
end

main()
