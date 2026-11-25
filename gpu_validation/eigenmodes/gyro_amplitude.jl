# Free gyration of the vortex of disc_gyrotropic.jl for several amplitudes (GPU, Float64):
# relaxed under Bx = B, field off, gyration. The frequency extrapolated to zero amplitude is
# the linear eigenfrequency.
#   julia --project=. gpu_validation/eigenmodes/gyro_amplitude.jl OUT TOTAL DT B_mT...
using JuliaMag, CUDA, Printf
CUDA.functional() || error("no GPU")
OUT = ARGS[1]; TOTAL = parse(Float64, ARGS[2]); DT = parse(Float64, ARGS[3]); mkpath(OUT)
mesh = Mesh((64, 64, 8), (2e-9, 2e-9, 2e-9))
rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.08e-11, alpha = 0.01))
setregion!(rp, 0; Msat = 0.0); setregion!(rp, 1; Msat = 8e5, Aex = 1.08e-11)
setgeometry!(rp, Cylinder(128e-9, 1.0); id = 1)
for a in ARGS[4:end]
    B = parse(Float64, a) * 1e-3
    m0 = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1)); clearempty!(m0, rp)
    w = togpu(World(mesh, rp; demag = true, Bext = (B, 0, 0)))
    mn = Minimizer(w, togpu(m0); stopdm = 1e-9); minimize!(mn; maxsteps = 50_000)
    setexternalfield!(w, (0, 0, 0))
    @printf("B = %s mT: relaxed in %d steps\n", a, mn.step); flush(stdout)
    ts = TimeStepper(w, mn.m; solver = :rk4, fixdt = DT)
    open(joinpath(OUT, "gyro_$(a)mT.txt"), "w") do io
        println(io, "# t (s)\tmx\tmy\tmz")
        t0 = time()
        for c in 0:round(Int, TOTAL / 1e-11)
            c > 0 && advanceto!(ts, c * 1e-11)
            @printf(io, "%.6e\t%.12e\t%.12e\t%.12e\n", c * 1e-11, average_region(Array(mn.m), rp, 1)...)
            c % 100 == 0 && (flush(io); @printf("  t = %.1f ns, %.0f s\n", c * 0.01, time() - t0); flush(stdout))
        end
    end
end
println("done")
