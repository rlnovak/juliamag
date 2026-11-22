# Cost of the time stepper on the GPU, without output: the 128 nm disc of
# disk_40mA_sim.jl (64×64×8, Slonczewski current), Float32. For each solver it runs a
# warm-up (compilation), then a timed stretch, and reports ms per step and per
# evaluation, against the cost of one bare right-hand side (llgrhs!) measured the
# same way. The difference per step is the stepper's own work (stage combinations,
# normalizations, error reductions, host waits).
#
# Run: julia --project=<env with JuliaMag + CUDA> bench_stepper.jl [PREC]
# The same script runs against older code trees (it only uses TimeStepper,
# advanceto!, llgrhs!), so old and new steppers can be timed on one GPU.

using JuliaMag, CUDA, Printf

T = length(ARGS) >= 1 ? Dict("Float64" => Float64, "Float32" => Float32)[ARGS[1]] : Float32
CUDA.functional() || error("No functional CUDA GPU found.")

D, THICK, CELL = 128e-9, 16e-9, 2e-9
mesh = Mesh((64, 64, 8), (CELL, CELL, CELL))
mat  = Material{T}(Material(Msat = 800e3, Aex = 10.8e-12, alpha = 0.01,
                            pol = 0.37, lambda = 1.0, epsilonPrime = 0.01))
rp = RegionParams(mesh, mat)
setregion!(rp, 0; Msat = 0.0)
setgeometry!(rp, Cylinder(D, 2THICK); id = 1, edgesmooth = 4)
w = World(mesh, rp; demag = true)
JZ = 0.040 / (π * D^2 / 4)
setcurrent!(w, (0.0, 0.0, JZ); zhangli = false, slonczewski = true,
            fixedlayer = (cos(20π / 180), sin(20π / 180), 0.0))
wg = togpu(w)
m0 = togpu(setconfig(T, mesh, VortexConfig(mesh; circ = 1, pol = 1)))

println(deviceinfo(m0), ", ", T); flush(stdout)

# Bare right-hand side.
dm = similar(m0)
for _ in 1:20; llgrhs!(dm, m0, wg); end
CUDA.synchronize()
n = 500
t0 = time_ns(); for _ in 1:n; llgrhs!(dm, m0, wg); end; CUDA.synchronize()
rhs = (time_ns() - t0) / 1e6 / n
@printf("%-24s %8.3f ms\n", "llgrhs! (bare)", rhs); flush(stdout)

for (solver, fixdt, span) in ((:rk45, 0.0, 0.3e-9), (:rk4, 5e-14, 0.1e-9),
                              (:heun, 5e-14, 0.1e-9), (:rk23, 0.0, 0.2e-9))
    ts = TimeStepper(wg, copy(m0); solver, fixdt)
    advanceto!(ts, 5e-12)                                  # compile + settle dt
    s0, e0 = ts.nsteps, ts.nevals
    CUDA.synchronize()
    t0 = time_ns()
    advanceto!(ts, ts.t + span)
    CUDA.synchronize()
    ms = (time_ns() - t0) / 1e6
    ns, ne = ts.nsteps - s0, ts.nevals - e0
    @printf("%-6s %-10s %6d steps %7d evals  %7.3f ms/step  %6.3f ms/eval  overhead %6.3f ms/step\n",
            solver, fixdt == 0 ? "adaptive" : "fixdt", ns, ne, ms / ns, ms / ne, ms / ns - rhs * ne / ns)
    flush(stdout)
end
