# The 40 ns current-driven disc of gpu_validation/disk_stt, written with the
# Simulation driver layer — line for line the mumax3 script
# gpu_validation/disk_stt/mumax3/stt_es4_noZL_40ns.mx3:
#
#   SetGridsize / SetCellsize / EdgeSmooth / setgeom   → Mesh, RegionParams, setgeometry!
#   m = vortex(1, 1); relax()                           → setmag!, relax!
#   TableAdd(…); tableautosave(5e-12)                   → tableadd!, tableautosave!
#   Pol, Lambda, EpsilonPrime, FixedLayer, J,
#   DisableZhangLiTorque = true                         → setcurrent!(…; slonczewski = true,
#                                                                       zhangli = false)
#   run(40e-9)  (RK45, MaxErr 1e-5)                     → run!(sim, 40e-9)
#
# Unlike disk_d1=128nm_t=16nm_40mA.jl (a hand-written fixed-step RK4 loop), the time
# step is chosen by the solver (mumax3's Dormand–Prince by default), and the run
# logs its wall-clock start and end, device, solver and evaluation count.
#
# Run:  julia --project=<env with JuliaMag + CUDA> disk_40mA_sim.jl [OUTDIR [PREC [SOLVER [MAXERR [TOTAL]]]]]
#       defaults: disk_40mA_sim.out, Float32, rk45, 1e-5, 40e-9. SOLVER rk4 uses a fixed
#       step of 5e-14 s (the hand-written driver's).

using JuliaMag, CUDA, Printf

OUTDIR = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "disk_40mA_sim.out")
T      = length(ARGS) >= 2 ? Dict("Float64" => Float64, "Float32" => Float32)[ARGS[2]] : Float32
SOLVER = length(ARGS) >= 3 ? Symbol(ARGS[3]) : :rk45
MAXERR = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-5
TOTAL  = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 40e-9
CUDA.functional() || error("No functional CUDA GPU found.")

D, THICK, CELL = 128e-9, 16e-9, 2e-9
mesh = Mesh((64, 64, 8), (CELL, CELL, CELL))
mat  = Material{T}(Material(Msat = 800e3, Aex = 10.8e-12, alpha = 0.01,
                            pol = 0.37, lambda = 1.0, epsilonPrime = 0.01))
rp = RegionParams(mesh, mat)
setregion!(rp, 0; Msat = 0.0)                                   # empty background
setgeometry!(rp, Cylinder(D, 2THICK); id = 1, edgesmooth = 4)  # EdgeSmooth = 4

sim = Simulation(mesh, rp; demag = true, outdir = OUTDIR)
setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))
sim = togpu(sim)
relax!(sim; stopdm = 1e-6)
@printf("relaxed ⟨m⟩ = (%.5f, %.5f, %.5f)\n", average(sim)...); flush(stdout)

const JZ = 0.040 / (π * D^2 / 4)
for q in (q_maxtorque(), q_demagenergy(), q_exchangeenergy(), q_zeemanenergy(), q_energy(),
          Quantity("Jz", "A/m2", 1, (w, m, t) -> [JZ]; device = true), q_vortexcore())
    tableadd!(sim, q)
end
setcurrent!(sim, (0.0, 0.0, JZ); zhangli = false, slonczewski = true,
            fixedlayer = (cos(20π / 180), sin(20π / 180), 0.0))
tableautosave!(sim, 5e-12)
if SOLVER === :rk4
    setsolver!(sim, :rk4; fixdt = 5e-14)
else
    setsolver!(sim, SOLVER; maxerr = MAXERR)
end
run!(sim, TOTAL)
@printf("NSteps %d  NUndone %d  NEvals %d  PeakErr %.3e\n", sim.nsteps, sim.nundone,
        sim.nevals, sim.peakerr)
