# Lean GPU driver for the 128 nm disk gyration tests (item 1b/1c).
# Usage: julia gyro_lean.jl ES DT TOTAL MODE OUT
#   ES    edgesmooth (0 = staircase like the mumax3 reference, 4 = production port)
#   DT    fixed RK4 step [s]
#   TOTAL simulated time [s]
#   MODE  "stt"  : Slonczewski 40 mA from the relaxed vortex (as the production run)
#         "sttzl": Slonczewski + Zhang-Li with J=(0,0,jz), xi=0 - what mumax3 does
#                  by default when J is set (DisableZhangLiTorque=false)
#         "free" : relax under Bx = 10 mT, switch the field off, gyrate freely (no STT)
# Logs every 5 ps: t, <m> over the disk (region 1), vortex core x,y. No energies
# (those CPU calls were what made the production run take ~7 h).
using JuliaMag, CUDA, Printf
ES = parse(Int, ARGS[1]); DT = parse(Float64, ARGS[2]); TOTAL = parse(Float64, ARGS[3])
MODE = ARGS[4]; OUT = ARGS[5]
CUDA.functional() || error("no GPU")

D = 128e-9; THICK = 16e-9; CELL = 2e-9
mesh = Mesh((64, 64, 8), (CELL, CELL, CELL))
mat  = Material(Msat = 800e3, Aex = 10.8e-12, alpha = 0.01,
                pol = 0.37, lambda = 1.0, epsilonPrime = 0.01)
rp = RegionParams(mesh, mat); setregion!(rp, 0; Msat = 0.0)
setgeometry!(rp, Cylinder(D, 2THICK); id = 1, edgesmooth = ES)
m_cpu = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1)); clearempty!(m_cpu, rp)

Brelax = MODE == "free" ? (10e-3, 0.0, 0.0) : (0.0, 0.0, 0.0)
world = togpu(World(mesh, rp; demag = true, Bext = Brelax))
m = togpu(m_cpu)
mn = Minimizer(world, m; stopdm = 1e-6); minimize!(mn; maxsteps = 20_000); m = mn.m
setexternalfield!(world, (0.0, 0.0, 0.0))
a0 = average_region(Array(m), rp, 1)
@printf("[%s es=%d dt=%.2e] relaxed <m>_disk = (%.5f, %.5f, %.5f) steps=%d\n", MODE, ES, DT, a0..., mn.step)
flush(stdout)

JC = MODE in ("stt", "sttzl") ? 0.040 / (π * D^2 / 4) : 0.0
ZL = MODE == "sttzl"
FIXED = (cos(20π/180), sin(20π/180), 0.0)
msh = world.mesh
function rhs!(dm, mm)
    B = world._Bbuf
    effectivefield!(B, mm, world)
    torque!(dm, mm, B, mat.alpha)
    JC != 0 && slonczewskitorque!(dm, mm, msh, mat, JC, FIXED, THICK; add = true)
    ZL && zhanglitorque!(dm, mm, msh, mat, (0.0, 0.0, JC); add = true)
    return dm
end
k1 = similar(m); k2 = similar(m); k3 = similar(m); k4 = similar(m); tmp = similar(m)
function rk4!(mm, dt, n)
    for _ in 1:n
        rhs!(k1, mm); @. tmp = mm + 0.5dt*k1; normalize!(tmp); rhs!(k2, tmp)
        @. tmp = mm + 0.5dt*k2; normalize!(tmp); rhs!(k3, tmp)
        @. tmp = mm + dt*k3;    normalize!(tmp); rhs!(k4, tmp)
        @. mm = mm + (dt/6)*(k1 + 2k2 + 2k3 + k4); normalize!(mm)
    end
    mm
end
save_dt = 5e-12; per = round(Int, save_dt / DT); nch = round(Int, TOTAL / save_dt)
io = open(OUT, "w")
println(io, "# t mx my mz core_x core_y   (", MODE, " es=", ES, " dt=", DT, ")")
function logrow(t)
    mh = Array(m); a = average_region(mh, rp, 1); vx, vy, _, _ = vortexcore(mh, mesh)
    @printf(io, "%.6e\t%.7f\t%.7f\t%.7f\t%.5e\t%.5e\n", t, a..., vx, vy)
end
logrow(0.0); t0 = time()
for c in 1:nch
    rk4!(m, DT, per); logrow(c * save_dt)
    c % 200 == 0 && (@printf("  t=%.1f ns  wall=%.0f s\n", c*save_dt*1e9, time()-t0); flush(stdout); flush(io))
end
close(io); @printf("done %s  wall=%.0f s\n", OUT, time()-t0)
