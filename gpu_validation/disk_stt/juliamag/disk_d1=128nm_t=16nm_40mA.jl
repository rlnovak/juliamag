# Single Permalloy disk driven by a perpendicular (Slonczewski) spin current — GPU.
# JuliaMag port of the mumax3 script disk_d1=128nm_t=16nm_40mA (sims1).
#
# Physical model: Slonczewski torque only, EdgeSmooth 4. mumax3 adds the Zhang–Li
# torque by default whenever J is set (DisableZhangLiTorque = false); with a current
# along z it acts through ∂z m across the thickness and lowers the gyration
# frequency by 4.4%. The standard model of the current-perpendicular vortex
# oscillator is Slonczewski (+ Oersted field; Khvalkovskiy et al., PRB 80, 140401(R)
# (2009)), so this port applies Slonczewski only. Its like-for-like mumax3
# counterpart is gpu_validation/disk_stt/mumax3/stt_es4_noZL_40ns.mx3 (EdgeSmooth = 4,
# DisableZhangLiTorque = true). Neither code includes the Oersted field.
#
# Table columns follow the mumax3 reference table:
#   t  mx my mz  maxTorque  E_demag E_exch E_Zeeman E_total  Jz  core_x core_y
#   ⟨m⟩       = average_magnet: fill-weighted over the disk, mumax3's table m
#   maxTorque = max |dm/dt| / γLL incl. the spin torque, in T (mumax3 maxTorque =
#               max |torque| / γ0)
#
# Runs on a CUDA GPU: the World and the state are moved to the device with togpu.
# Energies, maxTorque and the vortex core are evaluated on a host copy (Array(m))
# at the logging points only.
#
# Run:   julia --project=<env with JuliaMag + CUDA> disk_d1=128nm_t=16nm_40mA.jl [TOTAL [TABLE]]
#        TOTAL defaults to 40e-9 s; TABLE to disk_d1=128nm_t=16nm_40mA_table.txt
# Plots: Python/matplotlib (repo examples/plot_disk40_compare.py).

using JuliaMag
using CUDA
using Printf

CUDA.functional() || error("No functional CUDA GPU found.")
println("CUDA device: ", CUDA.name(CUDA.device())); flush(stdout)

const OUT   = @__DIR__
const TOTAL = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 40e-9
const TABLE = length(ARGS) >= 2 ? ARGS[2] : joinpath(OUT, "disk_d1=128nm_t=16nm_40mA_table.txt")

# --- Geometry (128 nm disk, 16 nm thick, 2 nm cells) -------------------------
const D     = 128e-9
const THICK = 16e-9
const CELL  = 2e-9
const NX = round(Int, D / CELL); const NY = round(Int, D / CELL); const NZ = round(Int, THICK / CELL)

mesh = Mesh((NX, NY, NZ), (CELL, CELL, CELL))
mat  = Material(Msat = 800e3, Aex = 10.8e-12, alpha = 0.01,
                pol = 0.37, lambda = 1.0, epsilonPrime = 0.01)   # Slonczewski params
rp   = RegionParams(mesh, mat)
setregion!(rp, 0; Msat = 0.0)                          # empty background
setgeometry!(rp, Cylinder(D, 2 * THICK); id = 1, edgesmooth = 4) # the disk

# --- Initial vortex state, cleared, then moved to the GPU --------------------
# clearempty! is CPU-only, so clear the empty cells on the host BEFORE togpu.
m_cpu = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1))
clearempty!(m_cpu, rp)

# CPU World for the host-side diagnostics; the GPU World is a device copy of it
# (togpu copies, so the two never share state).
wcpu  = World(mesh, rp; demag = true)
world = togpu(wcpu)
m     = togpu(m_cpu)                                   # a CuArray

# --- Relax the vortex on the GPU ---------------------------------------------
println("Relaxing the vortex on the GPU…"); flush(stdout)
mn = Minimizer(world, m; stopdm = 1e-6); minimize!(mn; maxsteps = 20_000)
m  = mn.m
@printf("relaxed ⟨m⟩ = (%.5f, %.5f, %.5f) after %d steps\n",
        average_magnet(Array(m), rp)..., mn.step); flush(stdout)

# --- Current and fixed-layer polarization (mumax3 values) --------------------
const JTOT  = 0.040
const AREA  = π * D^2 / 4
const JC    = JTOT / AREA
const ANGLE = 20 * π / 180
const FIXED = (cos(ANGLE), sin(ANGLE), 0.0)
@printf("Current density Jz = %.6e A/m²\n", JC); flush(stdout)

# --- Integrate with the Slonczewski torque (fixed-step RK4, on GPU) ----------
# dt = 5e-14 s: dt/2 gives the same frequency and max |Δ⟨m⟩| < 1e-6 (validated).
msh = world.mesh
function rhs!(dm, mm)
    B = world._Bbuf
    effectivefield!(B, mm, world)
    torque!(dm, mm, B, mat.alpha)
    slonczewskitorque!(dm, mm, msh, mat, JC, FIXED, THICK; add = true)
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
    return mm
end

dt      = 5e-14
save_dt = 5e-12
nchunk  = round(Int, TOTAL / save_dt)
per     = max(1, round(Int, save_dt / dt))

dmh = zeros(3, NX, NY, NZ)
function maxtq(mh)   # mumax3 maxTorque: max |torque| / γ0 incl. the spin torque [T]
    B = wcpu._Bbuf
    effectivefield!(B, mh, wcpu)
    torque!(dmh, mh, B, mat.alpha)
    slonczewskitorque!(dmh, mh, msh, mat, JC, FIXED, THICK; add = true)
    return maxtorque(dmh) / γLL
end

tbl = open(TABLE, "w")
println(tbl, "# JuliaMag(GPU) — single 128 nm × 16 nm disk, Slonczewski STT only, Jtot=40 mA, EdgeSmooth 4")
println(tbl, "# t (s)\tmx ()\tmy ()\tmz ()\tmaxTorque (T)\tE_demag (J)\tE_exch (J)\tE_Zeeman (J)\tE_total (J)\tJz (A/m2)\tcore_x (m)\tcore_y (m)")
function logrow!(t)
    mh = Array(m)                                     # host copy for the diagnostics
    a  = average_magnet(mh, rp)                       # mumax3 table m (fill-weighted)
    vx, vy, _, _ = vortexcore(mh, msh)
    @printf(tbl, "%.6e\t%.7f\t%.7f\t%.7f\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.5e\t%.5e\n",
            t, a[1], a[2], a[3], maxtq(mh),
            demagenergy(mh, wcpu.demagplan, msh, rp), exchangeenergy(mh, msh, rp),
            zeemanenergy(mh, wcpu.Bext, msh, rp), totalenergy(mh, wcpu), JC, vx, vy)
    return a, vx, vy
end

@printf("Integrating %.3g ns with the Slonczewski current (GPU)…\n", TOTAL * 1e9); flush(stdout)
logrow!(0.0)
t0 = time()
for c in 1:nchunk
    rk4!(m, dt, per)
    a, vx, vy = logrow!(c * per * dt)
    if c % 200 == 0
        @printf("  t = %.1f ns  ⟨m⟩ = (%.4f, %.4f, %.4f)  core = (%.1f, %.1f) nm  wall = %.0f s\n",
                c*per*dt*1e9, a..., vx*1e9, vy*1e9, time() - t0)
        flush(stdout); flush(tbl)
    end
end
close(tbl)
@printf("Done. wall = %.0f s. Wrote table → %s\n", time() - t0, TABLE)
