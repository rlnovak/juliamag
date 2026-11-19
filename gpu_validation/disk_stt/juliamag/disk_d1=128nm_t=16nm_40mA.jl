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
# Precision: Float64 (default) or Float32 — the latter is mumax3's precision; the
# whole run (material, fields, torques, integrator) is then single precision, and
# the demag kernel is still computed and transformed in Float64.
#
# Run:   julia --project=<env with JuliaMag + CUDA> disk_d1=128nm_t=16nm_40mA.jl [TOTAL [TABLE [PREC]]]
#        TOTAL defaults to 40e-9 s; TABLE to disk_d1=128nm_t=16nm_40mA_table.txt;
#        PREC is Float64 (default) or Float32.
# Plots: Python/matplotlib (repo examples/plot_disk40_compare.py).

using JuliaMag
using CUDA
using Printf

CUDA.functional() || error("No functional CUDA GPU found.")
println("CUDA device: ", CUDA.name(CUDA.device())); flush(stdout)

const OUT   = @__DIR__
const TOTAL = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 40e-9
const TABLE = length(ARGS) >= 2 ? ARGS[2] : joinpath(OUT, "disk_d1=128nm_t=16nm_40mA_table.txt")
const T     = length(ARGS) >= 3 ? Dict("Float64" => Float64, "Float32" => Float32)[ARGS[3]] : Float64
println("precision: ", T); flush(stdout)

# --- Geometry (128 nm disk, 16 nm thick, 2 nm cells) -------------------------
const D     = 128e-9
const THICK = 16e-9
const CELL  = 2e-9
const NX = round(Int, D / CELL); const NY = round(Int, D / CELL); const NZ = round(Int, THICK / CELL)

const mesh = Mesh((NX, NY, NZ), (CELL, CELL, CELL))
const mat  = Material{T}(Material(Msat = 800e3, Aex = 10.8e-12, alpha = 0.01,
                                  pol = 0.37, lambda = 1.0, epsilonPrime = 0.01))  # Slonczewski params
const rp   = RegionParams(mesh, mat)
setregion!(rp, 0; Msat = 0.0)                          # empty background
setgeometry!(rp, Cylinder(D, 2 * THICK); id = 1, edgesmooth = 4) # the disk

# --- Initial vortex state, cleared, then moved to the GPU --------------------
# clearempty! is CPU-only, so clear the empty cells on the host BEFORE togpu.
m_cpu = setconfig(T, mesh, VortexConfig(mesh; circ = 1, pol = 1))
clearempty!(m_cpu, rp)

# CPU World for the host-side diagnostics; the GPU World is a device copy of it
# (togpu copies, so the two never share state).
const wcpu  = World(mesh, rp; demag = true)
const world = togpu(wcpu)

# --- Relax the vortex on the GPU ---------------------------------------------
println("Relaxing the vortex on the GPU…"); flush(stdout)
mn = Minimizer(world, togpu(m_cpu); stopdm = 1e-6); minimize!(mn; maxsteps = 20_000)
const m = mn.m
@printf("relaxed ⟨m⟩ = (%.5f, %.5f, %.5f) after %d steps%s\n",
        average_magnet(Array(m), rp)..., mn.step,
        mn.step >= 20_000 ? "  (WARNING: stopdm not reached)" : ""); flush(stdout)

# --- Current and fixed-layer polarization (mumax3 values) --------------------
const JTOT  = 0.040
const AREA  = π * D^2 / 4
const JC    = JTOT / AREA
const ANGLE = 20 * π / 180
const FIXED = (cos(ANGLE), sin(ANGLE), 0.0)
@printf("Current density Jz = %.6e A/m²\n", JC); flush(stdout)

# --- Integrate with the Slonczewski torque (fixed-step RK4, on GPU) ----------
# dt = 5e-14 s: dt/2 gives the same frequency and max |Δ⟨m⟩| < 1e-6 (validated).
# Every constant of the step is in T, so a Float32 run stays in single precision.
function rhs!(dm, mm)
    B = world._Bbuf
    effectivefield!(B, mm, world)
    torque!(dm, mm, B, mat.alpha)
    slonczewskitorque!(dm, mm, mesh, mat, JC, FIXED, THICK; add = true)
    return dm
end
const k1 = similar(m); const k2 = similar(m); const k3 = similar(m); const k4 = similar(m)
const tmp = similar(m)
function rk4!(mm, dt::T, n)
    hdt = T(0.5) * dt; dt6 = dt / T(6); two = T(2)
    for _ in 1:n
        rhs!(k1, mm); @. tmp = mm + hdt*k1; normalize!(tmp); rhs!(k2, tmp)
        @. tmp = mm + hdt*k2; normalize!(tmp); rhs!(k3, tmp)
        @. tmp = mm + dt*k3;  normalize!(tmp); rhs!(k4, tmp)
        @. mm = mm + dt6*(k1 + two*k2 + two*k3 + k4); normalize!(mm)
    end
    return mm
end

const dt      = 5e-14
const save_dt = 5e-12
const nchunk  = round(Int, TOTAL / save_dt)
const per     = max(1, round(Int, save_dt / dt))

const dmh = zeros(T, 3, NX, NY, NZ)
function maxtq(mh)   # mumax3 maxTorque: max |torque| / γ0 incl. the spin torque [T]
    B = wcpu._Bbuf
    effectivefield!(B, mh, wcpu)
    torque!(dmh, mh, B, mat.alpha)
    slonczewskitorque!(dmh, mh, mesh, mat, JC, FIXED, THICK; add = true)
    return maxtorque(dmh) / γLL
end

const tbl = open(TABLE, "w")
println(tbl, "# JuliaMag(GPU, $T) — single 128 nm × 16 nm disk, Slonczewski STT only, Jtot=40 mA, EdgeSmooth 4")
println(tbl, "# t (s)\tmx ()\tmy ()\tmz ()\tmaxTorque (T)\tE_demag (J)\tE_exch (J)\tE_Zeeman (J)\tE_total (J)\tJz (A/m2)\tcore_x (m)\tcore_y (m)")
function logrow!(t)
    mh = Array(m)                                     # host copy for the diagnostics
    a  = average_magnet(mh, rp)                       # mumax3 table m (fill-weighted)
    vx, vy, _, _ = vortexcore(mh, mesh)
    @printf(tbl, "%.6e\t%.7f\t%.7f\t%.7f\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.5e\t%.5e\n",
            t, a[1], a[2], a[3], maxtq(mh),
            demagenergy(mh, wcpu.demagplan, mesh, rp), exchangeenergy(mh, mesh, rp),
            zeemanenergy(mh, wcpu.Bext, mesh, rp), totalenergy(mh, wcpu), JC, vx, vy)
    return a, vx, vy
end

function main()
    @printf("Integrating %.3g ns with the Slonczewski current (GPU)…\n", TOTAL * 1e9); flush(stdout)
    t0 = time()
    try                                  # the table is flushed and closed even on a crash
        logrow!(0.0)
        for c in 1:nchunk
            rk4!(m, T(dt), per)
            a, vx, vy = logrow!(c * per * dt)
            if c % 200 == 0
                @printf("  t = %.1f ns  ⟨m⟩ = (%.4f, %.4f, %.4f)  core = (%.1f, %.1f) nm  wall = %.0f s\n",
                        c*per*dt*1e9, a..., vx*1e9, vy*1e9, time() - t0)
                flush(stdout); flush(tbl)
            end
        end
    finally
        flush(tbl); close(tbl)
    end
    @printf("Done. wall = %.0f s. Wrote table → %s\n", time() - t0, TABLE)
end
main()
