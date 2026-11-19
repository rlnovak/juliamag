# GPU cost breakdown of the 40 ns disk driver (disk_d1=128nm_t=16nm_40mA.jl):
# time per call of each term of the right-hand side, of one RK4 step, and of one
# host-side table row, in Float64 and Float32; GPU allocations per RHS; and the
# device-side kernel profile of one RHS.
#
# Run:  julia --project=<env with JuliaMag + CUDA> bench_rhs.jl

using JuliaMag, CUDA, Printf, Statistics
using LinearAlgebra: mul!

CUDA.functional() || error("no CUDA GPU")
println("device: ", CUDA.name(CUDA.device()))

const D, THICK, CELL = 128e-9, 16e-9, 2e-9

function setup(T)
    mesh = Mesh((64, 64, 8), (CELL, CELL, CELL))
    mat  = Material(Msat = T(800e3), Aex = T(10.8e-12), alpha = T(0.01),
                    pol = T(0.37), lambda = T(1.0), epsilonPrime = T(0.01))
    rp   = RegionParams(mesh, mat)
    setregion!(rp, 0; Msat = zero(T))
    setgeometry!(rp, Cylinder(D, 2 * THICK); id = 1, edgesmooth = 4)
    m_cpu = setconfig(T, mesh, VortexConfig(mesh; circ = 1, pol = 1))
    clearempty!(m_cpu, rp)
    wcpu = World(mesh, rp; demag = true)
    return mesh, mat, rp, wcpu, togpu(wcpu), m_cpu, togpu(m_cpu)
end

# median wall time [s] of f() over n calls, GPU synchronized
function tmed(f, n)
    f(); CUDA.synchronize()                        # warm-up / compile
    ts = Float64[]
    for _ in 1:n
        t0 = time_ns(); f(); CUDA.synchronize(); push!(ts, (time_ns() - t0) * 1e-9)
    end
    return median(ts)
end

const JC    = 0.040 / (π * D^2 / 4)
const FIXED = (cos(20π/180), sin(20π/180), 0.0)

for T in (Float64, Float32)
    mesh, mat, rp, wcpu, world, m_cpu, m = setup(T)
    p = world.material; plan = world.demagplan
    B = world._Bbuf; dm = similar(m); k1 = similar(m); k2 = similar(m); k3 = similar(m)
    k4 = similar(m); tmp = similar(m)
    rhs!(d, mm) = (effectivefield!(B, mm, world); torque!(d, mm, B, mat.alpha);
                   slonczewskitorque!(d, mm, mesh, mat, JC, FIXED, THICK; add = true); d)
    # step constants in T: a Float64 literal (0.5dt) would promote the Float32
    # broadcast kernels to FP64 and falsify the Float32 timing
    dt = T(5e-14); hdt = T(0.5) * dt; dt6 = dt / T(6); two = T(2)
    function rk4!(mm)
        rhs!(k1, mm); @. tmp = mm + hdt*k1; normalize!(tmp); rhs!(k2, tmp)
        @. tmp = mm + hdt*k2; normalize!(tmp); rhs!(k3, tmp)
        @. tmp = mm + dt*k3;  normalize!(tmp); rhs!(k4, tmp)
        @. mm = mm + dt6*(k1 + two*k2 + two*k3 + k4); normalize!(mm)
    end
    copyto!(tmp, m)
    n = 200
    r = Dict{String,Float64}()
    r["exchange"]      = tmed(() -> JuliaMag.exchange!(B, m, mesh, p; add = false), n)
    r["demag"]         = tmed(() -> JuliaMag._demag!(B, m, world; add = true), n)
    if hasproperty(plan, :bout)            # batched plan (3 components per transform)
        Ext = Base.get_extension(JuliaMag, :JuliaMagCUDAExt)
        r["  rFFT ×3"]  = tmed(() -> mul!(plan.mhat, plan.pfor, plan.padm), n)
        r["  irFFT ×3"] = tmed(() -> Ext._exec_c2r!(plan.bout, plan.pinv, plan.mhat), n)
    else                                   # pre-optimization plan, one component
        r["  rFFT ×3"]  = 3 * tmed(() -> mul!(plan.mhat[1], plan.pfor, plan.padm[1]), n)
        r["  irFFT ×3"] = 3 * tmed(() -> mul!(plan.padm[1], plan.pinv, plan.bhat[1]), n)
    end
    r["effectivefield"] = tmed(() -> effectivefield!(B, m, world), n)
    r["torque (LLG)"]  = tmed(() -> torque!(dm, m, B, mat.alpha), n)
    r["slonczewski"]   = tmed(() -> slonczewskitorque!(dm, m, mesh, mat, JC, FIXED, THICK; add = true), n)
    r["normalize"]     = tmed(() -> normalize!(tmp), n)
    r["RHS total"]     = tmed(() -> rhs!(dm, m), n)
    mm = copy(m)
    r["RK4 step"]      = tmed(() -> rk4!(mm), 50)

    # host-side table row, exactly as logrow! in the driver (CPU World)
    dmh = zeros(T, size(m_cpu))
    function row()
        mh = Array(m)
        a = average_magnet(mh, rp); vx, vy, _, _ = vortexcore(mh, mesh)
        Bh = wcpu._Bbuf; effectivefield!(Bh, mh, wcpu); torque!(dmh, mh, Bh, mat.alpha)
        slonczewskitorque!(dmh, mh, mesh, mat, JC, FIXED, THICK; add = true)
        mt = maxtorque(dmh)
        e = (demagenergy(mh, wcpu.demagplan, mesh, rp), exchangeenergy(mh, mesh, rp),
             zeemanenergy(mh, wcpu.Bext, mesh, rp), totalenergy(mh, wcpu))
        return a, vx, mt, e
    end
    r["table row (CPU)"] = tmed(row, 10)

    println("\n=== ", T, "  (", Threads.nthreads(), " Julia threads) ===")
    for k in ("exchange", "demag", "  rFFT ×3", "  irFFT ×3", "effectivefield", "torque (LLG)",
              "slonczewski", "normalize", "RHS total", "RK4 step", "table row (CPU)")
        @printf("  %-16s %9.3f ms\n", k, r[k] * 1e3)
    end
    steps_per_row = 100                                  # 5 ps / 50 fs
    per_ns = 200 * (steps_per_row * r["RK4 step"] + r["table row (CPU)"])
    @printf("  predicted wall per ns: %.0f s  (RK4 %.0f s + table rows %.0f s)\n",
            per_ns, 200 * steps_per_row * r["RK4 step"], 200 * r["table row (CPU)"])

    # GPU allocations of one RHS
    rhs!(dm, m); CUDA.synchronize()
    st = CUDA.@timed rhs!(dm, m)
    @printf("  one RHS: %d GPU allocations, %.1f MiB\n",
            st.gpu_memstats.alloc_count, st.gpu_memstats.alloc_bytes / 2^20)
    if T == Float64
        println("\n--- device-side profile of one RHS (Float64) ---")
        try
            prof = CUDA.@profile trace = false rhs!(dm, m)
            show(stdout, prof); println()
        catch err
            println("  (profiler unavailable: ", sprint(showerror, err), ")")
        end
    end
end
