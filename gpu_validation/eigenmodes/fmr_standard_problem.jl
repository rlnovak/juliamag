# FMR standard problem: Baker et al., J. Magn. Magn. Mater. 421, 428 (2017), against the
# eigenvalue method of that paper (Table II, 5 x 5 x 5 nm cells, finite differences).
#
#   julia --project=. gpu_validation/eigenmodes/fmr_standard_problem.jl OUT
#
# 120 x 120 x 10 nm permalloy, Ms = 8e5 A/m, A = 13 pJ/m, no anisotropy, 80 kA/m in plane at
# 35 degrees from x, damping 0.008. The paper uses gamma = 2.210173e5 m/(A s): the
# frequencies are scaled from JuliaMag's gamma by the ratio. Writes OUT/eigenmodes.txt and
# OUT/modes (OVF). Dense and matrix-free solvers, and cells of 5 and 2.5 nm (matrix-free).

using JuliaMag, Printf

const OUT = ARGS[1]
const FINE = get(ENV, "FINE", "1") == "1"
mkpath(OUT)
const MU0 = 4π * 1e-7
const GRATIO = 2.210173e5 / MU0 / JuliaMag.γLL              # paper's gamma / JuliaMag's
const TABLE = [8.270 1.549; 9.402 1.639; 10.839 1.437; 11.233 1.452; 11.992 1.401; 13.045 1.345; 13.816 1.292;
               14.276 1.253; 15.316 1.191; 15.907 1.156; 16.718 1.126; 17.234 1.094; 17.457 1.094; 18.409 1.030;
               19.806 0.963]                                 # f [GHz], damping time [ns]
const ALPHA = 0.008

function equilibrium(n)                                      # n: number of cells per 5 nm
    mesh = Mesh((24n, 24n, 2n), (5e-9 / n, 5e-9 / n, 5e-9 / n))
    B = MU0 * 8e4
    w = World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = ALPHA);
              Bext = (B * 0.81923192051904048, B * 0.57346234436332832, 0))
    mn = Minimizer(w, uniform(mesh, (0.82, 0.57, 0.0) ./ hypot(0.82, 0.57)); stopdm = 1e-11)
    t = @elapsed minimize!(mn; maxsteps = 100_000)
    H = TangentHessian(w, mn.m)
    @printf("cells %s of %.2f nm: equilibrium in %d steps (%.0f s), max|m x B| %.1e T, <m> = (%.6f, %.6f, %.6f)\n",
            string(mesh.size), 5 / n, mn.step, t, H.torque, average(mn.m)...)
    flush(stdout)
    return w, mn.m
end

function report(em, io)
    @printf(io, "# k\tf (GHz)\tf paper\trel\ttau (ns)\ttau paper\trel\tresidual\n")
    for k in eachindex(em.frequencies)
        f = em.frequencies[k] * GRATIO / 1e9
        τ = 1e9 / (em.linewidths[k] * GRATIO)
        if k <= size(TABLE, 1)
            @printf(io, "%d\t%.6f\t%.3f\t%+.1e\t%.5f\t%.3f\t%+.1e\t%.0e\n", k, f, TABLE[k, 1], f / TABLE[k, 1] - 1, τ,
                    TABLE[k, 2], τ / TABLE[k, 2] - 1, em.residuals[k])
        else
            @printf(io, "%d\t%.6f\t-\t-\t%.5f\t-\t-\t%.0e\n", k, f, τ, em.residuals[k])
        end
    end
end

w, m = equilibrium(1)
t = @elapsed em = eigenmodes(w, m; nmodes = 20)
println("dense (", em.method, "), ", round(t; digits = 1), " s; frequencies scaled by gamma ratio ", GRATIO)
report(em, stdout)
open(io -> report(em, io), joinpath(OUT, "eigenmodes.txt"), "w")
foreach(k -> savemode(joinpath(OUT, "modes"), em, k, w.mesh), 1:8)
saveovf(joinpath(OUT, "equilibrium.ovf"), m, w.mesh)
t = @elapsed el = eigenmodes(w, m; nmodes = 15, method = :lanczos)
@printf("lanczos, %.1f s: largest |rel. difference| to dense: f %.1e, linewidth %.1e, residual %.0e\n", t,
        maximum(abs.(el.frequencies ./ em.frequencies[1:15] .- 1)), maximum(abs.(el.linewidths ./ em.linewidths[1:15] .- 1)),
        maximum(el.residuals))
flush(stdout)
if FINE
    w2, m2 = equilibrium(2)
    t = @elapsed e2 = eigenmodes(w2, m2; nmodes = 15, method = :lanczos, tol = 1e-7)
    @printf("lanczos, 2.5 nm cells, %.1f s\n", t)
    report(e2, stdout)
    open(io -> report(e2, io), joinpath(OUT, "eigenmodes_2.5nm.txt"), "w")
end
println("done")
