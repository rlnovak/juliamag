# FMR standard problem: effect of the accuracy of the demagnetizing kernel.
#   julia --project=. gpu_validation/eigenmodes/fmr_accuracy.jl
using JuliaMag, Printf
const MU0 = 4π * 1e-7
const GRATIO = 2.210173e5 / MU0 / JuliaMag.γLL
const F = [8.270, 9.402, 10.839, 11.233, 11.992, 13.045, 13.816, 14.276, 15.316, 15.907, 16.718, 17.234, 17.457, 18.409, 19.806]
const TAU = [1.549, 1.639, 1.437, 1.452, 1.401, 1.345, 1.292, 1.253, 1.191, 1.156, 1.126, 1.094, 1.094, 1.030, 0.963]
for acc in (6.0, 12.0, 24.0, 48.0)
    mesh = Mesh((24, 24, 2), (5e-9, 5e-9, 5e-9)); B = MU0 * 8e4
    mat = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.008)
    t = @elapsed w = World(mesh, mat; accuracy = acc, Bext = (B * 0.81923192051904048, B * 0.57346234436332832, 0))
    mn = Minimizer(w, uniform(mesh, (0.82, 0.57, 0.0) ./ hypot(0.82, 0.57)); stopdm = 1e-11); minimize!(mn)
    em = eigenmodes(w, mn.m; nmodes = 15)
    f = em.frequencies .* GRATIO ./ 1e9; τ = 1e9 ./ (em.linewidths .* GRATIO)
    one = World(Mesh((1, 1, 1), (5e-9, 5e-9, 5e-9)), mat; accuracy = acc)
    n = -linearfield!(zeromag(one.mesh), uniform(one.mesh, (0, 0, 1)), one)[3] / (MU0 * 8e5)
    u = w; Bz = linearfield!(zeromag(mesh), uniform(mesh, (0, 0, 1)), w)
    nz = -sum(Bz[3, :, :, :]) / length(Bz[3, :, :, :]) / (MU0 * 8e5)
    @printf("accuracy %4.0f (kernel %.1f s): cube N %.8f  box Nz %.8f  rel f: first %+.2e, range %+.2e..%+.2e; tau %+.2e..%+.2e\n", acc, t, n, nz,
            f[1] / F[1] - 1, extrema(f ./ F .- 1)..., extrema(τ ./ TAU .- 1)...)
    println("    f [GHz]: ", join((@sprintf("%.4f", x) for x in f), " "))
    flush(stdout)
end
