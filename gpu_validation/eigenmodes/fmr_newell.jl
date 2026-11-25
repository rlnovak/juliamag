# FMR standard problem with Newell's demagnetizing tensor (as the paper), 5 nm cells.
#   julia --project=. gpu_validation/eigenmodes/fmr_newell.jl
using JuliaMag, Printf
include(joinpath(@__DIR__, "newell.jl"))
const MU0 = 4π * 1e-7
const GRATIO = 2.210173e5 / MU0 / JuliaMag.γLL
const F = [8.270, 9.402, 10.839, 11.233, 11.992, 13.045, 13.816, 14.276, 15.316, 15.907, 16.718, 17.234, 17.457, 18.409, 19.806]
const TAU = [1.549, 1.639, 1.437, 1.452, 1.401, 1.345, 1.292, 1.253, 1.191, 1.156, 1.126, 1.094, 1.094, 1.030, 0.963]
mat = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.008)
# checks of the tensor: cube 1/3; trace 1 at the origin and 0 elsewhere; a flat cell against Aharoni's formula through the sum
for cs in ((5e-9, 5e-9, 5e-9), (4e-9, 6e-9, 2e-9))
    k = newellkernel(Mesh((4, 3, 2), cs))
    tr = k.Kxx .+ k.Kyy .+ k.Kzz
    @printf("cells %s: N(0) = %.12f %.12f %.12f, trace(0)+1 = %.1e, max|trace| elsewhere %.1e\n", cs, -k.Kxx[1], -k.Kyy[1], -k.Kzz[1], tr[1] + 1,
            maximum(abs, [tr[i, j, l] for i in (1, 2, 3, 4), j in 1:3, l in 1:2 if (i, j, l) != (1, 1, 1)]))
end
mesh = Mesh((24, 24, 2), (5e-9, 5e-9, 5e-9)); B = MU0 * 8e4
t = @elapsed w = usenewell!(World(mesh, mat; Bext = (B * 0.81923192051904048, B * 0.57346234436332832, 0)))
# against JuliaMag's kernel: largest difference per component
kj = demagkernel(Float64, mesh); kn = newellkernel(mesh)
println("JuliaMag (accuracy 6) - Newell, largest |difference|: ", [(f, maximum(abs, getfield(kj, f) .- getfield(kn, f))) for f in (:Kxx, :Kyy, :Kzz, :Kxy, :Kxz, :Kyz)])
mn = Minimizer(w, uniform(mesh, (0.82, 0.57, 0.0) ./ hypot(0.82, 0.57)); stopdm = 1e-11); minimize!(mn)
em = eigenmodes(w, mn.m; nmodes = 15)
f = em.frequencies .* GRATIO ./ 1e9; τ = 1e9 ./ (em.linewidths .* GRATIO)
Bz = linearfield!(zeromag(mesh), uniform(mesh, (0, 0, 1)), w)
println("box Nz ", -sum(Bz[3, :, :, :]) / length(Bz[3, :, :, :]) / (MU0 * 8e5), "   kernel ", round(t), " s")
for k in 1:15
    @printf("%2d  f %.6f  paper %.3f  rel %+.2e   tau %.5f  paper %.3f  rel %+.2e\n", k, f[k], F[k], f[k] / F[k] - 1, τ[k], TAU[k], τ[k] / TAU[k] - 1)
end
