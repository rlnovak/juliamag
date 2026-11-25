# Gyrotropic mode of the vortex in the 128 nm x 16 nm permalloy disc of disk_stt (64 x 64 x 8
# cells of 2 nm, staircase boundary, Ms = 8e5 A/m, A = 1.08e-11 J/m, alpha = 0.01).
#
#   julia --project=. gpu_validation/eigenmodes/disc_gyrotropic.jl OUT [cpu|gpu]
#
# Reference: free gyration of 10 mT amplitude in mumax3 and JuliaMag, f = 969.8 MHz and
# decay rate 0.0797 1/ns (gpu_validation/disk_stt, revalidation). The eigenmode is the limit
# of small amplitude. JuliaMag's gamma, 1.7595e11, is mumax3's.

using JuliaMag, Printf
const OUT = ARGS[1]
const GPU = length(ARGS) > 1 && ARGS[2] == "gpu"
if GPU
    using CUDA
end
mkpath(OUT)
const n = parse(Int, get(ENV, "NXY", "64")); const nz = parse(Int, get(ENV, "NZ", "8"))   # smaller: quick check
const NMODES = parse(Int, get(ENV, "NMODES", "6"))
mesh = Mesh((n, n, nz), (128e-9 / n, 128e-9 / n, 16e-9 / nz))
rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.08e-11, alpha = 0.01))
setregion!(rp, 0; Msat = 0.0)
setregion!(rp, 1; Msat = 8e5, Aex = 1.08e-11)
setgeometry!(rp, Cylinder(128e-9, 1.0); id = 1)
w = World(mesh, rp; demag = true)
# vortex, circulation +1, polarity +1
m = zeromag(Float64, mesh)
for k in 1:nz, j in 1:n, i in 1:n
    x, y, _ = JuliaMag.cellcenter(mesh, i, j, k)
    r = hypot(x, y); p = exp(-(r / 12e-9)^2)
    m[:, i, j, k] .= (-y / max(r, 1e-12) * sqrt(1 - p^2), x / max(r, 1e-12) * sqrt(1 - p^2), p)
end
normalize!(m)
clearempty!(m, rp)
if GPU
    w = togpu(w); m = togpu(m)
end
mn = Minimizer(w, m; stopdm = 1e-10)
t = @elapsed minimize!(mn; maxsteps = 200_000)
H = TangentHessian(w, mn.m)
@printf("%s: %d magnetic cells, relaxed in %d steps (%.0f s), max|m x B| %.1e T\n", GPU ? "GPU" : "CPU",
        count(>(0), Array(H.sw)), mn.step, t, H.torque)
flush(stdout)
saveovf(joinpath(OUT, "vortex.ovf"), Array(mn.m), mesh)
t = @elapsed em = eigenmodes(w, mn.m; nmodes = NMODES, method = :lanczos, alpha = 0.01, tol = 1e-8, cgtol = 1e-10, verbose = true)
@printf("eigenmodes: %.0f s\n# k\tf (MHz)\tGamma (1/ns)\tresidual\n", t)
for k in eachindex(em.frequencies)
    @printf("%d\t%.4f\t%.6f\t%.0e\n", k, em.frequencies[k] / 1e6, em.linewidths[k] / 1e9, em.residuals[k])
end
@printf("reference (free gyration, 10 mT): f = 969.8 MHz (rel %+.1e), Gamma = 0.0797 1/ns (rel %+.1e)\n",
        em.frequencies[1] / 969.8e6 - 1, em.linewidths[1] / 0.0797e9 - 1)
foreach(k -> savemode(joinpath(OUT, "modes"), em, k, mesh), eachindex(em.frequencies))
open(joinpath(OUT, "modes.txt"), "w") do io
    println(io, "# k\tf (Hz)\tGamma (1/s)\tresidual")
    foreach(k -> println(io, k, '\t', em.frequencies[k], '\t', em.linewidths[k], '\t', em.residuals[k]), eachindex(em.frequencies))
end
println("done")
