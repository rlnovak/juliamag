# Stoner-Wohlfarth macrospin: barrier and curvatures against the analytic values.
#   julia --project=. gpu_validation/gneb/stoner_wohlfarth.jl > gpu_validation/gneb/results/stoner_wohlfarth.txt
# One 4 nm cell, Ms = 8e5 A/m, K = 5e5 J/m^3, field -2Kh/Ms along the easy axis:
# barrier KV(1-h)^2, saddle curvatures -2K(1-h^2)/Ms and 0.
using JuliaMag, Printf
const Ms, K = 8e5, 5e5
println("# h\tbarrier (J)\tKV(1-h)^2 (J)\trel\tcurvature 1 (T)\t-2K(1-h^2)/Ms (T)\tcurvature 2 (T)\tindex")
for h in 0.0:0.1:0.9
    mesh = Mesh((1, 1, 1), (4e-9, 4e-9, 4e-9))
    w = World(mesh, Material(Msat = Ms, Aex = 1.3e-11, alpha = 0.5, Ku = K, anisU = (0, 0, 1)); demag = false,
              Bext = (0, 0, -2K * h / Ms))
    neb = GNEB(w, uniform(mesh, (0, 0, 1)), uniform(mesh, (0, 0, -1)); nimages = 24)
    relax!(neb; stopforce = 1e-9)
    s = certifysaddle(neb)
    ex = K * volume(mesh) * (1 - h)^2
    @printf("%.1f\t%.12e\t%.12e\t%+.1e\t%+.9e\t%+.9e\t%+.1e\t%d\n", h, barrier(neb), ex, barrier(neb) / ex - 1, s.values[1],
            -2K * (1 - h^2) / Ms, s.values[2], s.index)
end
