# Shift of the gyrotropic frequency by the damping, exactly: the eigenvalues of the damped
# operator gamma/(1+alpha^2) (J - alpha) A of a coarse disc (NXY x NXY x NZ cells, default 28 x 28 x 1), against the
# undamped frequency and the first-order linewidth.
#   julia --project=. gpu_validation/eigenmodes/damped_shift.jl
using JuliaMag, Printf
using LinearAlgebra: eigvals
n = parse(Int, get(ENV, "NXY", "28")); nz = parse(Int, get(ENV, "NZ", "1"))
mesh = Mesh((n, n, nz), (128e-9 / n, 128e-9 / n, 16e-9 / nz))
rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.08e-11, alpha = 0.01))
setregion!(rp, 0; Msat = 0.0); setregion!(rp, 1; Msat = 8e5, Aex = 1.08e-11)
setgeometry!(rp, Cylinder(128e-9, 1.0); id = 1)
w = World(mesh, rp; demag = true)
m = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1)); clearempty!(m, rp)
mn = Minimizer(w, m; stopdm = 1e-11); minimize!(mn; maxsteps = 100_000)
em = eigenmodes(w, mn.m; nmodes = 1, method = :dense)
A, = densehessian(TangentHessian(w, mn.m))
JA = JuliaMag._Jrows!(similar(A), A)
γ = JuliaMag.γLL
@printf("%d unknowns; undamped f = %.6f MHz, first-order Gamma = %.6f 1/ns\n", size(A, 1), em.frequencies[1] / 1e6, em.linewidths[1] / 1e9)
f1 = em.frequencies[1]
for α in (0.005, 0.01, 0.02)
    ev = eigvals(γ / (1 + α^2) .* (JA .- α .* A))
    ev = ev[imag.(ev).>0]; z = ev[argmin(abs.(imag.(ev) ./ 2π .- f1))]
    s = em.linewidths[1] * (1 + 0.01^2) / 0.01 / γ / (2π * f1 / γ)
    @printf("alpha %.3f: f = %.6f MHz (rel %+.3e, over alpha^2 %+.3f), Gamma = %.6f 1/ns (rel to first order %+.1e); one-mode estimate %+.3f alpha^2\n",
            α, imag(z) / 2π / 1e6, imag(z) / 2π / f1 - 1, (imag(z) / 2π / f1 - 1) / α^2, -real(z) / 1e9,
            -real(z) / (em.linewidths[1] * (α / (1 + α^2)) / (0.01 / (1 + 0.01^2))) - 1, -1 - (s^2 - 1) / 2)
end
