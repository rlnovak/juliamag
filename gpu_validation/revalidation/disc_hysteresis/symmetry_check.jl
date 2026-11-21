# Diagnostic: is the descending-branch state of the disc hysteresis exactly mirror
# symmetric at -10 mT, and does a tiny symmetry-breaking perturbation make a vortex
# nucleate at -15 mT (as in the pre-rewrite run)?
using JuliaMag, Printf, Random

mesh = Mesh((100, 100, 4), (5e-9, 5e-9, 5e-9))
rp   = RegionParams(mesh, material("Permalloy"))
setregion!(rp, 0; Msat = 0.0)
defregion!(rp, 1, Cylinder(500e-9, 1e6))
sim = Simulation(mesh, rp; demag = true, verbose = false)
setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))

# Mirror asymmetries of a state: max |m − S(m)| for the four reflections/products.
function asym(m)
    Nx, Ny = size(m, 2), size(m, 3)
    rx(a) = a[:, end:-1:1, :, :]; ry(a) = a[:, :, end:-1:1, :]
    out = Dict{String,Float64}()
    for (nm, op, sg) in (("y→−y, (mx,−my,mz)", ry, (1, -1, 1)), ("y→−y, (−mx,my,−mz)", ry, (-1, 1, -1)),
                         ("x→−x, (mx,−my,mz)", rx, (1, -1, 1)), ("x→−x, (−mx,my,−mz)", rx, (-1, 1, -1)),
                         ("C2 (x,y)→(−x,−y), (mx,my,mz)", a -> rx(ry(a)), (1, 1, 1)),
                         ("C2 (x,y)→(−x,−y), (−mx,−my,mz)", a -> rx(ry(a)), (-1, -1, 1)))
        s = op(m)
        out[nm] = maximum(abs, m .- s .* reshape(collect(sg), 3, 1, 1, 1))
    end
    out
end

Bmax, dB = 0.15, 0.005
Bs = vcat(0:dB:Bmax, Bmax:-dB:-0.010)
for B in Bs
    setexternalfield!(sim, (B, 0.0, 0.0)); relax!(sim; stopdm = 1e-6)
end
m10 = copy(sim.m)
@printf("state at -10 mT: <m> = (%.4f, %.4f, %.4f)\n", average(sim)...)
for (k, v) in sort(collect(asym(m10))); @printf("  asym %-34s %.3e\n", k, v); end

for (label, eps) in (("unperturbed", 0.0), ("perturbed 1e-6", 1e-6), ("perturbed 1e-3", 1e-3))
    m = copy(m10)
    if eps > 0
        rng = MersenneTwister(1)
        m .+= eps .* randn(rng, size(m)); normalize!(m); clearempty!(m, rp)
    end
    setmag!(sim, m); setexternalfield!(sim, (-0.015, 0.0, 0.0)); relax!(sim; stopdm = 1e-6)
    a = average(sim)
    @printf("%-16s at -15 mT: <m> = (%.4f, %.4f, %.4f)\n", label, a...)
end
