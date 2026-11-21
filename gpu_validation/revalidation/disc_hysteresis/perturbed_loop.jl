# Disc vortex hysteresis with an infinitesimal, seeded symmetry-breaking
# perturbation (amplitude ARGS[1], seed ARGS[2]) added before each relaxation.
using JuliaMag, Printf, Random
eps = parse(Float64, ARGS[1]); rng = MersenneTwister(parse(Int, ARGS[2])); out = ARGS[3]
mesh = Mesh((100, 100, 4), (5e-9, 5e-9, 5e-9))
rp   = RegionParams(mesh, material("Permalloy"))
setregion!(rp, 0; Msat = 0.0)
defregion!(rp, 1, Cylinder(500e-9, 1e6))
sim = Simulation(mesh, rp; demag = true, verbose = false)
setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))
savequantities!(sim, q_Bext())
Bmax, dB = 0.15, 0.005
Bs = vcat(0:dB:Bmax, Bmax:-dB:-Bmax, -Bmax:dB:Bmax)
for B in Bs
    setexternalfield!(sim, (B, 0.0, 0.0))
    m = copy(sim.m); m .+= eps .* randn(rng, size(m)); normalize!(m); setmag!(sim, m)
    relax!(sim; stopdm = 1e-6)
    savenow!(sim)
end
writetable(sim.table, out)
