# Disc vortex hysteresis (examples/disc_hysteresis.jl) with the field tilted in-plane
# by ARGS[1] degrees from x, breaking the y → −y mirror symmetry of the problem.
using JuliaMag, Printf
tilt = parse(Float64, ARGS[1]) * π / 180
out = ARGS[2]
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
    setexternalfield!(sim, (B * cos(tilt), B * sin(tilt), 0.0))
    relax!(sim; stopdm = 1e-6)
    savenow!(sim)
end
writetable(sim.table, out)
println("done ", out)
