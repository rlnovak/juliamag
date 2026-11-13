# Cell-by-cell torque check: evaluate the JuliaMag LL and Slonczewski torques on the
# magnetization saved by mumax3 (stt_es4_noZL_snap.mx3) and compare with mumax3's own
# LLtorque / STTorque / torque OVFs at the same instant. Separates "the two codes
# evaluate the torque differently" from "the two trajectories differ locally".
#
# Run:  julia --project=<repo> torque_on_mumax_m.jl OUTDIR IDX
#       OUTDIR = mumax3 .out directory, IDX = snapshot index (0 → 12 ns, 1 → 17 ns)

using JuliaMag, Printf

const DIR = ARGS[1]
const IDX = parse(Int, ARGS[2])
ovf(name) = loadovf(joinpath(DIR, @sprintf("%s%06d.ovf", name, IDX)))[1]

D, THICK, CELL = 128e-9, 16e-9, 2e-9
mesh = Mesh((64, 64, 8), (CELL, CELL, CELL))
mat  = Material(Msat = 800e3, Aex = 10.8e-12, alpha = 0.01,
                pol = 0.37, lambda = 1.0, epsilonPrime = 0.01)
rp   = RegionParams(mesh, mat)
setregion!(rp, 0; Msat = 0.0)
setgeometry!(rp, Cylinder(D, 2 * THICK); id = 1, edgesmooth = 4)
world = World(mesh, rp; demag = true)

# (the es4 fill was already checked identical to mumax3's geom cell by cell; loadovf
# reads vector files only, so geom000000.ovf is not re-read here)
m   = Float64.(ovf("m"))
τmx = Float64.(ovf("torque")); τLLmx = Float64.(ovf("LLtorque")); τSTmx = Float64.(ovf("STTorque"))
Bmx = Float64.(ovf("B_eff"))

JC    = 0.040 / (π * D^2 / 4)
FIXED = (cos(20π/180), sin(20π/180), 0.0)
B  = similar(m); effectivefield!(B, m, world)
τLL = zeros(size(m)); torque!(τLL, m, B, mat.alpha); τLL ./= γLL
τST = zeros(size(m)); slonczewskitorque!(τST, m, mesh, mat, JC, FIXED, THICK); τST ./= γLL
τ = τLL .+ τST

nrm(a) = dropdims(sqrt.(sum(abs2, a; dims = 1)); dims = 1)
inside = (rp.regions.id .== 1)
function report(lab, a, b)
    d = nrm(a .- b); ref = maximum(nrm(b))
    I = argmax(d .* inside)
    @printf("%-9s max|Δ| = %.3e T (%.2e of max|ref| %.3e) at %s fill %.3f\n",
            lab, d[I], d[I] / ref, ref, Tuple(I), rp.fill[I])
end
report("B_eff", B, Bmx)
report("LLtorque", τLL, τLLmx)
report("STTorque", τST, τSTmx)
report("torque", τ, τmx)
for (lab, a) in (("mumax3", τmx), ("JuliaMag", τ))
    n = nrm(a) .* inside; I = argmax(n)
    @printf("%-9s max|τ| = %.5e T at cell %s, fill %.3f, m = (%.3f, %.3f, %.3f)\n",
            lab, n[I], Tuple(I), rp.fill[I], m[1, I], m[2, I], m[3, I])
end
# where are the largest torques? top 8 cells (mumax3), with their fill
n = nrm(τmx) .* inside
for I in partialsortperm(vec(n), 1:8; rev = true)
    c = CartesianIndices(n)[I]
    @printf("  mumax3 |τ| %.4e  JM %.4e  cell %-12s fill %.3f  r = %.1f nm\n",
            n[c], nrm(τ)[c], string(Tuple(c)), rp.fill[c],
            hypot((c[1] - 32.5) * 2, (c[2] - 32.5) * 2))
end
