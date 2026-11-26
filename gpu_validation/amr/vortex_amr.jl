# A vortex in a disc driven by a current through two leads, with anisotropic magnetoresistance.
#
#   julia --project=. gpu_validation/amr/vortex_amr.jl DRHO OUT     (DRHO = (ρ∥ − ρ⊥)/ρ⊥, e.g. 0.02)
#
# Permalloy disc, 128 nm × 10 nm, 64 × 64 × 1 cells; non-magnetic leads 40 nm wide on both sides (ρ⊥),
# contacts at the outer ends: 2 mA in, 0 V out. Zhang-Li torque (P = 0.4, ξ = 0.05), no Oersted field.
# The vortex is relaxed, then the current is switched on for 3 ns. Table: ⟨m⟩, the core, the
# resistance between the contacts.
using JuliaMag, Printf

const DRHO = parse(Float64, ARGS[1]); const OUT = ARGS[2]; mkpath(OUT)
ρperp = 200e-9; ρpar = ρperp * (1 + DRHO)
n = 64; h = 128e-9 / n
mesh = Mesh((n + 16, n, 1), (h, h, 10e-9))                 # 8 extra columns per side for the leads
rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02, pol = 0.4, xi = 0.05))
setregion!(rp, 0; Msat = 0.0)
setgeometry!(rp, Cylinder(128e-9, 1.0); id = 1)
lead(x1, x2) = (x, y, z) -> x1 <= x < x2 && abs(y) < 20e-9
X = (n + 16) * h / 2
setregion!(rp, 4; Msat = 0.0); setregion!(rp, 5; Msat = 0.0)
defregion!(rp, 4, lead(-X, -58e-9)); defregion!(rp, 5, lead(58e-9, X))         # leads overlap the disc edge
setregion!(rp, 2; Msat = 0.0); setregion!(rp, 3; Msat = 0.0)
defregion!(rp, 2, lead(-X, -X + h)); defregion!(rp, 3, lead(X - h, X))          # contacts: the outer columns

sim = Simulation(mesh, rp; demag = true, verbose = false)
setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))
relax!(sim; stopdm = 1e-9)
setconduction!(sim; rho_par = ρpar, rho_perp = ρperp, regions = [1, 2, 3, 4, 5])
setresistivity!(sim, 2; rho_par = ρperp); setresistivity!(sim, 3; rho_par = ρperp)   # leads and contacts: isotropic
setresistivity!(sim, 4; rho_par = ρperp); setresistivity!(sim, 5; rho_par = ρperp)
addcontact!(sim, 2; current = 2e-3); addcontact!(sim, 3; potential = 0.0)
setcurrent!(sim, (0, 0, 0); zhangli = true)
tableadd!(sim, q_vortexcore()); tableadd!(sim, q_resistance()); tableautosave!(sim, 5e-12)
t = @elapsed run!(sim, 3e-9)
writetable(sim, joinpath(OUT, "table_drho$(DRHO).txt"))
c = solveconduction!(sim)
@printf("Δρ/ρ⊥ = %.3f: %.0f s, R = %.6f Ohm, last solve %d iterations\n", DRHO, t, (c.contacts[1].V - c.contacts[2].V) / c.contacts[1].I, c.iterations)
saveovf(joinpath(OUT, "j_drho$(DRHO).ovf"), currentmap(sim), mesh)
saveovf(joinpath(OUT, "phi_drho$(DRHO).ovf"), potentialmap(sim), mesh)
