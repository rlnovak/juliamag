# muMAG Standard Problem 4 — JuliaMag, with energies logged (for the mumax3
# cross-check). Same setup as examples/stdproblem4.jl; writes t, mx,my,mz and the
# four energy terms so every column can be compared against the mumax3 table.

using JuliaMag
using Printf

const OUT = get(ENV, "JM_OUT", @__DIR__)   # output directory for the table

mesh = Mesh((160, 40, 1), (3.125e-9, 3.125e-9, 3e-9))
mat  = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)

# S-state: saturate (1,1,1), minimize at zero field.
m = uniform(mesh, (1, 1, 1))
w0 = World(mesh, mat; demag = true)
mn = Minimizer(w0, m; stopdm = 1e-6); minimize!(mn; maxsteps = 20_000); m0 = mn.m
@printf("S-state ⟨m⟩ = (%.6f, %.6f, %.6f)\n", average(m0)...); flush(stdout)

# Field 1, integrate 1 ns, sample 400 points; log m + energies at each.
world = World(mesh, mat; demag = true, Bext = (-24.6e-3, 4.3e-3, 0.0))
it = Integrator(world, m0; abstol = 1e-6, reltol = 1e-5, dtmax = 5e-13, tend = 1e-9)
n = 400; Δt = 1e-9 / n

tbl = open(joinpath(OUT, "std4_juliamag_table.txt"), "w")
println(tbl, "# JuliaMag std4 — t mx my mz E_demag E_exch E_Zeeman E_total")
function logrow!()
    mh = state(it); a = average(mh)
    Ed = demagenergy(mh, world.demagplan, mesh, mat)
    Ex = exchangeenergy(mh, mesh, mat)
    Ez = zeemanenergy(mh, world.Bext, mesh, mat)
    Et = totalenergy(mh, world)
    @printf(tbl, "%.6e\t%.6f\t%.6f\t%.6f\t%.6e\t%.6e\t%.6e\t%.6e\n",
            currenttime(it), a[1], a[2], a[3], Ed, Ex, Ez, Et)
end
logrow!()
for s in 1:n
    advance!(it, Δt); logrow!()
end
close(tbl)
@printf("done. final ⟨m⟩ = (%.6f, %.6f, %.6f)\n", average(state(it))...)
