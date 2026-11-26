# The problem of fem_reference.py with JuliaMag: R and the potential at the cell centres.
#   julia --project=. gpu_validation/amr/juliamag_texture.jl OUT [wall|uniform45]
using JuliaMag, Printf

const L, W, t = 400e-9, 80e-9, 4e-9
const ρpar, ρperp, δ = 210e-9, 200e-9, 15e-9
OUT = ARGS[1]; mkpath(OUT)
TEXTURE = length(ARGS) > 1 ? ARGS[2] : "wall"; tag = TEXTURE == "wall" ? "" : TEXTURE * "_"
for (nx, ny) in ((50, 10), (100, 20), (200, 40), (400, 80))
    h = L / nx
    mesh = Mesh((nx + 2, ny, 1), (h, W / ny, t))            # one contact column at each end, outside the sample
    rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1e-11, alpha = 0.1))
    defregion!(rp, 1, XRange(-Inf, -L / 2)); defregion!(rp, 2, XRange(L / 2, Inf))
    w = World(mesh, rp; demag = false)
    setconduction!(w; rho_par = ρpar, rho_perp = ρperp)
    addcontact!(w, 1; potential = 1.0); addcontact!(w, 2; potential = 0.0)
    m = TEXTURE == "uniform45" ? uniform(mesh, (1, 1, 0) ./ sqrt(2)) :
        setconfig(mesh, (x, y, z) -> (θ = 2atan(exp((x - 0.3y) / δ)); (cos(θ), sin(θ), 0.0)))
    c = solveconduction!(w, m)
    R = 1 / c.contacts[1].I
    @printf("cells %dx%d: R = %.9f Ohm, I2/I1 + 1 = %.1e, %d iterations\n", nx, ny, R, c.contacts[2].I / c.contacts[1].I + 1, c.iterations)
    open(joinpath(OUT, "jm_$(tag)$(nx)x$(ny).txt"), "w") do io
        println(io, "# R ", R)
        for j in 1:ny
            println(io, join([c.phi[1, i + 1, j, 1] for i in 1:nx], " "))
        end
    end
end
