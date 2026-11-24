# The `reject` safeguard of the GNEB optimizer: steps to convergence, barrier and the
# hidden-maximum check, on the problems of compare_gneb.jl (JuliaMag only).
#
#   julia --project=. gpu_validation/gneb/reject_study.jl CASE N SPRING REJECT...
# CASE: ellipse_uniform, ellipse_wall, skyrmion, lattice4 (definitions as compare_gneb.jl).
using JuliaMag, Printf
using LinearAlgebra: dot, cross

const CASE = ARGS[1]
const N = parse(Int, ARGS[2]); const K = parse(Float64, ARGS[3])
const REJ = parse.(Float64, ARGS[4:end])
const MAXS = parse(Int, get(ENV, "MAXS", "40000"))

relaxed(w, m) = (mn = Minimizer(w, clearempty!(m, w.material); stopdm = 1e-9); minimize!(mn; maxsteps = 200_000); mn.m)

if startswith(CASE, "ellipse")
    const mesh = Mesh((100, 40, 1), (5e-9, 5e-9, 4e-9))
    xc(i) = (i - 0.5) * 5e-9 - 250e-9
    yc(j) = (j - 0.5) * 5e-9 - 100e-9
    rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5))
    setregion!(rp, 0; Msat = 0.0)
    setregion!(rp, 1; Msat = 8e5, Aex = 1.3e-11)
    for j in 1:40, i in 1:100
        (xc(i) / 235e-9)^2 + (yc(j) / 85e-9)^2 <= 1 && defregioncell!(rp.regions, 1, i, j, 1)
    end
    const world = World(mesh, rp; demag = true)
    function wall(x0)
        m = zeromag(Float64, mesh)
        for I in CartesianIndices(axes(m)[2:4])
            θ = π / (1 + exp(-(xc(I[1]) - x0) / 20e-9))
            m[:, I] .= (cos(θ), sin(θ), 0.0)
        end
        return normalize!(m)
    end
    const ways = CASE == "ellipse_uniform" ? [uniform(mesh, (0, 1, 0))] :
                 [wall(x0) for x0 in (200e-9, 100e-9, 0.0, -100e-9, -200e-9)]
    const states = [[relaxed(world, uniform(mesh, (1, 0, 0)))]; [clearempty!(m, world.material) for m in ways];
                    [relaxed(world, uniform(mesh, (-1, 0, 0)))]]
else
    const mesh = Mesh((60, 60, 1), (2e-9, 2e-9, 2e-9); pbc = (1, 1, 0))
    const world = World(mesh, Material(Msat = 3.84e5, Aex = 3.25e-12, alpha = 0.5, Dbulk = 5.83e-4);
                        demag = false, Bext = (0, 0, 0.12))
    function charge(m)                       # Berg-Luscher charge of the periodic layer
        nx, ny = size(m, 2), size(m, 3); q = 0.0
        tri(a, b, c) = 2atan(dot(a, cross(b, c)), 1 + dot(a, b) + dot(b, c) + dot(c, a))
        for j in 1:ny, i in 1:nx
            ip, jp = mod1(i + 1, nx), mod1(j + 1, ny)
            a, b, c, d = m[:, i, j, 1], m[:, ip, j, 1], m[:, ip, jp, 1], m[:, i, jp, 1]
            q += tri(a, b, c) + tri(a, c, d)
        end
        return q / 4π
    end
    seed = zeromag(Float64, mesh)
    for j in 1:60, i in 1:60
        x, y = i - 30.0, j - 30.0; r = hypot(x, y)
        if CASE == "skyrmion"
            θ = π / (1 + exp((r - 10) / 2))
            seed[:, i, j, 1] .= r == 0 ? (0.0, 0.0, -1.0) : (-sin(θ) * y / r, sin(θ) * x / r, cos(θ))
        else
            seed[:, i, j, 1] .= r < 10 ? (0.01, 0.0, -1.0) : (0.0, 0.0, 1.0)
        end
    end
    const states = [relaxed(world, normalize!(seed)), uniform(mesh, (0, 0, 1))]
    round(Int, charge(states[1])) == (CASE == "skyrmion" ? -1 : -4) || error("not the intended start state")
end

# compile before timing; an undone step costs two band evaluations, an accepted one one
let w = GNEB(world, states; nimages = N, spring = K)
    relax!(w; maxsteps = 5, reject = 2, stopforce = 1e-5)
end
for r in REJ
    neb = GNEB(world, states; nimages = N, spring = K)
    t = @elapsed relax!(neb; stopforce = 1e-5, maxsteps = MAXS, reject = r)
    @printf("%-15s N=%2d k=%.1f reject=%-5g steps %6d  undone %5d  evaluations %6d  max|F| %.1e T  climb %2d  barrier %.10e J  highest midpoint - highest image %+.2e J  %.0f s\n",
            CASE, N, K, r, neb.steps, neb.rejected, neb.steps + neb.rejected, neb.maxforce, neb.climb, barrier(neb), maximum(midpointenergies(neb)) - maximum(energies(neb)), t)
    println("    E-E1 (1e-20 J) ", join((@sprintf("%.3f", 1e20 * (x - neb.E[1])) for x in neb.E), " "))
    startswith(CASE, "ellipse") || println("    Q              ", join((@sprintf("%.2f", charge(m)) for m in images(neb)), " "))
    flush(stdout)
end
