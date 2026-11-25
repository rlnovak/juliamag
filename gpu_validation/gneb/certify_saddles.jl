# Order of the saddle points of the GNEB cases of compare_gneb.jl (and the end states):
# lowest curvatures of the energy (certifysaddle, curvatures of src/eigenmodes.jl).
#
#   julia --project=. gpu_validation/gneb/certify_saddles.jl CASE OUT      (one process per case)
#
# CASE: ellipse_uniform, ellipse_wall, skyrmion, lattice4 (see compare_gneb.jl), wire
# (small, checks the script). Dense eigensolver: 2N = 5 000 to 7 200 unknowns. OUT: text
# file with the path (image, s, E) and the saddle state as OVF. Stop force 1e-7 T; the
# number of images can be given as NIMG, the spring constant as SPRING.

using JuliaMag, Printf
using LinearAlgebra: norm, cross, dot

const CASE, OUT = ARGS[1], ARGS[2]
const NIMG = parse(Int, get(ENV, "NIMG", "21"))
const SPRING = parse(Float64, get(ENV, "SPRING", "1.0"))
const STOP = 1e-7
mkpath(OUT)

function problem(case)
    if case == "wire"
        mesh = Mesh((32, 1, 1), (2e-9, 2e-9, 2e-9))
        w = World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5, Ku = 5e5, anisU = (1, 0, 0)); demag = false,
                  Bext = (-2 * 5e5 * 0.3 / 8e5, 0, 0))
        mw = zeromag(Float64, mesh)
        for i in 1:32
            θ = π / (1 + exp(-(i - 16) / 2.5))
            mw[:, i, 1, 1] .= (cos(θ), sin(θ), 0.02)
        end
        return w, [uniform(mesh, (1, 0, 0)), normalize!(mw), uniform(mesh, (-1, 0, 0))]
    elseif startswith(case, "ellipse")
        mesh = Mesh((100, 40, 1), (5e-9, 5e-9, 4e-9))
        xc(i) = (i - 0.5) * 5e-9 - 250e-9
        yc(j) = (j - 0.5) * 5e-9 - 100e-9
        rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5))
        setregion!(rp, 0; Msat = 0.0)
        setregion!(rp, 1; Msat = 8e5, Aex = 1.3e-11)
        for j in 1:40, i in 1:100
            (xc(i) / 235e-9)^2 + (yc(j) / 85e-9)^2 <= 1 && defregioncell!(rp.regions, 1, i, j, 1)
        end
        w = World(mesh, rp; demag = true)
        mA, mB = relaxed(w, uniform(mesh, (1, 0, 0))), relaxed(w, uniform(mesh, (-1, 0, 0)))
        function wall(x0)
            m = zeromag(Float64, mesh)
            for I in CartesianIndices(axes(m)[2:4])
                θ = π / (1 + exp(-(xc(I[1]) - x0) / 20e-9))
                m[:, I] .= (cos(θ), sin(θ), 0.0)
            end
            return normalize!(m)
        end
        ways = case == "ellipse_uniform" ? [uniform(mesh, (0, 1, 0))] : [wall(x) for x in (200e-9, 100e-9, 0.0, -100e-9, -200e-9)]
        return w, [[mA]; [clearempty!(m, rp) for m in ways]; [mB]]
    else
        mesh = Mesh((60, 60, 1), (2e-9, 2e-9, 2e-9); pbc = (1, 1, 0))
        w = World(mesh, Material(Msat = 3.84e5, Aex = 3.25e-12, alpha = 0.5, Dbulk = 5.83e-4); demag = false,
                  Bext = (0, 0, 0.12), openbc = true)
        m = zeromag(Float64, mesh)
        for j in 1:60, i in 1:60
            x, y = i - 30.0, j - 30.0; r = hypot(x, y)
            if case == "skyrmion"
                θ = π / (1 + exp((r - 10) / 2))
                m[:, i, j, 1] .= r == 0 ? (0.0, 0.0, -1.0) : (-sin(θ) * y / r, sin(θ) * x / r, cos(θ))
            elseif case == "lattice4"
                m[:, i, j, 1] .= r < 10 ? (0.01, 0.0, -1.0) : (0.0, 0.0, 1.0)
            else
                error("unknown case $case")
            end
        end
        mA = relaxed(w, normalize!(m))
        q = charge(mA)
        @printf("charge of the start state %+.4f\n", q)
        round(Int, q) == (case == "skyrmion" ? -1 : -4) || error("not the intended start state")
        return w, [mA, relaxed(w, uniform(mesh, (0, 0, 1)))]
    end
end

function relaxed(w, m0)
    mn = Minimizer(w, clearempty!(copy(m0), w.material); stopdm = 1e-11)
    minimize!(mn; maxsteps = 200_000)
    return mn.m
end

# Berg-Luscher charge of the periodic layer: an integer.
function charge(m)
    nx, ny = size(m, 2), size(m, 3); q = 0.0
    tri(a, b, c) = 2atan(dot(a, cross(b, c)), 1 + dot(a, b) + dot(b, c) + dot(c, a))
    for j in 1:ny, i in 1:nx
        ip, jp = mod1(i + 1, nx), mod1(j + 1, ny)
        a, b, c, d = m[:, i, j, 1], m[:, ip, j, 1], m[:, ip, jp, 1], m[:, i, jp, 1]
        q += tri(a, b, c) + tri(a, c, d)
    end
    return q / 4π
end

show6(v) = join((@sprintf("%+.6e", x) for x in v), "  ")

println("CASE ", CASE, "  images ", NIMG, "  spring ", SPRING, " T")
const world, states = problem(CASE)
for (lab, m) in (("first end", first(states)), ("last end", last(states)))
    t = @elapsed c = curvatures(world, m; nev = 8)
    @printf("%s: max|m x B| %.1e T, lowest curvatures [T]: %s  (%s, asymmetry %.0e, residuals <= %.0e, %.0f s)\n",
            lab, c.torque, show6(c.values), c.method, c.asymmetry, maximum(c.residuals), t)
    flush(stdout)
end
const neb = GNEB(world, states; nimages = NIMG, spring = SPRING)
tr = @elapsed relax!(neb; stopforce = STOP, maxsteps = 60_000)
@printf("GNEB: barrier %.10e J, max|F| %.1e T, steps %d, climbing image %d, %.0f s\n", barrier(neb), neb.maxforce,
        neb.steps, neb.climb, tr)
flush(stdout)
open(joinpath(OUT, "path_$CASE.txt"), "w") do io
    println(io, "# image\ts\tE (J)\tE-E0 (J)")
    s, E = pathcoordinate(neb), energies(neb)
    foreach(i -> println(io, i, '\t', s[i], '\t', E[i], '\t', E[i] - E[1]), eachindex(E))
end
const itop = neb.climb != 0 ? neb.climb : argmax(energies(neb))
saveovf(joinpath(OUT, "saddle_$CASE.ovf"), images(neb)[itop], world.mesh)
t = @elapsed s = certifysaddle(neb; nev = 8)
@printf("saddle (image %d): index %d, near zero %d, torque %.1e T, along the path %+.6e T (%s, %.0f s)\n", s.image,
        s.index, s.nzero, s.torque, s.along, s.method, t)
println("   curvatures [T]: ", show6(s.values), "   residuals <= ", @sprintf("%.0e", maximum(s.residuals)))
t = @elapsed l = certifysaddle(neb; nev = 4, method = :lobpcg, tol = 1e-7, maxiter = 5000)
println("   lobpcg:         ", show6(l.values), "   residuals <= ", @sprintf("%.0e  (%.0f s)", maximum(l.residuals), t))
println("done")
