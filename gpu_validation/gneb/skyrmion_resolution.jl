# Too few images or a soft spring: what the band does in JuliaMag and MicroMagnetic.jl, from
# the same initial path, and whether the energy of the geodesic midpoints between
# neighbouring images shows a maximum that the images miss.
#
#   julia --project=gpu_validation/gneb/micromagnetic gpu_validation/gneb/skyrmion_resolution.jl CASE N SPRING REJECT
# CASE: skyrmion (Q = -1) or lattice4 (Q = -4); REJECT: see relax! (Inf = the default).
using JuliaMag, Printf
using LinearAlgebra: dot, cross
import MicroMagnetic as MM

const CASE = ARGS[1]
const N = parse(Int, ARGS[2]); const K = parse(Float64, ARGS[3]); const REJ = parse(Float64, ARGS[4])
const MAXS = parse(Int, get(ENV, "MAXS", "20000"))
const mesh = Mesh((60, 60, 1), (2e-9, 2e-9, 2e-9); pbc = (1, 1, 0))
const world = World(mesh, Material(Msat = 3.84e5, Aex = 3.25e-12, alpha = 0.5, Dbulk = 5.83e-4);
                    demag = false, Bext = (0, 0, 0.12))
const VMs = cellvolume(mesh) * 3.84e5
function mmsim()
    sim = MM.Sim(MM.FDMesh(; nx = 60, ny = 60, nz = 1, dx = 2e-9, dy = 2e-9, dz = 2e-9, pbc = "xy");
                 name = "skx", driver = "None", save_data = false)
    MM.set_Ms(sim, 3.84e5); MM.add_exch(sim, 3.25e-12); MM.add_dmi(sim, 5.83e-4)
    MM.add_zeeman(sim, (0, 0, 120 * MM.mT))
    return sim
end
# Start state of the periodic-film cases. "skyrmion": one Bloch skyrmion. "lattice4": what
# JuliaMag's minimizer makes of the MicroMagnetic tutorial seed (a disc of -z, charge 0):
# a lattice of four. (MicroMagnetic's SD relaxes that seed to one skyrmion.)
function filmseed(case)
    m = zeromag(Float64, mesh)
    for j in 1:60, i in 1:60
        x, y = i - 30.0, j - 30.0; r = hypot(x, y)
        if case == "skyrmion"
            θ = π / (1 + exp((r - 10) / 2))
            m[:, i, j, 1] .= r == 0 ? (0.0, 0.0, -1.0) : (-sin(θ) * y / r, sin(θ) * x / r, cos(θ))
        else
            m[:, i, j, 1] .= r < 10 ? (0.01, 0.0, -1.0) : (0.0, 0.0, 1.0)
        end
    end
    return normalize!(m)
end
# Berg-Luscher charge of the periodic layer: an exact integer.
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

mn = Minimizer(world, filmseed(CASE); stopdm = 1e-9); minimize!(mn; maxsteps = 200_000)
const mA = mn.m; const mB = uniform(mesh, (0, 0, 1))
round(Int, charge(mA)) == (CASE == "skyrmion" ? -1 : -4) || error("not the intended start state")

function show(label, ims)
    E = [totalenergy(m, world) for m in ims]
    mid = [totalenergy(JuliaMag._slerp!(similar(ims[1]), ims[i], ims[i+1], 0.5), world) for i in 1:length(ims)-1]
    top = maximum(E) - E[1]
    @printf("%s\n    barrier %.6e J; highest geodesic midpoint %.6e J\n", label, top, maximum(mid) - E[1])
    println("    E-E1 (1e-20 J) ", join((@sprintf("%7.3f", 1e20 * (x - E[1])) for x in E), " "))
    println("    Q              ", join((@sprintf("%7.2f", charge(m)) for m in ims), " "))
    flush(stdout)
end
neb = GNEB(world, [mA, mB]; nimages = N, spring = K)
const path0 = reduce(hcat, [vec(m) for m in images(neb)])
relax!(neb; stopforce = 1e-5, maxsteps = MAXS, reject = REJ)
show(@sprintf("JuliaMag %s N=%d k=%.1f reject=%g: steps %d, max|F| %.1e T, climbing image %d, Hermite maximum %.6e J, length %.1f",
              CASE, N, K, REJ, neb.steps, neb.maxforce, neb.climb, saddle(neb)[2] - neb.E[1], pathcoordinate(neb)[end]), images(neb))
if REJ == Inf
    band = MM.GNEB(mmsim, path0; name = "", spring_constant = K * VMs, energy_scale = VMs)
    kw = (save_data_every = -1, log_every = 10^9, solver = :vp, maximum_steps = MAXS)
    MM.relax_gneb!(band; climbing = false, force_tolerance = 1e-4 * VMs, kw...)
    s1, c1 = band.steps, band.converged
    MM.relax_gneb!(band; climbing = true, force_tolerance = 1e-5 * VMs, kw...)
    show(@sprintf("MicroMagnetic %s N=%d k=%.1f: steps %d+%d, converged %s/%s, max|F| %.1e T, climbing image %d", CASE, N, K,
                  s1, band.steps, c1, band.converged, band.maximum_force / VMs, band.climbing_image),
         [reshape(band.images[:, i], 3, 60, 60, 1) for i in 1:N])
end
