# Cross-code GNEB comparison: JuliaMag against MicroMagnetic.jl (its transition/gneb.jl).
#
#   julia --project=gpu_validation/gneb/micromagnetic gpu_validation/gneb/compare_gneb.jl CASE OUT
#
# CASE  ellipse_uniform   magnum.np demo energy_barrier_ellipse: +x -> +y -> -x, a path
#                         that keeps the inversion symmetry m(r) = m(-r)
#       ellipse_wall      same element, the waypoint is a wall at x = 0 (no inversion symmetry)
#       skyrmion          film of the MicroMagnetic.jl tutorial (periodic, bulk DMI): collapse of
#                         one Bloch skyrmion, Q = -1 checked
#       lattice4          same film, a lattice of four skyrmions (Q = -4) to the uniform state
#
# Both codes get the same end states (relaxed in JuliaMag) and the same initial path.
#   1. Energy of identical states, term by term: the discretized functionals must agree
#      before barriers can be compared.
#   2. The barrier of each code, with its own force, spring and optimizer.
#   3. JuliaMag's barrier against the number of images and the spring constant.
# Text results go to stdout; images and energies to OUT (read by ellipse_magnumnp.py).

using JuliaMag, Printf, Random
using LinearAlgebra: norm, cross, dot  # not normalize!, which JuliaMag also exports
import MicroMagnetic as MM

const CASE = ARGS[1]
const OUT = ARGS[2]
const QUICK = get(ENV, "QUICK", "0") == "1"      # few steps: checks the script only
mkpath(OUT)

# --- the problem in both codes -------------------------------------------------------
if startswith(CASE, "ellipse")
    const mesh = Mesh((100, 40, 1), (5e-9, 5e-9, 4e-9))
    const Ms, Aex = 8e5, 1.3e-11
    const ea, eb = 470e-9 / 2, 170e-9 / 2
    # cell centres, as the magnum.np demo
    xc(i) = (i - 0.5) * 5e-9 - 250e-9
    yc(j) = (j - 0.5) * 5e-9 - 100e-9
    magnetic(i, j) = (xc(i) / ea)^2 + (yc(j) / eb)^2 <= 1
    rp = RegionParams(mesh, Material(Msat = Ms, Aex = Aex, alpha = 0.5))
    setregion!(rp, 0; Msat = 0.0)
    setregion!(rp, 1; Msat = Ms, Aex = Aex)
    for j in 1:40, i in 1:100
        magnetic(i, j) && defregioncell!(rp.regions, 1, i, j, 1)
    end
    const world = World(mesh, rp; demag = true)
    function mmsim()
        m = MM.FDMesh(; nx = 100, ny = 40, nz = 1, dx = 5e-9, dy = 5e-9, dz = 4e-9)
        sim = MM.Sim(m; name = "ellipse", driver = "None", save_data = false)
        MM.set_Ms(sim, (i, j, k, dx, dy, dz) -> magnetic(i, j) ? Ms : 0.0)
        MM.add_exch(sim, Aex)
        MM.add_demag(sim)
        return sim
    end
else
    const mesh = Mesh((60, 60, 1), (2e-9, 2e-9, 2e-9); pbc = (1, 1, 0))
    const Ms, Aex = 3.84e5, 3.25e-12
    const world = World(mesh, Material(Msat = Ms, Aex = Aex, alpha = 0.5, Dbulk = 5.83e-4);
                        demag = false, Bext = (0, 0, 0.12))
    function mmsim()
        m = MM.FDMesh(; nx = 60, ny = 60, nz = 1, dx = 2e-9, dy = 2e-9, dz = 2e-9, pbc = "xy")
        sim = MM.Sim(m; name = "skx", driver = "None", save_data = false)
        MM.set_Ms(sim, Ms)
        MM.add_exch(sim, Aex)
        MM.add_dmi(sim, 5.83e-4)
        MM.add_zeeman(sim, (0, 0, 120 * MM.mT))
        return sim
    end
end
const active = [msatc != 0 for msatc in JuliaMag._msatcells(world.material, mesh)]   # (1,Nx,Ny,Nz)
const VMs = cellvolume(mesh) * Ms                    # J per tesla
println("CASE ", CASE, "  magnetic cells ", count(active), "  isconservative ", isconservative(world))

# MicroMagnetic stores spins as a vector (3N), x fastest: the same order as vec(m).
# Its GNEB wants unit spins on magnetic cells; the others are zeroed here.
tovec(m) = vec(m .* active)

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

# --- 1. the energy functional --------------------------------------------------------
function julia_terms(m)
    w = world
    d = Dict("exch" => exchangeenergy(m, w.mesh, w.material),
             "total" => energy!(similar(m), m, w))
    w.demagplan === nothing || (d["demag"] = demagenergy(m, w.demagplan, w.mesh, w.material))
    JuliaMag.hasdmi(w.material) && (d["dmi"] = dmienergy(m, w.mesh, w.material))
    any(!iszero, w.Bext) && (d["zeeman"] = zeemanenergy(m, w.Bext, w.mesh, w.material))
    return d
end
const SIM = mmsim()
function mm_terms(m)
    copyto!(SIM.spin, tovec(m))
    MM.effective_field_energy(SIM, SIM.spin, 0.0)
    d = Dict{String,Float64}("total" => sum(SIM.energy))
    for it in SIM.interactions
        d[lowercase(it.name)] = sum(it.energy)
    end
    return d
end
mm_energy(m) = mm_terms(m)["total"]

println("\n[1] energies of identical states (J): JuliaMag, MicroMagnetic, relative difference")
println("    MicroMagnetic term names: ", [it.name for it in SIM.interactions])
let rng = MersenneTwister(3)
    smooth = zeromag(Float64, mesh)
    for I in CartesianIndices(axes(smooth)[2:4])
        a = 2π * I[1] / mesh.size[1]; b = 2π * I[2] / mesh.size[2]
        smooth[:, I] .= (cos(a), sin(a) * cos(b), 0.5 + 0.4sin(b))
    end
    normalize!(smooth)
    for (lab, m) in (("random", randommag!(rng, zeromag(Float64, mesh))), ("smooth", smooth))
        a, b = julia_terms(m), mm_terms(m)
        for k in sort(collect(keys(a)))
            haskey(b, k) ? (@printf("    %-7s %-7s % .12e % .12e  %+.1e\n", lab, k, a[k], b[k], (b[k] - a[k]) / abs(a[k]))) :
                           (@printf("    %-7s %-7s % .12e  (no such term name in MicroMagnetic)\n", lab, k, a[k]))
        end
    end
end

# --- end states and initial path -----------------------------------------------------
function relaxed(m0)
    mn = Minimizer(world, clearempty!(copy(m0), world.material); stopdm = 1e-9)
    minimize!(mn; maxsteps = QUICK ? 200 : 200_000)
    B = similar(mn.m); effectivefield!(B, mn.m, world)
    tq = 0.0                                          # max |m x B| on magnetic cells [T]
    for I in CartesianIndices(axes(B)[2:4])
        active[1, I] || continue
        tq = max(tq, norm(cross(mn.m[:, I], B[:, I])))
    end
    @printf("    relaxed: %d steps, max|m x B| = %.2e T, <m> = (%+.4f, %+.4f, %+.4f)\n", mn.step, tq,
            (sum(mn.m[c, I] * active[1, I] for I in CartesianIndices(axes(B)[2:4])) / count(active) for c in 1:3)...)
    return mn.m
end
println("\nend states:")
if startswith(CASE, "ellipse")
    const mA = relaxed(uniform(mesh, (1, 0, 0)))
    const mB = relaxed(uniform(mesh, (-1, 0, 0)))
    if CASE == "ellipse_uniform"
        const ways = [uniform(mesh, (0, 1, 0))]
    else
        # a wall swept from one end to the other, turning through +y in every cell
        function wall(x0)
            m = zeromag(Float64, mesh)
            for I in CartesianIndices(axes(m)[2:4])
                θ = π / (1 + exp(-(xc(I[1]) - x0) / 20e-9))
                m[:, I] .= (cos(θ), sin(θ), 0.0)
            end
            return normalize!(m)
        end
        const ways = [wall(x0) for x0 in (200e-9, 100e-9, 0.0, -100e-9, -200e-9)]
    end
    const states = [[mA]; [clearempty!(m, world.material) for m in ways]; [mB]]
else
    const mA = relaxed(filmseed(CASE))
    const mB = relaxed(uniform(mesh, (0, 0, 1)))
    @printf("    charge of the start state %+.4f, of the end state %+.4f; E[start]-E[end] = %.6e J\n",
            charge(mA), charge(mB), totalenergy(mA, world) - totalenergy(mB, world))
    round(Int, charge(mA)) == (CASE == "skyrmion" ? -1 : -4) || error("not the intended start state")
    const states = [mA, mB]
end
const NIMG = 21
const STOP = 1e-5                                     # T
const MAXS = QUICK ? 30 : 40_000

# --- 2. JuliaMag ---------------------------------------------------------------------
function run_julia(; nimages = NIMG, spring = 1.0)
    neb = GNEB(world, states; nimages = nimages, spring = spring)
    t = @elapsed relax!(neb; stopforce = STOP, maxsteps = MAXS)
    return neb, t
end
println("\n[2] JuliaMag GNEB")
const neb0 = GNEB(world, states; nimages = NIMG)
const path0 = reduce(hcat, [tovec(m) for m in images(neb0)])          # shared initial path
const neb, tj = run_julia()
const EJ = energies(neb)
const ΔJ = barrier(neb)
@printf("    N=%d k=1 T: dE = %.10e J, steps %d, max|F| = %.1e T, climbing image %d, E[end]-E[1] = %.3e J, %.0f s\n",
        NIMG, ΔJ, neb.steps, neb.maxforce, neb.climb, EJ[end] - EJ[1], tj)
open(joinpath(OUT, "juliamag_path.txt"), "w") do io
    println(io, "# image\ts\tE (J)")
    foreach(i -> println(io, i, '\t', pathcoordinate(neb)[i], '\t', EJ[i]), 1:NIMG)
end
let ms = images(neb), io = open(joinpath(OUT, "juliamag_images.bin"), "w")   # Float64 (3,Nx,Ny,Nz,N)
    foreach(m -> write(io, m .* active), ms); close(io)
end
println("    energy of the JuliaMag images in MicroMagnetic:")
const EMJ = [mm_energy(m) for m in images(neb)]
const itop = argmax(EJ)
@printf("      barrier of the same saddle image: JuliaMag %.10e, MicroMagnetic %.10e, rel %+.2e\n",
        ΔJ, EMJ[itop] - EMJ[1], (EMJ[itop] - EMJ[1]) / ΔJ - 1)
@printf("      max |E_MM - E_JM| over images / dE = %.2e (differences from image 1)\n",
        maximum(abs.((EMJ .- EMJ[1]) .- (EJ .- EJ[1]))) / ΔJ)

# --- 3. MicroMagnetic ----------------------------------------------------------------
println("\n[3] MicroMagnetic.jl GNEB (same end states, same initial path)")
function run_mm(solver)
    band = MM.GNEB(mmsim, path0; name = "", spring_constant = 1.0 * VMs, energy_scale = VMs)
    kw = (save_data_every = -1, log_every = 10^9, solver = solver)
    t = @elapsed begin
        MM.relax_gneb!(band; climbing = false, maximum_steps = MAXS, force_tolerance = 10STOP * VMs, kw...)
        c1, s1 = band.converged, band.steps
        MM.relax_gneb!(band; climbing = true, maximum_steps = MAXS, force_tolerance = STOP * VMs, kw...)
    end
    Δ = maximum(band.energies) - band.energies[1]
    @printf("    %-12s dE = %.10e J (JuliaMag %+.2e rel), converged %s/%s, steps %d+%d, max force %.1e T, climbing image %d, %.0f s\n",
            solver, Δ, Δ / ΔJ - 1, c1, band.converged, s1, band.steps, band.maximum_force / VMs, band.climbing_image, t)
    return band
end
const band = run_mm(:vp)
open(joinpath(OUT, "micromagnetic_path.txt"), "w") do io
    println(io, "# image\ts\tE (J)")
    foreach(i -> println(io, i, '\t', band.reaction_coordinate[i], '\t', band.energies[i]), 1:NIMG)
end
run_mm(:lbfgs_atlas)

# --- 4. images and spring ------------------------------------------------------------
println("\n[4] JuliaMag: barrier against the number of images and the spring constant")
for (n, k) in ((11, 1.0), (31, 1.0), (NIMG, 0.3), (NIMG, 3.0))
    b, t = run_julia(; nimages = n, spring = k)
    @printf("    N=%2d k=%.1f T: dE rel to N=%d, k=1: %+.2e, steps %d, max|F| = %.1e T, %.0f s\n",
            n, k, NIMG, barrier(b) / ΔJ - 1, b.steps, b.maxforce, t)
    flush(stdout)
end
println("done")
