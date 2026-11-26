# The problems of mumax3/*.mx3 with JuliaMag (GPU, Float32 as mumax3): fields and currents that vary
# in space and time.
#
#   julia --project=. gpu_validation/fields/fields_juliamag.jl CASE OUTDIR
#
# CASE: pulse, regions, zhangli_half, nanocontact. Writes OUTDIR/CASE/table.txt.

using JuliaMag, CUDA, Printf

const T = Float32
const CASE = ARGS[1]
const OUT = joinpath(ARGS[2], CASE)
mkpath(OUT)

# A material of Float32, with the STT constants given
mat(; Msat, Aex, alpha, pol = nothing, xi = nothing, lambda = 1.0, epsilonPrime = nothing) =
    Material{T}(Material(Msat = Msat, Aex = Aex, alpha = alpha, pol = pol, xi = xi, lambda = lambda, epsilonPrime = epsilonPrime))

function geometry(mesh, m0, shape)                 # mumax3 setgeom: cells outside are empty (fill 0)
    rp = RegionParams(mesh, m0); rp.fill .= 0
    setgeometry!(rp, shape; id = 0)
    return rp
end

function finish(sim, quantities, every, duration)
    for q in quantities
        tableadd!(sim, q)
    end
    tableautosave!(sim, every)
    run!(sim, duration)
    writetable(sim, joinpath(OUT, "table.txt"))
end

if CASE == "pulse"
    mesh = Mesh((64, 64, 1), (200e-9 / 64, 200e-9 / 64, 10e-9))
    rp = geometry(mesh, mat(Msat = 800e3, Aex = 13e-12, alpha = 0.01), Circle(200e-9))
    sim = togpu(Simulation(mesh, rp))
    setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))
    relax!(sim; mumax = true)
    pulse = fieldmap(T, mesh, (x, y, z) -> (exp(-((x - 40e-9)^2 + y^2) / (2 * 30e-9 * 30e-9)), 0, 0))
    addfield!(sim, pulse, t -> 0.02 * exp(-((t - 0.2e-9) / 50e-12)^2))
    finish(sim, (q_Bext(), q_zeemanenergy(), q_energy(), q_vortexcore()), 10e-12, 2e-9)
elseif CASE == "regions"
    mesh = Mesh((128, 64, 1), (4e-9, 4e-9, 4e-9))
    rp = RegionParams(mesh, mat(Msat = 800e3, Aex = 13e-12, alpha = 0.02))
    defregion!(rp, 1, XRange(-Inf, 0)); defregion!(rp, 2, XRange(0, Inf))
    sim = togpu(Simulation(mesh, rp))
    setmag!(sim, UniformConfig(1, 0.1, 0))
    relax!(sim; mumax = true)
    addfield!(sim, regionmap(rp, 1, (1, 0, 0)), t -> 0.01 * sin(2π * 3e9 * t))
    addfield!(sim, regionmap(rp, 1, (0, 0, 1)), 0.002)
    addfield!(sim, regionmap(rp, 2, (0, 1, 0)), t -> 0.008 * cos(2π * 5e9 * t))
    finish(sim, (q_Bext(), q_zeemanenergy(), q_energy(), q_m_region(1)), 10e-12, 1e-9)
elseif CASE == "zhangli_half"
    mesh = Mesh((32, 32, 4), (100e-9 / 32, 100e-9 / 32, 10e-9 / 4))
    rp = RegionParams(mesh, mat(Msat = 800e3, Aex = 13e-12, alpha = 0.1, pol = 1.0, xi = 0.05))
    defregion!(rp, 1, XRange(0, Inf))
    sim = togpu(Simulation(mesh, rp))
    setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))
    relax!(sim; mumax = true)
    setcurrent!(sim, (0, 0, 0); zhangli = true)
    addcurrent!(sim, regionmap(rp, 1, (1e12, 0, 0)))
    finish(sim, (q_J(), q_energy(), q_vortexcore()), 5e-12, 1e-9)
elseif CASE == "nanocontact"
    mesh = Mesh((64, 64, 1), (2e-9, 2e-9, 4e-9))
    rp = geometry(mesh, mat(Msat = 800e3, Aex = 10.8e-12, alpha = 0.01, pol = 0.37, lambda = 1.0, epsilonPrime = 0.01), Circle(128e-9))
    defregion!(rp, 1, Circle(40e-9))
    sim = togpu(Simulation(mesh, rp))
    setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))
    relax!(sim; mumax = true)
    setcurrent!(sim, (0, 0, 0); zhangli = false, slonczewski = true, fixedlayer = (cos(20π / 180), sin(20π / 180), 0))
    addcurrent!(sim, regionmap(rp, 1, (0, 0, 1e12)))
    finish(sim, (q_J(), q_energy(), q_vortexcore()), 10e-12, 2e-9)
else
    error("unknown case $CASE")
end
println("done ", CASE)
