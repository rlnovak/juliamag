# A Permalloy disc: in-plane hysteresis loop of a magnetic vortex.
#
# A vortex sits at the centre of a disc in zero field. An in-plane field along x
# displaces the core sideways, raising ⟨mx⟩; sweeping the field up and down and
# back traces the characteristic (narrow, nearly reversible) vortex hysteresis
# loop, with ⟨mx⟩ = 0 at the vortex state and saturating toward ±1 at high field.
#
# The disc is non-rectangular geometry on a rectangular mesh: cells outside the
# cylinder are empty (Msat = 0). The loop is computed with the energy minimizer,
# starting each field step from the previous state so the vortex is tracked
# continuously.
#
# Run:  julia --project=examples examples/disc_hysteresis.jl

using JuliaMag
using Printf
using Plots

function main()
    mesh = Mesh((64, 64, 1), (5e-9, 5e-9, 5e-9))       # 320 × 320 nm mesh
    rp   = RegionParams(mesh, material("Permalloy"))
    setregion!(rp, 0; Msat = 0.0)                       # background empty
    defregion!(rp, 1, Cylinder(300e-9, 1e6))           # 300 nm Permalloy disc
    sim = Simulation(mesh, rp; demag = true)

    setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))  # seed a vortex

    # Field sweep along x: 0 → +Bmax → −Bmax → +Bmax [T].
    Bmax, dB = 0.08, 0.005
    Bs = vcat(0:dB:Bmax, Bmax:-dB:-Bmax, -Bmax:dB:Bmax)

    mxs = Float64[]
    println("Sweeping the in-plane field along x…"); flush(stdout)
    for B in Bs
        JuliaMag.setexternalfield!(sim.world, (B, 0.0, 0.0))
        relax!(sim; stopdm = 1e-6)
        push!(mxs, average(sim.m)[1])
    end

    xc, yc, _, _ = vortexcore(sim.m, mesh)
    @printf("Done. ⟨mx⟩ range [%.3f, %.3f]; final core at (%.1f, %.1f) nm\n",
            minimum(mxs), maximum(mxs), xc * 1e9, yc * 1e9)

    # Hysteresis curve.
    plt = plot(Bs .* 1e3, mxs; xlabel = "µ₀Hx (mT)", ylabel = "⟨mx⟩",
               title = "Vortex hysteresis in a 300 nm Permalloy disc",
               titlefontsize = 10, legend = false, lw = 2, marker = :circle, ms = 2)
    out = joinpath(@__DIR__, "disc_hysteresis.png")
    savefig(plt, out)
    println("Wrote figure → ", out)
end

main()
