# A Permalloy disc: in-plane hysteresis loop of a magnetic vortex.
#
# A vortex sits at the centre of a disc in zero field. An in-plane field along x
# displaces the core sideways, raising ⟨mx⟩ along a nearly linear reversible
# branch; at a critical field the vortex annihilates and the disc reaches its
# saturated branch (⟨mx⟩ ≈ ±0.78 — a thin disc never magnetizes fully in-plane,
# shape demag keeps the edge spins curled). Reducing the field renucleates the
# vortex on the return branch, giving the S-shaped vortex hysteresis loop.
#
# A well-formed vortex loop needs a disc that is wide and thin: here 500 nm
# diameter × 20 nm thick (a classic experimental vortex size). A small, thin disc
# gives a nearly square loop instead — the reversible core-displacement branch is
# what distinguishes the vortex loop, and it needs room for the core to move.
#
# The field is swept over ±150 mT, well past the annihilation field, so the disc
# reaches its saturated branch and the plateau is flat at both ends.
#
# The disc is non-rectangular geometry on a rectangular mesh: cells outside the
# cylinder are empty (Msat = 0). The loop is computed with the energy minimizer,
# starting each field step from the previous state so the vortex is tracked
# continuously. One table row (Bx, mx, my, mz) is written per field step.
#
# Run:  julia --project=examples examples/disc_hysteresis.jl

using JuliaMag
using Printf
using Plots

function main()
    mesh = Mesh((100, 100, 4), (5e-9, 5e-9, 5e-9))     # 500 × 500 × 20 nm mesh
    rp   = RegionParams(mesh, material("Permalloy"))
    setregion!(rp, 0; Msat = 0.0)                       # background empty
    defregion!(rp, 1, Cylinder(500e-9, 1e6))           # 500 nm Permalloy disc
    sim = Simulation(mesh, rp; demag = true)

    setmag!(sim, VortexConfig(mesh; circ = 1, pol = 1))  # seed a vortex

    # Save Bx and ⟨m⟩ once per field step: columns t, mx, my, mz, B_extx,y,z.
    savequantities!(sim, q_Bext())

    # Field sweep along x: 0 → +Bmax → −Bmax → +Bmax [T]; ±150 mT saturates.
    Bmax, dB = 0.15, 0.005
    Bs = vcat(0:dB:Bmax, Bmax:-dB:-Bmax, -Bmax:dB:Bmax)

    Bxs = Float64[]; mxs = Float64[]
    println("Sweeping the in-plane field along x…"); flush(stdout)
    for B in Bs
        JuliaMag.setexternalfield!(sim.world, (B, 0.0, 0.0))
        relax!(sim; stopdm = 1e-6)
        savenow!(sim)                                   # one table row per step
        push!(Bxs, B); push!(mxs, average(sim.m)[1])
    end

    # Write the output table (t, mx, my, mz, B_extx, B_exty, B_extz).
    tblpath = joinpath(@__DIR__, "disc_hysteresis.txt")
    writetable(sim.table, tblpath)
    println("Wrote table  → ", tblpath)

    xc, yc, _, _ = vortexcore(sim.m, mesh)
    @printf("Done. ⟨mx⟩ range [%.3f, %.3f]; final core at (%.1f, %.1f) nm\n",
            minimum(mxs), maximum(mxs), xc * 1e9, yc * 1e9)

    # Hysteresis curve.
    plt = plot(Bxs .* 1e3, mxs; xlabel = "µ₀Hx (mT)", ylabel = "⟨mx⟩",
               title = "Vortex hysteresis in a 500 nm × 20 nm Permalloy disc",
               titlefontsize = 10, legend = false, lw = 2, marker = :circle, ms = 2)
    out = joinpath(@__DIR__, "disc_hysteresis.png")
    savefig(plt, out)
    println("Wrote figure → ", out)
end

main()
