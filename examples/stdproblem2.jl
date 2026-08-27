# µMAG Standard Problem 2 — remanent magnetization vs. particle size.
# https://www.ctcms.nist.gov/~rdm/std2/spec2.html
#
# A rectangular prism with aspect ratio L : d : t = 5 : 1 : 0.1, parametrized by
# the reduced size d/lex, where lex = sqrt(A / Km) is the magnetostatic exchange
# length and Km = ½ μ0 Ms². The sample is saturated along the body diagonal
# [1,1,1], the field is removed, and the state is relaxed to remanence. The
# reported quantities are the remanent ⟨mx⟩ (along the long axis L) and ⟨my⟩
# (along d), as functions of d/lex.
#
# This script sweeps a representative set of d/lex values and compares the
# remanent magnetization against the OOMMF reference table (stdprob2_oommf.txt).
#
# Run:  julia --project=examples examples/stdproblem2.jl

using JuliaMag
using Printf
include(joinpath(@__DIR__, "makie_shim.jl"))

const Msat = 8.0e5
const Aex  = 1.3e-11
const lex  = sqrt(Aex / (0.5 * JuliaMag.μ0 * Msat^2))   # magnetostatic exchange length

# Relax the remanent state for a given d/lex and return (mx, my).
function remanence(dlex; cells_per_lex = 2.0)
    d = dlex * lex                    # width
    L = 5 * d                         # length (long axis, x)
    t = 0.1 * d                       # thickness (z)

    # Mesh: keep cells ≲ lex, and at least a few along each axis.
    cx = lex / cells_per_lex
    Nx = max(8, round(Int, L / cx))
    Ny = max(4, round(Int, d / cx))
    Nz = 1                            # thin film: single layer along t
    mesh = Mesh((Nx, Ny, Nz), (L/Nx, d/Ny, t))
    mat  = Material(Msat = Msat, Aex = Aex, alpha = 0.5)

    world = World(mesh, mat; demag = true)
    # Saturate along the body diagonal, then relax at zero field.
    m = setconfig(mesh, UniformConfig(1, 1, 1))
    mn = Minimizer(world, m; stopdm = 1e-7)
    minimize!(mn; maxsteps = 50_000)
    mx, my, _ = average(mn.m)
    return abs(mx), abs(my), mn.step
end

# OOMMF reference: columns d/lex, ..., Mx/Ms (5), My/Ms (6).
function read_oommf(path)
    dlex = Float64[]; mx = Float64[]; my = Float64[]
    for line in eachline(path)
        v = split(strip(line))
        (isempty(v) || !occursin(r"^[0-9.]+$", v[1])) && continue
        push!(dlex, parse(Float64, v[1]))
        push!(mx, parse(Float64, v[5]))
        push!(my, parse(Float64, v[6]))
    end
    (dlex, mx, my)
end

function main()
    here = @__DIR__
    @printf("Standard Problem 2 — lex = %.3f nm\n\n", lex * 1e9)

    # A representative sweep spanning the OOMMF range (small → large particle).
    sizes = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    jdlex = Float64[]; jmx = Float64[]; jmy = Float64[]
    println(" d/lex     mx        my      (minimizer steps)")
    for dl in sizes
        mx, my, steps = remanence(dl)
        push!(jdlex, dl); push!(jmx, mx); push!(jmy, my)
        @printf("  %5.1f   %.4f    %.4f    (%d)\n", dl, mx, my, steps)
        flush(stdout)
    end

    odlex, omx, omy = read_oommf(joinpath(here, "stdprob2_oommf.txt"))

    # Write the JuliaMag results as an ASCII tab-separated table.
    tblpath = joinpath(here, "stdproblem2.txt")
    open(tblpath, "w") do io
        println(io, "# Standard Problem 2 — remanence vs. size (JuliaMag)")
        println(io, "# lex = ", @sprintf("%.6g", lex), " m; Msat = ", Msat,
                    " A/m; Aex = ", Aex, " J/m")
        println(io, "# d/lex\tmx\tmy")
        for i in eachindex(jdlex)
            @printf(io, "%.4f\t%.6f\t%.6f\n", jdlex[i], jmx[i], jmy[i])
        end
    end
    println("Wrote data table → ", tblpath)

    # Overlay JuliaMag (lines) with the OOMMF reference (triangles).
    plt = plot(xlabel = "d / lex", ylabel = "remanent ⟨m⟩",
               title = "Standard Problem 2: remanence vs. size",
               titlefontsize = 9, legend = :bottomleft, legendfontsize = 6)
    plot!(plt, jdlex, jmx; label = "mx (JuliaMag)", color = :red,  lw = 2)
    plot!(plt, jdlex, jmy; label = "my (JuliaMag)", color = :blue, lw = 2)
    scatter!(plt, odlex, omx; label = "mx (OOMMF)", color = :red,
             marker = :utriangle, ms = 4, msw = 0)
    scatter!(plt, odlex, omy; label = "my (OOMMF)", color = :blue,
             marker = :utriangle, ms = 4, msw = 0)

    out = joinpath(here, "stdproblem2_compare.png")
    savefig(plt, out)
    println("\nWrote comparison figure → ", out)
end

main()
