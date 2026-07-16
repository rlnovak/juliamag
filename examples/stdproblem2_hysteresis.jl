# µMAG Standard Problem 2 — coercive field Hc and switching field Hs vs. size.
# https://www.ctcms.nist.gov/~rdm/std2/spec2.html
#
# The field is applied along the body diagonal [1,1,1]. Starting from positive
# saturation, the field is ramped down through zero and negative; at each field
# the state is relaxed with the energy minimizer. The reported quantities:
#   Hc = |field| at which the projection m·[1,1,1]/√3 = (mx+my+mz)/√3 crosses zero.
#   Hs = |field| of the irreversible switching jump (largest drop in the
#        projection between consecutive field steps).
# Both are reported reduced as H/Ms = B/(μ0 Ms).
#
# Hysteresis is far more expensive than the remanence sweep (a full minimization
# at each of many field steps), so this validates a representative set of sizes,
# not the entire OOMMF table.
#
# Run:  julia --project=examples examples/stdproblem2_hysteresis.jl

using JuliaMag
using Printf
using Plots

const Msat = 8.0e5
const Aex  = 1.3e-11
const lex  = sqrt(Aex / (0.5 * JuliaMag.μ0 * Msat^2))
const û    = (1, 1, 1) ./ sqrt(3)                    # field / saturation axis

# Build the mesh + world for a given d/lex.
function make_world(dlex; cells_per_lex = 2.0)
    d = dlex * lex; L = 5d; t = 0.1d
    cx = lex / cells_per_lex
    Nx = max(8, round(Int, L / cx)); Ny = max(4, round(Int, d / cx)); Nz = 1
    mesh = Mesh((Nx, Ny, Nz), (L/Nx, d/Ny, t))
    mat  = Material(Msat = Msat, Aex = Aex, alpha = 0.5)
    (mesh, mat, World(mesh, mat; demag = true))
end

proj(m) = sum(average(m))/sqrt(3)                    # ⟨m⟩·[1,1,1]/√3

# Descending hysteresis branch from +saturation; returns (Hc/Ms, Hs/Ms).
function hysteresis(dlex; Hmax = 0.08, dH = 0.001)
    mesh, mat, world = make_world(dlex)
    Bsat = Hmax * JuliaMag.μ0 * Msat                 # convert H/Ms → B [T]
    m = setconfig(mesh, UniformConfig(û...))

    # Saturate in a strong +[1,1,1] field.
    setexternalfield!(world, Bsat .* û)
    mn = Minimizer(world, m; stopdm = 1e-6); minimize!(mn; maxsteps = 50_000); m = mn.m

    hs_over_ms = Hmax                                # H/Ms values (descending)
    prev_p = proj(m)
    Hc = NaN; Hs = NaN; maxdrop = 0.0
    h = Hmax
    while h > -Hmax
        h -= dH
        setexternalfield!(world, (h * JuliaMag.μ0 * Msat) .* û)
        mn = Minimizer(world, m; stopdm = 1e-6); minimize!(mn; maxsteps = 50_000); m = mn.m
        p = proj(m)
        # Hc: projection crosses zero (linear interpolation between steps).
        if isnan(Hc) && prev_p > 0 && p <= 0
            Hc = abs(h + dH * p / (p - prev_p))
        end
        # Hs: largest single-step drop in the projection (the switching jump).
        drop = prev_p - p
        if drop > maxdrop
            maxdrop = drop; Hs = abs(h + dH/2)
        end
        prev_p = p
        # Stop once fully reversed.
        p < -0.95 && break
    end
    (Hc, Hs)
end

function read_oommf(path)
    dlex = Float64[]; hc = Float64[]; hs = Float64[]
    for line in eachline(path)
        v = split(strip(line))
        (isempty(v) || !occursin(r"^[0-9.]+$", v[1])) && continue
        push!(dlex, parse(Float64, v[1]))
        push!(hc, parse(Float64, v[3])); push!(hs, parse(Float64, v[4]))
    end
    (dlex, hc, hs)
end

function main()
    here = @__DIR__
    @printf("Standard Problem 2 hysteresis — lex = %.3f nm\n\n", lex * 1e9)

    sizes = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    jd = Float64[]; jhc = Float64[]; jhs = Float64[]
    println(" d/lex    Hc/Ms     Hs/Ms")
    for dl in sizes
        hc, hs = hysteresis(dl)
        push!(jd, dl); push!(jhc, hc); push!(jhs, hs)
        @printf("  %5.1f   %.5f   %.5f\n", dl, hc, hs); flush(stdout)
    end

    od, ohc, ohs = read_oommf(joinpath(here, "stdprob2_oommf.txt"))

    plt = plot(xlabel = "d / lex", ylabel = "H / Ms",
               title = "Standard Problem 2: coercive & switching fields",
               titlefontsize = 9, legend = :right, legendfontsize = 6,
               xscale = :log10)
    plot!(plt, od, ohc; label = "Hc/Ms (OOMMF)", color = :red,  lw = 1.5)
    plot!(plt, od, ohs; label = "Hs/Ms (OOMMF)", color = :blue, lw = 1.5, ls = :dash)
    scatter!(plt, jd, jhc; label = "Hc/Ms (JuliaMag)", color = :red,
             marker = :circle, ms = 5, msw = 0)
    scatter!(plt, jd, jhs; label = "Hs/Ms (JuliaMag)", color = :blue,
             marker = :utriangle, ms = 5, msw = 0)

    out = joinpath(here, "stdproblem2_hysteresis.png")
    savefig(plt, out)
    println("\nWrote figure → ", out)
end

main()
