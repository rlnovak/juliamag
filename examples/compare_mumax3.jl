# Compare JuliaMag's standard-problem-4 output against mumax3 and OOMMF reference
# runs. Reads all three tables, interpolates onto common times, reports the
# max/RMS deviation per component, and overlays the curves (mumax3 as circles,
# OOMMF as squares, JuliaMag as solid lines).
#
# Run:  julia --project=examples examples/compare_mumax3.jl

using Printf
using Plots

# Read a whitespace table, taking t, mx, my, mz from the given 1-based columns.
function readtable(path; tcol, mxcol, mycol, mzcol)
    t = Float64[]; mx = Float64[]; my = Float64[]; mz = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        v = split(s)
        length(v) < max(tcol, mxcol, mycol, mzcol) && continue
        push!(t,  parse(Float64, v[tcol]))
        push!(mx, parse(Float64, v[mxcol]))
        push!(my, parse(Float64, v[mycol]))
        push!(mz, parse(Float64, v[mzcol]))
    end
    (t, mx, my, mz)
end

function interp(t, y, tq)
    tq <= t[1] && return y[1]
    tq >= t[end] && return y[end]
    k = searchsortedfirst(t, tq)
    t0, t1 = t[k-1], t[k]; y0, y1 = y[k-1], y[k]
    y0 + (y1 - y0) * (tq - t0) / (t1 - t0)
end

here = @__DIR__
# JuliaMag: t mx my mz
jt, jmx, jmy, jmz = readtable(joinpath(here, "stdproblem4.txt"); tcol=1, mxcol=2, mycol=3, mzcol=4)
# mumax3: t mx my mz
mt, mmx, mmy, mmz = readtable(joinpath(here, "stdproblem4_mumax3.txt"); tcol=1, mxcol=2, mycol=3, mzcol=4)
# OOMMF .odt: mx/my/mz are columns 15/16/17, simulation time is column 19
ot, omx, omy, omz = readtable(joinpath(here, "stdprob4a_oommf.odt"); tcol=19, mxcol=15, mycol=16, mzcol=17)

# Deviation of JuliaMag from a reference, at the reference's times within the
# range JuliaMag actually simulated (JuliaMag runs to 1 ns; OOMMF runs to 5 ns).
const TMAX = 1e-9
function report(refname, rt, rmx, rmy, rmz)
    idx = findall(t -> t <= jt[end] + 1e-15, rt)
    println("JuliaMag vs $refname (over 0–$(round(jt[end]*1e9, digits=2)) ns):")
    for (name, jm, rm) in (("mx", jmx, rmx), ("my", jmy, rmy), ("mz", jmz, rmz))
        d = [interp(jt, jm, rt[i]) - rm[i] for i in idx]
        @printf("  ⟨%s⟩:  max |Δ| = %.4f   RMS = %.4f\n", name, maximum(abs, d), sqrt(sum(abs2, d)/length(d)))
    end
end
report("mumax3", mt, mmx, mmy, mmz)
report("OOMMF", ot, omx, omy, omz)

# Keep only points within the plotted window, then subsample so markers are readable.
function windowsub(t, y, n)
    keep = findall(τ -> τ <= TMAX + 1e-15, t)
    t = t[keep]; y = y[keep]
    step = max(1, length(t) ÷ n)
    (t[1:step:end], y[1:step:end])
end

plt = plot(xlabel = "time (ns)", ylabel = "⟨m⟩",
           title = "Standard Problem 4: JuliaMag vs mumax3 vs OOMMF",
           titlefontsize = 9, legend = :topright, legendfontsize = 6,
           xlims = (0, TMAX * 1e9))

# JuliaMag: solid lines.
plot!(plt, jt .* 1e9, jmx; label = "mx (JuliaMag)", color = :red,   lw = 2)
plot!(plt, jt .* 1e9, jmy; label = "my (JuliaMag)", color = :green, lw = 2)
plot!(plt, jt .* 1e9, jmz; label = "mz (JuliaMag)", color = :blue,  lw = 2)

# mumax3: filled circles.
for (y, col, lab) in ((mmx, :red, "mx"), (mmy, :green, "my"), (mmz, :blue, "mz"))
    tt, yy = windowsub(mt, y, 40)
    scatter!(plt, tt .* 1e9, yy; label = "$lab (mumax3)", color = col, marker = :circle, ms = 3.5, msw = 0)
end

# OOMMF: open triangles (white fill, coloured edge).
for (y, col, lab) in ((omx, :red, "mx"), (omy, :green, "my"), (omz, :blue, "mz"))
    tt, yy = windowsub(ot, y, 25)
    scatter!(plt, tt .* 1e9, yy; label = "$lab (OOMMF)", markercolor = :white,
             markerstrokecolor = col, marker = :utriangle, ms = 4.5, msw = 1.2)
end

out = joinpath(here, "stdproblem4_compare.png")
savefig(plt, out)
println("Wrote comparison figure → ", out)
