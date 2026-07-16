# Compare JuliaMag's standard-problem-4 output against a mumax3 reference run.
# Reads both tables, interpolates onto common times, reports the max/RMS
# deviation per component, and overlays the curves.
#
# Run:  julia --project=examples examples/compare_mumax3.jl

using Printf
using Plots

readtable(path) = begin
    t = Float64[]; mx = Float64[]; my = Float64[]; mz = Float64[]
    for line in eachline(path)
        (isempty(line) || startswith(strip(line), "#")) && continue
        v = split(line)
        push!(t, parse(Float64, v[1])); push!(mx, parse(Float64, v[2]))
        push!(my, parse(Float64, v[3])); push!(mz, parse(Float64, v[4]))
    end
    (t, mx, my, mz)
end

# linear interpolation of y(t) at query point tq
function interp(t, y, tq)
    tq <= t[1] && return y[1]
    tq >= t[end] && return y[end]
    k = searchsortedfirst(t, tq)
    t0, t1 = t[k-1], t[k]; y0, y1 = y[k-1], y[k]
    y0 + (y1 - y0) * (tq - t0) / (t1 - t0)
end

here = @__DIR__
jt, jmx, jmy, jmz = readtable(joinpath(here, "stdproblem4.txt"))
mt, mmx, mmy, mmz = readtable(joinpath(here, "stdproblem4_mumax3.txt"))

# Compare on the mumax3 sample times (they span the same 0–1 ns).
dev(jm, mm) = begin
    d = [interp(jt, jm, mt[i]) - mm[i] for i in eachindex(mt)]
    (maximum(abs, d), sqrt(sum(abs2, d) / length(d)))
end
for (name, jm, mm) in (("mx", jmx, mmx), ("my", jmy, mmy), ("mz", jmz, mmz))
    mx_, rms_ = dev(jm, mm)
    @printf("⟨%s⟩:  max |Δ| = %.4f   RMS = %.4f\n", name, mx_, rms_)
end

plt = plot(xlabel = "time (ns)", ylabel = "⟨m⟩",
           title = "Standard Problem 4: JuliaMag vs mumax3", legend = :right)
plot!(plt, jt .* 1e9, jmx; label = "mx (JuliaMag)", color = :red, lw = 2)
plot!(plt, jt .* 1e9, jmy; label = "my (JuliaMag)", color = :green, lw = 2)
plot!(plt, jt .* 1e9, jmz; label = "mz (JuliaMag)", color = :blue, lw = 2)
plot!(plt, mt .* 1e9, mmx; label = "mx (mumax3)", color = :red, ls = :dash, lw = 1)
plot!(plt, mt .* 1e9, mmy; label = "my (mumax3)", color = :green, ls = :dash, lw = 1)
plot!(plt, mt .* 1e9, mmz; label = "mz (mumax3)", color = :blue, ls = :dash, lw = 1)

out = joinpath(here, "stdproblem4_compare.png")
savefig(plt, out)
println("Wrote comparison figure → ", out)
