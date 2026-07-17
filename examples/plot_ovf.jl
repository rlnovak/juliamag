# Plot a magnetization field saved as an OVF file: an in-plane vector field
# (quiver of mx,my) over an out-of-plane colour map (mz), for the first z-layer.
#
# This is a visualization utility, so it lives in examples/ and depends on Plots
# rather than the core package. Use it on the OVF snapshots written by, e.g.,
# examples/skyrmion_drive.jl.
#
# As a script:  julia --project=examples examples/plot_ovf.jl file.ovf [out.png]
# As a library:  include("examples/plot_ovf.jl"); plotovf("file.ovf")

using JuliaMag
using Plots

"""
    plotovf(ovffile; out=nothing, stride=nothing, layer=1) -> plot

Read `ovffile` and plot the first (or `layer`-th) z-layer: an `mz` colour map
with an overlaid in-plane (mx,my) quiver. Arrows are drawn every `stride` cells
(auto ≈ 20 arrows across x if not given). Saves a PNG if `out` is given.
"""
function plotovf(ovffile::AbstractString; out = nothing, stride = nothing, layer = 1)
    m, header = loadovf(ovffile)
    mesh = meshfromovf(header)
    Nx, Ny, _ = mesh.size
    cx, cy, _ = mesh.cellsize
    # Cell-centre coordinates (nm) with the origin at the sample centre.
    xs = ((1:Nx) .- (Nx + 1) / 2) .* (cx * 1e9)
    ys = ((1:Ny) .- (Ny + 1) / 2) .* (cy * 1e9)

    mz = @view m[3, :, :, layer]
    plt = heatmap(xs, ys, permutedims(mz); c = :bwr, clims = (-1, 1),
                  aspect_ratio = :equal, xlabel = "x (nm)", ylabel = "y (nm)",
                  colorbar_title = "mz", title = basename(ovffile), titlefontsize = 9)

    # Quiver of the in-plane component, subsampled so the arrows stay legible.
    st = stride === nothing ? max(1, Nx ÷ 20) : stride
    qx = Float64[]; qy = Float64[]; qu = Float64[]; qv = Float64[]
    scale = 0.8 * st * cx * 1e9                    # arrow length ~ one stride
    for j in 1:st:Ny, i in 1:st:Nx
        push!(qx, xs[i]); push!(qy, ys[j])
        push!(qu, m[1, i, j, layer] * scale); push!(qv, m[2, i, j, layer] * scale)
    end
    quiver!(plt, qx, qy; quiver = (qu, qv), color = :black, lw = 0.8)

    out !== nothing && (savefig(plt, out); println("Wrote → ", out))
    return plt
end

# Batch helper: plot every OVF matching a glob into PNGs beside them.
function plotovfs(files::AbstractVector{<:AbstractString})
    for f in files
        plotovf(f; out = replace(f, r"\.ovf$" => ".png"))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: julia --project=examples examples/plot_ovf.jl file.ovf [out.png]")
    ovffile = ARGS[1]
    out = length(ARGS) >= 2 ? ARGS[2] : replace(ovffile, r"\.ovf$" => ".png")
    plotovf(ovffile; out = out)
end
