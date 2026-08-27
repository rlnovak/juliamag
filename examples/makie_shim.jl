# A minimal Plots.jl-compatible shim backed by CairoMakie.
#
# Colab's NVIDIA driver ships a libglapi that lacks `_glapi_tls_Current`, which
# breaks the Libglvnd_jll -> GLFW_jll -> GR_jll chain that Plots.jl loads
# unconditionally (Plots' `load_default_backend()` hardcodes `:gr`, so no
# preference or env var avoids it). CairoMakie touches no OpenGL at all.
#
# This implements only the surface the JuliaMag example drivers use:
#   plot, plot!, scatter!, hline!, vline!, savefig, gr
# with the keyword arguments they pass. It is not a general Plots replacement.
#
# Usage: replace `using Plots` with `include("makie_shim.jl")` in a driver.

# CairoMakie is kept inside this module and never `using`-ed at top level:
# it exports `Mesh` (via GeometryBasics), which would collide with JuliaMag's
# own `Mesh`. Only the Plots-compatible names below are exported.
module MakieShim

using CairoMakie
using CairoMakie: Figure, Axis, lines!, hlines!, vlines!, axislegend, save

export ShimPlot, plot, plot!, scatter!, hline!, vline!, savefig, gr, pythonplot

CairoMakie.activate!(type = "png")

# A figure plus its single axis, standing in for a Plots `Plot` object.
mutable struct ShimPlot
    fig::Figure
    ax::Axis
    has_legend::Bool
    legend_pos::Union{Symbol,Nothing}
end

# `gr()` / `pythonplot()` are no-ops: CairoMakie is already active.
gr() = nothing
pythonplot() = nothing

# Plots' :topright / :bottomleft / ... -> Makie's (halign, valign).
function _legend_align(pos)
    pos === :bottomleft  && return (:left,  :bottom)
    pos === :bottomright && return (:right, :bottom)
    pos === :topleft     && return (:left,  :top)
    pos === :topright    && return (:right, :top)
    pos === :right       && return (:right, :center)
    pos === :left        && return (:left,  :center)
    return (:right, :top)
end

# Plots' linestyle spellings -> Makie's.
_linestyle(ls) = ls === :dash ? :dash :
                 ls === :dot ? :dot :
                 ls === :dashdot ? :dashdot : nothing

# Plots' marker shapes -> Makie's.
_marker(m) = m === :utriangle ? :utriangle :
             m === :circle ? :circle :
             m === :square ? :rect :
             m === :diamond ? :diamond : :circle

# Build the figure/axis. Accepts the axis-level kwargs the drivers pass.
function plot(args...; xlabel = "", ylabel = "", title = "", titlefontsize = 12,
              legend = nothing, legendfontsize = 10, kw...)
    fig = Figure(size = (700, 460))
    ax = Axis(fig[1, 1]; xlabel = string(xlabel), ylabel = string(ylabel),
              title = string(title), titlesize = titlefontsize)
    p = ShimPlot(fig, ax, false, legend === false ? nothing : legend)
    # `plot(x, y; ...)` also draws its first series.
    isempty(args) || plot!(p, args...; legendfontsize = legendfontsize, kw...)
    return p
end

# Draw one or more line series. `y` may be a matrix (one column per series),
# in which case `label` may be a row-vector of labels, as Plots allows.
function plot!(p::ShimPlot, x, y; label = nothing, lw = 1.5, color = nothing,
               ls = nothing, marker = nothing, ms = 4, kw...)
    ycols = y isa AbstractMatrix ? [view(y, :, j) for j in axes(y, 2)] : [y]
    labels = label isa AbstractArray ? vec(collect(label)) : [label]
    for (j, yc) in enumerate(ycols)
        lab = j <= length(labels) ? labels[j] : nothing
        kwargs = Dict{Symbol,Any}(:linewidth => lw)
        color === nothing || (kwargs[:color] = color)
        st = _linestyle(ls); st === nothing || (kwargs[:linestyle] = st)
        if lab === nothing || lab == ""
            lines!(p.ax, x, yc; kwargs...)
        else
            lines!(p.ax, x, yc; label = string(lab), kwargs...)
            p.has_legend = true
        end
        # Plots draws markers on top of the line when `marker` is given.
        if marker !== nothing
            mkw = Dict{Symbol,Any}(:marker => _marker(marker), :markersize => ms * 2)
            color === nothing || (mkw[:color] = color)
            CairoMakie.scatter!(p.ax, x, yc; mkw...)
        end
    end
    return p
end

# `plot!(x, y)` with no explicit plot object is not used by the drivers, but
# keep the single-argument `plot!(p, y)` form working.
plot!(p::ShimPlot, y; kw...) = plot!(p, 1:length(y), y; kw...)

function scatter!(p::ShimPlot, x, y; label = nothing, color = nothing,
                  marker = :circle, ms = 4, msw = nothing, kw...)
    kwargs = Dict{Symbol,Any}(:marker => _marker(marker), :markersize => ms * 2)
    color === nothing || (kwargs[:color] = color)
    if label === nothing || label == ""
        CairoMakie.scatter!(p.ax, x, y; kwargs...)
    else
        CairoMakie.scatter!(p.ax, x, y; label = string(label), kwargs...)
        p.has_legend = true
    end
    return p
end

function hline!(p::ShimPlot, ys; color = :gray, ls = nothing, label = nothing, kw...)
    for yv in ys
        st = _linestyle(ls)
        if st === nothing
            hlines!(p.ax, [yv]; color = color)
        else
            hlines!(p.ax, [yv]; color = color, linestyle = st)
        end
    end
    return p
end

function vline!(p::ShimPlot, xs; color = :gray, ls = nothing, label = nothing, kw...)
    for xv in xs
        st = _linestyle(ls)
        if st === nothing
            vlines!(p.ax, [xv]; color = color)
        else
            vlines!(p.ax, [xv]; color = color, linestyle = st)
        end
    end
    return p
end

function savefig(p::ShimPlot, path::AbstractString)
    if p.has_legend && p.legend_pos !== nothing
        ha, va = _legend_align(p.legend_pos)
        axislegend(p.ax; position = (ha, va), framevisible = true)
    elseif p.has_legend
        axislegend(p.ax)
    end
    save(path, p.fig)
    return path
end

end # module MakieShim

using .MakieShim
