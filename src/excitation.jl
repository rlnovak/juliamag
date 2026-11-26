# Applied field and current that vary in space and in time.
#
# Besides its uniform part (World.Bext, SpinCurrent.J), the applied field is a sum of terms, each a
# fixed map on the mesh times a scalar function of time, as mumax3's B_ext.Add(mask, f(t)):
#
#     B(r, t) = B_ext(t) + Σ_k  M_k(r) f_k(t),      and the same for the current density J(r, t).
#
# A map is a (3, Nx, Ny, Nz) array in tesla (field) or A/m² (current), kept on the device of the
# World. The scalar factors are set at the time of every evaluation of the equation of motion
# (`_drive!`), and the terms are added in one broadcast each: on a GPU nothing is copied from the
# host during a run. A region is a map that is a vector inside the region and zero elsewhere
# (`regionmap`); a field that is not separable in space and time is a sum of terms.

"""
    MapTerm

One term `map × f(t)` of the applied field or current. `f` is a function of time, or a number for a
term constant in time; `c` is its value at the present time.
"""
mutable struct MapTerm{A}
    map::A
    f::Any
    c::Float64
end

_factor(f::Real, t) = Float64(f)
_factor(f, t) = Float64(f(t))
_timedependent(q::MapTerm) = !(q.f isa Real)

"""
    fieldmap([T=Float64], mesh, f) -> Array{T,4}

A vector map on the cells of `mesh`: `f(x, y, z)` (metres from the centre of the mesh, as the shapes)
returns the vector at the centre of each cell. For [`addfield!`](@ref) (tesla) or
[`addcurrent!`](@ref) (A/m²).

    pulse = fieldmap(mesh) do x, y, z
        (0, 0, exp(-(x^2 + y^2) / (2 * (30e-9)^2)))
    end

`fieldmap(mesh, shape, v)` is `v` inside `shape` (a [`Shape`](@ref)) and zero outside.
"""
fieldmap(mesh::Mesh, f) = fieldmap(Float64, mesh, f)
function fieldmap(::Type{T}, mesh::Mesh, f) where {T}
    a = zeros(T, 3, mesh.size...)
    @inbounds for k in 1:mesh.size[3], j in 1:mesh.size[2], i in 1:mesh.size[1]
        v = f(cellcenter(mesh, i, j, k)...)
        a[1, i, j, k] = v[1]; a[2, i, j, k] = v[2]; a[3, i, j, k] = v[3]
    end
    all(isfinite, a) || throw(ArgumentError("fieldmap: the function gives a value that is not finite"))
    return a
end
fieldmap(mesh::Mesh, shape::Function, v) = fieldmap(Float64, mesh, shape, v)
fieldmap(::Type{T}, mesh::Mesh, shape::Function, v) where {T} = fieldmap(T, mesh, (x, y, z) -> shape(x, y, z) ? v : (0, 0, 0))

"""
    regionmap(params, id, v) -> Array{Float64,4}

The vector `v` in the cells of region `id` of a `RegionParams` (or of a `Simulation`), zero elsewhere:
a field or current per region for [`addfield!`](@ref) and [`addcurrent!`](@ref). The terms add to the
uniform part, whereas mumax3's `B_ext.SetRegion(r, v)` replaces it in region r: the two agree when the
uniform part is zero. The map is taken from the regions as they are now.
"""
function regionmap(rp::RegionParams, id::Integer, v)
    0 <= id < MAXREGIONS || throw(ArgumentError("region id must be 0–$(MAXREGIONS-1), got $id"))
    a = zeros(3, size(rp.regions.id)...)
    for I in CartesianIndices(rp.regions.id)
        rp.regions.id[I] == id && (a[1, I] = v[1]; a[2, I] = v[2]; a[3, I] = v[3])
    end
    return a
end
regionmap(::Material, id, v) = throw(ArgumentError("regionmap: a scalar Material has no regions; use fieldmap"))

# The map on the device and in the precision of `like`.
function _devicemap(like::AbstractArray{T,4}, map) where {T}
    size(map) == size(like) || throw(DimensionMismatch("the map has size $(size(map)), the mesh needs $(size(like))"))
    all(isfinite, map) || throw(ArgumentError("the map has values that are not finite"))
    return copyto!(similar(like), T.(Array(map)))
end
