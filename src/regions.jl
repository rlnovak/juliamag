# Material regions.
#
# Ported from mumax3 (engine/regions.go). A Regions object assigns each cell an
# integer region id (0–255); region 0 is the default that fills the whole sample.
# `defregion!` paints every cell whose centre lies inside a Shape with a given
# id, so different parts of the sample can carry different material parameters
# (Sec. on RegionParams). The region array is the bridge between geometry
# (Shapes) and physics (per-region parameters).

const MAXREGIONS = 256   # region id fits in a UInt8, as in mumax3.

"""
    Regions(mesh) -> Regions

Per-cell region-id map for `mesh`, initialized to region 0 everywhere.
"""
struct Regions
    mesh::Mesh
    id::Array{UInt8,3}     # (Nx, Ny, Nz)
end

Regions(mesh::Mesh) = Regions(mesh, zeros(UInt8, mesh.size...))

Base.size(r::Regions) = size(r.id)
Base.getindex(r::Regions, i, j, k) = r.id[i, j, k]

# Cell centre position (x, y, z) [m] from the sample centre — same convention as
# Shapes and Config.
@inline function cellcenter(mesh::Mesh, i, j, k)
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    ((i - (Nx + 1) / 2) * cx,
     (j - (Ny + 1) / 2) * cy,
     (k - (Nz + 1) / 2) * cz)
end

"""
    defregion!(regions, id, shape) -> regions

Assign region `id` (0–255) to every cell whose centre lies inside `shape`.
Later calls overwrite earlier ones where they overlap, so build a geometry by
painting from the background outward (mumax3's `DefRegion`).
"""
function defregion!(r::Regions, id::Integer, shape::Shape)
    0 <= id < MAXREGIONS || throw(ArgumentError("region id must be 0–$(MAXREGIONS-1), got $id"))
    Nx, Ny, Nz = r.mesh.size
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x, y, z = cellcenter(r.mesh, i, j, k)
        if shape(x, y, z)
            r.id[i, j, k] = UInt8(id)
        end
    end
    return r
end

"""
    defregioncell!(regions, id, i, j, k) -> regions

Assign region `id` to a single cell by index (mumax3's `DefRegionCell`).
"""
function defregioncell!(r::Regions, id::Integer, i::Int, j::Int, k::Int)
    0 <= id < MAXREGIONS || throw(ArgumentError("region id must be 0–$(MAXREGIONS-1), got $id"))
    r.id[i, j, k] = UInt8(id)
    return r
end

"""
    regionlist(regions) -> Vector{Int}

Sorted list of the distinct region ids actually present.
"""
regionlist(r::Regions) = sort!(unique(Int.(r.id)))

"""
    regionvolume(regions, id) -> Int

Number of cells assigned to region `id`.
"""
regionvolume(r::Regions, id::Integer) = count(==(UInt8(id)), r.id)

function Base.show(io::IO, ::MIME"text/plain", r::Regions)
    ids = regionlist(r)
    print(io, "Regions on ", r.mesh.size, " mesh, ids ", ids, "\n")
    for id in ids
        print(io, "  region ", id, ": ", regionvolume(r, id), " cells\n")
    end
end
