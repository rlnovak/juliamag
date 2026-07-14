"""
    Definition of simulation mesh, geometry and regions with different material parameters for micromagnetic simulation.

    Rafael L. Novak, rlnovak@gmail.com, jan2020, UFSC/Blumenau (Brazil).
"""

module MeshGeometry

import Base: show, size, length

export Mesh, Geometry
export testmesh, size, length, show, NCells, CellSize, PBC, WorldSize

abstract type AbstractMesh end # Necessary?

struct Mesh <: AbstractMesh
    size::Vector{Int64}
    padsize::Vector{Int64}
    dataranges::Vector{UnitRange}
    cellsize::Vector{Float64}
    pbc::Vector{Int64}
    unit::String
end

# struct Geometry #<: AbstractMesh
#     geometry::Array{Int32, 3}
#     mesh::Mesh
# end

############# For testing purposes! Delete later. ###############
testmesh() = Mesh([8,8,1],[5e-9,5e-9,2e-9],[0,0,0])

#### Constructors ####
## TODO: Improve the conditions imposed on the constructor arguments!
function padSize(gridsize::Vector{Int64}, pbc)
    SMALL_N = 5::Int
    ###############################
    # Index -> Component convention
    X = 1::Int # 1 -> X
    Y = 2::Int # 2 -> Y
    Z = 3::Int # 3 -> Z
    ################################
    if pbc == [0,0,0] # No PBC along x, y or z directions!
        padsize = [i != Z || gridsize[i] > SMALL_N ? gridsize[i]*2 : gridsize[i]*2 - 1 for i in 1:length(gridsize)]
    else # PBC along one or more directions.
        pbc_axes = findall(pbc.!=0)
        padsize = [i ∉ pbc_axes ? (i != Z || gridsize[i] > SMALL_N ? gridsize[i]*2 : gridsize[i]*2 - 1 ) : gridsize[i] for i in 1:length(gridsize)]
    end
    return padsize
end

function Mesh(gridsize::Vector{Int64}, cellsize::Vector{Float64}, pbc, unit::String)
    if gridsize == [0,0,0]
        println("Invalid grid sizes! Values must be 1 or more cells along each Cartesian axis.")
    end
    if cellsize == [0.0, 0.0, 0.0]
        println("Invalid cell sizes! Values must not be zero.")
    end
    if unit == "nm" && log10(cellsize[end]) <= -2.0
        @. cellsize *= 1e9
    end
    if unit == "m" && log10(cellsize[end]) > -2.0
        @. cellsize *= 1e-9
    end
    padsize = padSize(gridsize, pbc)
    offsets = Int64[]
    Nx = gridsize[1]; Ny = gridsize[2]; Nz = gridsize[3];
    padNx = padsize[1]; padNy = padsize[2];
    for k in 0:Nz-1
        for i in 0:Nx-1
            off = k*padNy*padNx + 0.25*padNy*(padNx+1)+i*padNy
            push!(offsets, off+1)
            push!(offsets, off+Ny)
        end
    end
    ranges = UnitRange[]
    for i in 1:2:length(offsets)
        push!(ranges, offsets[i]:offsets[i+1])
    end
    Mesh(gridsize, padsize, ranges, cellsize, pbc, unit)
end

function Mesh(gridsize::Vector{Int64}, cellsize::Vector{Float64}, pbc, unit = "m")
    if gridsize == [0,0,0]
        println("Invalid grid sizes! Values must be 1 or more cells along each Cartesian axis.")
    end
    if cellsize == [0.0, 0.0, 0.0]
        println("Invalid cell sizes! Values must not be zero.")
    end
    @assert log10(cellsize[end]) <= -2.0
    padsize = padSize(gridsize, pbc)
    offsets = Int64[]
    Nx = gridsize[1]; Ny = gridsize[2]; Nz = gridsize[3];
    padNx = padsize[1]; padNy = padsize[2];
    for k in 0:Nz-1
        for i in 0:Nx-1
            off = k*padNy*padNx + 0.25*padNy*(padNx+1)+i*padNy
            push!(offsets, off+1)
            push!(offsets, off+Ny)
        end
    end
    ranges = UnitRange[]
    for i in 1:2:length(offsets)
        push!(ranges, offsets[i]:offsets[i+1])
    end
    Mesh(gridsize, padsize, ranges, cellsize, pbc, unit)
end

function Mesh(gridsize::Vector{Int64}, cellsize::Vector{Float64})
    if gridsize == [0,0,0]
        println("Invalid grid sizes! Values must be 1 or more cells along each Cartesian axis.")
    end
    if cellsize == [0.0, 0.0, 0.0]
        println("Invalid cell sizes! Values must not be zero.")
    end
    pbc = [0,0,0] # Ou teria que ser [1,1,1] ?
    unit = "m"
    if log10(cellsize[end]) > -2.0
        @. cellsize *= 1e-9
    end
    padsize = padSize(gridsize, pbc)
    offsets = Int64[]
    Nx = gridsize[1]; Ny = gridsize[2]; Nz = gridsize[3];
    padNx = padsize[1]; padNy = padsize[2];
    for k in 0:Nz-1
        for i in 0:Nx-1
            off = k*padNy*padNx + 0.25*padNy*(padNx+1)+i*padNy
            push!(offsets, off+1)
            push!(offsets, off+Ny)
        end
    end
    ranges = UnitRange[]
    for i in 1:2:length(offsets)
        push!(ranges, offsets[i]:offsets[i+1])
    end
    Mesh(gridsize, padsize, ranges, cellsize, pbc, unit)
end

Base.show(io::IO, ::MIME"text/plain", mesh::Mesh) =
    print(io, "\nFinite difference mesh with ", length(mesh), " cells.\n\n", "Cells: ", mesh.size[1], " x ", mesh.size[2], " x ", mesh.size[3], "\n",
    "Cell volume: ", mesh.cellsize[1], " x ", mesh.cellsize[2], " x ", mesh.cellsize[3], " ", mesh.unit, "^3.\n", "PBC: X → ",mesh.pbc[1],", Y → ",mesh.pbc[2],", Z → ",mesh.pbc[3])

Base.show(io::IO, ::MIME"text/html", mesh::Mesh) = 
    print(io, "Finite difference mesh:\n", mesh.size[1], " x ", mesh.size[2], " x ", mesh.size[3], " cells.\n",
    "Cell volume: ", mesh.cellsize[1], " x ", mesh.cellsize[2], " x ", mesh.cellsize[3], " ", mesh.unit, "<sup>3</sup>.\n",
    length(mesh)," cells.\n", "PBC: X -> ",mesh.pbc[1],", Y -> ",mesh.pbc[2],", Z -> ",mesh.pbc[3])

Base.size(mesh::Mesh) = mesh.size

Base.length(mesh::Mesh) = mesh.size[1]*mesh.size[2]*mesh.size[3]

#NCells(mesh::Mesh) = mesh.size[1]*mesh.size[2]*mesh.size[3] # SAme as length(). Maybe unnecessary.

CellSize(mesh::Mesh) = mesh.cellsize

PBC(mesh::Mesh) = mesh.pbc

WorldSize(mesh::Mesh) = (mesh.size[1]*mesh.cellsize[1], mesh.size[2]*mesh.cellsize[2], mesh.size[3]*mesh.cellsize[3])
#setPBC(mesh::Mesh) # Precisa?? A Mesh já não é criada com isso tudo definido e imutável? Senão, tem que trocar a definição para mutable struct, e trocar as tuples por arrays.
#setCellSize()
#setGridSize()

# function Geometry(mesh)
#     geom = 
#     Geometry(mesh, )
# end
end