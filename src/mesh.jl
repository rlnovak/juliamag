# Finite-difference mesh: a uniform grid of rectangular cells.
#
# Index convention throughout JuliaMag is (x, y, z), 1-based. A magnetization
# field is stored as an Array{T,4} of shape (3, Nx, Ny, Nz): the vector
# component is the fastest-varying index, so m[:, i, j, k] is contiguous and
# the whole array maps directly onto a CuArray later.

"""
    Mesh(size, cellsize; pbc=(0,0,0))

Uniform finite-difference mesh.

- `size`: number of cells `(Nx, Ny, Nz)`.
- `cellsize`: cell dimensions in metres `(cx, cy, cz)`.
- `pbc`: number of periodic images along each axis; `0` means no periodicity.

# Example
```julia
mesh = Mesh((160, 40, 1), (3.125e-9, 3.125e-9, 3e-9))
```
"""
struct Mesh
    size::NTuple{3,Int}
    cellsize::NTuple{3,Float64}
    pbc::NTuple{3,Int}

    function Mesh(size, cellsize; pbc = (0, 0, 0))
        sz = NTuple{3,Int}(size)
        cs = NTuple{3,Float64}(cellsize)
        pb = NTuple{3,Int}(pbc)
        all(>(0), sz) || throw(ArgumentError("mesh size must be ≥ 1 along every axis, got $sz"))
        all(>(0), cs) || throw(ArgumentError("cell size must be > 0 along every axis, got $cs"))
        all(>=(0), pb) || throw(ArgumentError("pbc must be ≥ 0 along every axis, got $pb"))
        new(sz, cs, pb)
    end
end

Base.size(m::Mesh) = m.size
Base.length(m::Mesh) = prod(m.size)

"Cell volume [m³]."
cellvolume(m::Mesh) = prod(m.cellsize)

"Total simulated volume [m³]."
volume(m::Mesh) = length(m) * cellvolume(m)

"Physical extent of the mesh `(Lx, Ly, Lz)` [m]."
worldsize(m::Mesh) = m.size .* m.cellsize

"True if the mesh is periodic along axis `i` (1 = x, 2 = y, 3 = z)."
isperiodic(m::Mesh, i::Int) = m.pbc[i] != 0

# Zero-padding for the demag convolution. Along a periodic axis the field wraps
# around, so no padding is needed. Otherwise the convolution must be linear, not
# circular, which needs at least 2N-1. We use 2N because power-of-two-ish FFT
# sizes are much faster — except for very small N, where 2N-1 already has only
# small prime factors and padding to 2N wastes memory for nothing.
const SMALL_N = 5

"""
    padsize(mesh) -> NTuple{3,Int}

Size of the zero-padded arrays used by the demag FFT convolution.
"""
function padsize(m::Mesh)
    ntuple(3) do i
        n = m.size[i]
        isperiodic(m, i) ? n : (n > SMALL_N ? 2n : 2n - 1)
    end
end

"""
    dataregion(mesh) -> NTuple{3,UnitRange{Int}}

The slice of a padded array that holds the actual (unpadded) data. The data is
placed at the start of each padded axis, which keeps the demag kernel's origin
at index 1 and makes the wrap-around indexing in [`demagkernel`](@ref) natural.
"""
dataregion(m::Mesh) = ntuple(i -> 1:m.size[i], 3)

function Base.show(io::IO, ::MIME"text/plain", m::Mesh)
    Nx, Ny, Nz = m.size
    cx, cy, cz = m.cellsize
    Lx, Ly, Lz = worldsize(m)
    print(io, "Mesh: ", Nx, " × ", Ny, " × ", Nz, " cells (", length(m), " total)\n")
    print(io, "  cell size: ", cx * 1e9, " × ", cy * 1e9, " × ", cz * 1e9, " nm\n")
    print(io, "  world size: ", Lx * 1e9, " × ", Ly * 1e9, " × ", Lz * 1e9, " nm\n")
    print(io, "  pbc: ", m.pbc)
end

Base.show(io::IO, m::Mesh) = print(io, "Mesh(", m.size, ", ", m.cellsize, "; pbc=", m.pbc, ")")
