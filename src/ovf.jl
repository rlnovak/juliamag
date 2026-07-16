# OVF (OOMMF Vector Field) file I/O.
#
# OVF is the standard micromagnetic field format, written by OOMMF and mumax3.
# This reads OVF 1.0/2.0 files (text or binary 4/8-byte) into the package's
# (3, Nx, Ny, Nz) layout, so a simulation can start from a saved state. Only the
# reader is needed for initial conditions; a writer can follow later.
#
# Header keys carry the mesh (xnodes/ynodes/znodes, *stepsize) and the data
# encoding. The binary formats start with a known check value (1234567.0 for
# 4-byte, 123456789012345.0 for 8-byte) used to detect byte order.

"""
    loadovf([T=Float64], filename) -> (m::Array{T,4}, header::Dict)

Read an OVF file into a `(3, Nx, Ny, Nz)` magnetization array (normalized as
stored) plus the parsed header. Supports OVF text and binary (4- and 8-byte)
data segments.
"""
loadovf(filename::AbstractString) = loadovf(Float64, filename)

function loadovf(::Type{T}, filename::AbstractString) where {T<:AbstractFloat}
    open(filename, "r") do io
        header = _ovf_header(io)
        nx = Int(header["xnodes"]); ny = Int(header["ynodes"]); nz = Int(header["znodes"])
        m = zeros(T, 3, nx, ny, nz)
        _ovf_data!(io, m, header)
        return m, header
    end
end

const _OVF_KEYS = ("xnodes", "ynodes", "znodes",
                   "xstepsize", "ystepsize", "zstepsize",
                   "xbase", "ybase", "zbase", "valuemultiplier")

# Parse the header up to and including the "Begin: Data ..." line, which is
# returned under the key "dataline".
function _ovf_header(io::IO)
    header = Dict{String,Any}()
    line = ""
    while !occursin("Begin: Data", line)
        eof(io) && error("OVF: reached end of file before the data segment")
        line = readline(io)
        for key in _OVF_KEYS
            if occursin(key, line)
                header[key] = parse(Float64, split(line)[end])
            end
        end
    end
    header["dataline"] = line
    return header
end

# Decode the data segment into m, given the parsed header. OVF stores the fastest
# index as x, then y, then z, with the 3 vector components consecutive.
function _ovf_data!(io::IO, m::AbstractArray{T,4}, header) where {T}
    nx = Int(header["xnodes"]); ny = Int(header["ynodes"]); nz = Int(header["znodes"])
    valm = T(get(header, "valuemultiplier", 1.0))
    tokens = split(header["dataline"])
    fmt = tokens[4]                    # "Text", "Binary"

    if fmt == "Text"
        for k in 1:nz, j in 1:ny, i in 1:nx
            vals = split(readline(io))
            m[1, i, j, k] = valm * parse(T, vals[1])
            m[2, i, j, k] = valm * parse(T, vals[2])
            m[3, i, j, k] = valm * parse(T, vals[3])
        end
    elseif fmt == "Binary"
        nbytes = parse(Int, tokens[5])
        F = nbytes == 4 ? Float32 : Float64
        check = read(io, F)             # byte-order check value
        expected = nbytes == 4 ? 1234567.0f0 : 123456789012345.0
        swap = !(check ≈ F(expected))   # if the check value is off, byte-swap
        for k in 1:nz, j in 1:ny, i in 1:nx
            for c in 1:3
                v = read(io, F)
                swap && (v = bswap(v))
                m[c, i, j, k] = valm * T(v)
            end
        end
    else
        error("OVF: unsupported data format \"$fmt\"")
    end
    return m
end

"""
    meshfromovf(header) -> Mesh

Build a [`Mesh`](@ref) from a parsed OVF header (cell sizes from `*stepsize`,
counts from `*nodes`).
"""
function meshfromovf(header)
    n = (Int(header["xnodes"]), Int(header["ynodes"]), Int(header["znodes"]))
    c = (header["xstepsize"], header["ystepsize"], header["zstepsize"])
    return Mesh(n, c)
end
