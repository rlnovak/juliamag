# Initial magnetization configurations.
#
# Ported from mumax3 (engine/config.go). A Config is a function of position
# (x, y, z) — in metres, measured from the CENTRE of the sample — returning the
# (unnormalized) magnetization vector there. `setconfig!` samples it at every
# cell centre and normalizes.
#
# Localizable textures (vortex, antivortex, vortex wall, skyrmions) are placed by
# translating the config with `translate` (mumax3's `.Transl`), so the same
# builder positions the core anywhere.

"""
    Config

An initial-state function `(x, y, z) -> NTuple{3}`, with `x,y,z` in metres from
the sample centre. Apply it to a magnetization array with [`setconfig!`](@ref).
"""
const Config = Function

# Cell centre position (x, y, z) [m] measured from the sample centre.
@inline function _cellcenter(mesh::Mesh, i, j, k)
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    x = (i - (Nx + 1) / 2) * cx
    y = (j - (Ny + 1) / 2) * cy
    z = (k - (Nz + 1) / 2) * cz
    return (x, y, z)
end

"""
    setconfig!(m, mesh, config) -> m

Fill `m` by sampling `config(x,y,z)` at each cell centre and normalizing.
"""
function setconfig!(m::AbstractArray{T,4}, mesh::Mesh, config::Config) where {T}
    Nx, Ny, Nz = mesh.size
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x, y, z = _cellcenter(mesh, i, j, k)
        v = config(x, y, z)
        n = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
        if n > 0
            m[1, i, j, k] = v[1] / n
            m[2, i, j, k] = v[2] / n
            m[3, i, j, k] = v[3] / n
        else
            m[1, i, j, k] = 0
            m[2, i, j, k] = 0
            m[3, i, j, k] = 1
        end
    end
    return m
end

"""
    setconfig([T=Float64], mesh, config) -> Array{T,4}

Allocate a magnetization array and fill it from `config`.
"""
setconfig(::Type{T}, mesh::Mesh, config::Config) where {T} = setconfig!(zeromag(T, mesh), mesh, config)
setconfig(mesh::Mesh, config::Config) = setconfig(Float64, mesh, config)

"""
    translate(config, dx, dy, dz) -> Config

Shift a configuration so its origin moves to `(dx, dy, dz)` [m] — used to place a
vortex/skyrmion core at a chosen location (mumax3's `.Transl`).
"""
translate(config::Config, dx, dy, dz) = (x, y, z) -> config(x - dx, y - dy, z - dz)

# Replace a NaN vector (r = 0 at a core) with the pure-polarization vector.
@inline function _nonan(v, pol)
    if isnan(v[1]) || isnan(v[2]) || isnan(v[3])
        return (0.0, 0.0, Float64(pol))
    end
    return v
end

# --- Config builders (all mirror engine/config.go) -------------------------

"""
    UniformConfig(mx, my, mz) -> Config
"""
UniformConfig(mx, my, mz) = (x, y, z) -> (Float64(mx), Float64(my), Float64(mz))

"""
    VortexConfig(mesh; circ=1, pol=1) -> Config

In-plane vortex with circulation `circ` (±1) and core polarization `pol` (±1).
The core is smoothed over a couple of cells so it relaxes cleanly.
"""
function VortexConfig(mesh::Mesh; circ::Int = 1, pol::Int = 1)
    diam2 = 2 * mesh.cellsize[1]^2
    return (x, y, z) -> begin
        r2 = x*x + y*y
        r = sqrt(r2)
        mx = -y * circ / r
        my = x * circ / r
        mz = 1.5 * pol * exp(-r2 / diam2)
        _nonan((mx, my, mz), pol)
    end
end

"""
    AntiVortexConfig(mesh; circ=1, pol=1) -> Config
"""
function AntiVortexConfig(mesh::Mesh; circ::Int = 1, pol::Int = 1)
    diam2 = 2 * mesh.cellsize[1]^2
    return (x, y, z) -> begin
        r2 = x*x + y*y
        r = sqrt(r2)
        mx = -x * circ / r
        my = y * circ / r
        mz = 1.5 * pol * exp(-r2 / diam2)
        _nonan((mx, my, mz), pol)
    end
end

"""
    NeelSkyrmionConfig(mesh; charge=1, pol=1) -> Config

Néel (hedgehog) skyrmion with topological `charge` (±1) and core polarization
`pol` (±1). The wall width is 8 cells (mumax3's default).
"""
function NeelSkyrmionConfig(mesh::Mesh; charge::Int = 1, pol::Int = 1)
    w2 = (8 * mesh.cellsize[1])^2
    return (x, y, z) -> begin
        r2 = x*x + y*y
        r = sqrt(r2)
        mz = 2 * pol * (exp(-r2 / w2) - 0.5)
        mx = (x * charge / r) * (1 - abs(mz))
        my = (y * charge / r) * (1 - abs(mz))
        _nonan((mx, my, mz), pol)
    end
end

"""
    BlochSkyrmionConfig(mesh; charge=1, pol=1) -> Config

Bloch (spiral) skyrmion with topological `charge` (±1) and core polarization
`pol` (±1).
"""
function BlochSkyrmionConfig(mesh::Mesh; charge::Int = 1, pol::Int = 1)
    w2 = (8 * mesh.cellsize[1])^2
    return (x, y, z) -> begin
        r2 = x*x + y*y
        r = sqrt(r2)
        mz = 2 * pol * (exp(-r2 / w2) - 0.5)
        mx = (-y * charge / r) * (1 - abs(mz))
        my = (x * charge / r) * (1 - abs(mz))
        _nonan((mx, my, mz), pol)
    end
end

"""
    VortexWallConfig(mesh, mleft, mright; circ=1, pol=1) -> Config

A vortex wall: uniform `±x` domains (`mleft`, `mright` along x) joined by a
vortex, with the domains beyond `±Ly/2`.
"""
function VortexWallConfig(mesh::Mesh, mleft, mright; circ::Int = 1, pol::Int = 1)
    h = worldsize(mesh)[2]
    v = VortexConfig(mesh; circ = circ, pol = pol)
    return (x, y, z) -> begin
        x < -h/2 && return (Float64(mleft), 0.0, 0.0)
        x > h/2  && return (Float64(mright), 0.0, 0.0)
        v(x, y, z)
    end
end

"""
    TwoDomainConfig(mesh, m1, mwall, m2) -> Config

Two uniform domains with a smoothed wall. `m1`, `mwall`, `m2` are 3-tuples: the
magnetization of the left domain, the wall centre, and the right domain. The
wall is a Gaussian ~2 cells wide, e.g.

    TwoDomainConfig(mesh, (1,0,0), (0,1,0), (-1,0,0))  # head-to-head, Néel wall
    TwoDomainConfig(mesh, (1,0,0), (0,0,1), (-1,0,0))  # head-to-head, Bloch wall
"""
function TwoDomainConfig(mesh::Mesh, m1, mwall, m2)
    ww = 2 * mesh.cellsize[1]
    m1 = NTuple{3,Float64}(m1); mw = NTuple{3,Float64}(mwall); m2 = NTuple{3,Float64}(m2)
    return (x, y, z) -> begin
        base = x < 0 ? m1 : m2
        g = exp(-(x / ww)^2)
        ((1 - g) * base[1] + g * mw[1],
         (1 - g) * base[2] + g * mw[2],
         (1 - g) * base[3] + g * mw[3])
    end
end

"""
    RandomConfig([rng]) -> Config

Random unit vector per cell, uniform on the sphere.
"""
function RandomConfig(rng::AbstractRNG)
    return (x, y, z) -> begin
        θ = 2π * rand(rng)
        zc = 2 * (rand(rng) - 0.5)
        b = sqrt(1 - zc^2)
        (b * cos(θ), b * sin(θ), zc)
    end
end
RandomConfig() = RandomConfig(Random.default_rng())
