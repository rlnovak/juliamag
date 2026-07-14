# The reduced magnetization field m = M / Msat, |m| = 1 everywhere.
#
# Stored as an Array{T,4} of shape (3, Nx, Ny, Nz). Every function here takes an
# AbstractArray{T,4}, so a CuArray dispatches to the same generic code — only
# the inner kernels get GPU-specific methods later.

"""
    zeromag([T=Float64], mesh) -> Array{T,4}

Allocate an uninitialized magnetization field of shape `(3, Nx, Ny, Nz)`, zeroed.
"""
zeromag(::Type{T}, m::Mesh) where {T<:AbstractFloat} = zeros(T, 3, m.size...)
zeromag(m::Mesh) = zeromag(Float64, m)

"""
    uniform!(m, dir)

Set every cell of `m` to the unit vector along `dir`.
"""
function uniform!(m::AbstractArray{T,4}, dir) where {T}
    u = normalize3(NTuple{3,T}(dir))
    @inbounds for k in axes(m, 4), j in axes(m, 3), i in axes(m, 2)
        m[1, i, j, k] = u[1]
        m[2, i, j, k] = u[2]
        m[3, i, j, k] = u[3]
    end
    return m
end

"""
    uniform([T=Float64], mesh, dir) -> Array{T,4}

Uniformly magnetized state pointing along `dir`.
"""
uniform(::Type{T}, mesh::Mesh, dir) where {T} = uniform!(zeromag(T, mesh), dir)
uniform(mesh::Mesh, dir) = uniform(Float64, mesh, dir)

"""
    randommag!([rng], m)

Fill `m` with random unit vectors, uniformly distributed on the sphere.
"""
function randommag!(rng::AbstractRNG, m::AbstractArray{T,4}) where {T}
    @inbounds for k in axes(m, 4), j in axes(m, 3), i in axes(m, 2)
        # Marsaglia: z uniform in [-1,1], azimuth uniform — gives a uniform sphere.
        z = 2 * rand(rng, T) - 1
        φ = 2 * T(π) * rand(rng, T)
        r = sqrt(1 - z^2)
        m[1, i, j, k] = r * cos(φ)
        m[2, i, j, k] = r * sin(φ)
        m[3, i, j, k] = z
    end
    return m
end
randommag!(m::AbstractArray{T,4}) where {T} = randommag!(Random.default_rng(), m)

"""
    randommag([T=Float64], mesh) -> Array{T,4}

Random magnetization state, one uniformly distributed unit vector per cell.
"""
randommag(::Type{T}, mesh::Mesh) where {T} = randommag!(zeromag(T, mesh))
randommag(mesh::Mesh) = randommag(Float64, mesh)

"""
    vortex!(m, mesh; core=1, circulation=1)

In-plane vortex centred on the sample, with the core pointing along `+z` if
`core > 0` and `-z` otherwise. `circulation` sets the handedness.
"""
function vortex!(m::AbstractArray{T,4}, mesh::Mesh; core::Int = 1, circulation::Int = 1) where {T}
    Nx, Ny, Nz = mesh.size
    cx = (Nx + 1) / 2
    cy = (Ny + 1) / 2
    c = T(sign(core))
    ζ = T(sign(circulation))
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        x = T(i - cx)
        y = T(j - cy)
        r = sqrt(x^2 + y^2)
        if r == 0
            m[1, i, j, k] = 0
            m[2, i, j, k] = 0
            m[3, i, j, k] = c
        else
            # In-plane component is tangential; the core tilts out of plane over
            # roughly one cell, which is enough to seed a relaxation.
            mz = c * exp(-r)
            s = sqrt(max(zero(T), 1 - mz^2))
            m[1, i, j, k] = -ζ * s * y / r
            m[2, i, j, k] = ζ * s * x / r
            m[3, i, j, k] = mz
        end
    end
    return m
end

"""
    vortex([T=Float64], mesh; core=1, circulation=1) -> Array{T,4}
"""
vortex(::Type{T}, mesh::Mesh; kw...) where {T} = vortex!(zeromag(T, mesh), mesh; kw...)
vortex(mesh::Mesh; kw...) = vortex(Float64, mesh; kw...)

"""
    normalize!(m)

Rescale every cell of `m` to unit length. The LLG torque conserves |m| only to
the order of the integrator, so this is applied after each step.
"""
function normalize!(m::AbstractArray{T,4}) where {T}
    @inbounds for k in axes(m, 4), j in axes(m, 3), i in axes(m, 2)
        mx = m[1, i, j, k]
        my = m[2, i, j, k]
        mz = m[3, i, j, k]
        n = sqrt(mx^2 + my^2 + mz^2)
        if n > 0
            inv_n = inv(n)
            m[1, i, j, k] = mx * inv_n
            m[2, i, j, k] = my * inv_n
            m[3, i, j, k] = mz * inv_n
        end
    end
    return m
end

"""
    average(m) -> NTuple{3,T}

Spatially averaged magnetization ⟨m⟩ over all cells. This is the quantity the
µMAG standard problems report.
"""
function average(m::AbstractArray{T,4}) where {T}
    sx = sy = sz = zero(T)
    @inbounds for k in axes(m, 4), j in axes(m, 3), i in axes(m, 2)
        sx += m[1, i, j, k]
        sy += m[2, i, j, k]
        sz += m[3, i, j, k]
    end
    n = T(length(m) ÷ 3)
    return (sx / n, sy / n, sz / n)
end

# Helper: normalize a 3-tuple.
function normalize3(u::NTuple{3,T}) where {T}
    n = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
    n > 0 || throw(ArgumentError("cannot normalize the zero vector"))
    return (u[1] / n, u[2] / n, u[3] / n)
end
normalize3(u) = normalize3(NTuple{3,Float64}(u))
