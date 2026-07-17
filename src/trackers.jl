# Feature trackers: locate a vortex core, a skyrmion, or a domain wall in the
# magnetization, and compute the topological charge. Ported from mumax3
# (engine/ext_corepos.go, ext_topologicalcharge.go, ext_centerwall.go).
#
# All positions are returned in metres from the sample centre, the same
# convention as Shapes and Config.

# Sub-cell parabolic interpolation of the location of an extremum, given the
# value f0 at the sample and f1, f2 at the neighbours one cell to each side.
@inline function _interp_max(f0::T, f1::T, f2::T) where {T}
    # Fit a parabola through (-1,f1),(0,f0),(1,f2); vertex offset in [-0.5,0.5].
    denom = f1 - 2f0 + f2
    denom == 0 && return zero(T)
    return T(0.5) * (f1 - f2) / denom
end

"""
    vortexcore(m, mesh) -> (x, y, z, polarity)

Locate a vortex core as the cell of maximum |m_z|, with sub-cell interpolation in
x and y. Returns the core position (m) from the sample centre and the core
polarity (the signed m_z there). Uses the first z-layer.
"""
function vortexcore(m::AbstractArray{T,4}, mesh::Mesh) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    k = 1
    mx = maxi = maxj = 0
    best = T(-1)
    @inbounds for j in 2:Ny-1, i in 2:Nx-1
        a = abs(m[3, i, j, k])
        if a > best
            best = a; maxi = i; maxj = j
        end
    end
    maxi == 0 && return (T(NaN), T(NaN), T(NaN), zero(T))   # flat state, no core
    @inbounds begin
        dx = _interp_max(abs(m[3, maxi, maxj, k]), abs(m[3, maxi-1, maxj, k]), abs(m[3, maxi+1, maxj, k]))
        dy = _interp_max(abs(m[3, maxi, maxj, k]), abs(m[3, maxi, maxj-1, k]), abs(m[3, maxi, maxj+1, k]))
        pol = m[3, maxi, maxj, k]
    end
    x = (maxi + dx - (Nx + 1) / 2) * cx
    y = (maxj + dy - (Ny + 1) / 2) * cy
    z = (k - (Nz + 1) / 2) * cz
    return (T(x), T(y), T(z), T(pol))
end

# Topological charge density q = m·(∂ₓm × ∂ᵧm) / 4π at cell (i,j,k), central
# differences (Neumann at edges). Returns the (unnormalized-by-area) density.
@inline function _topo_density(m, i, j, k, Nx, Ny, ic2x, ic2y)
    @inbounds begin
        il = max(i-1, 1); ir = min(i+1, Nx)
        jl = max(j-1, 1); jr = min(j+1, Ny)
        dxx = (m[1,ir,j,k]-m[1,il,j,k]) * ic2x; dxy = (m[2,ir,j,k]-m[2,il,j,k]) * ic2x; dxz = (m[3,ir,j,k]-m[3,il,j,k]) * ic2x
        dyx = (m[1,i,jr,k]-m[1,i,jl,k]) * ic2y; dyy = (m[2,i,jr,k]-m[2,i,jl,k]) * ic2y; dyz = (m[3,i,jr,k]-m[3,i,jl,k]) * ic2y
        mx, my, mz = m[1,i,j,k], m[2,i,j,k], m[3,i,j,k]
        # cross = ∂ₓm × ∂ᵧm
        cxo = dxy*dyz - dxz*dyy
        cyo = dxz*dyx - dxx*dyz
        czo = dxx*dyy - dxy*dyx
        return mx*cxo + my*cyo + mz*czo
    end
end

"""
    topologicalcharge(m, mesh) -> Q

The 2D topological charge Q = (1/4π) ∫ m·(∂ₓm × ∂ᵧm) dx dy over the first
z-layer. Q = ±1 for a single skyrmion.
"""
function topologicalcharge(m::AbstractArray{T,4}, mesh::Mesh) where {T}
    Nx, Ny, _ = mesh.size
    cx, cy, _ = mesh.cellsize
    ic2x = T(1 / (2cx)); ic2y = T(1 / (2cy))
    s = zero(T)
    @inbounds for j in 1:Ny, i in 1:Nx
        s += _topo_density(m, i, j, 1, Nx, Ny, ic2x, ic2y)
    end
    return s * T(cx * cy / (4π))
end

"""
    skyrmionpos(m, mesh) -> (x, y, z)

Skyrmion position as the centroid of the topological charge density, weighted by
|density|. Returns metres from the sample centre. Uses the first z-layer.

Along a periodic axis the centroid is computed circularly (as a mean angle), so
the position stays correct when the skyrmion straddles the wrap-around seam of a
periodic stripe — where a plain linear centroid would jump to the middle.
"""
function skyrmionpos(m::AbstractArray{T,4}, mesh::Mesh) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    ic2x = T(1 / (2cx)); ic2y = T(1 / (2cy))
    px, py = isperiodic(mesh, 1), isperiodic(mesh, 2)
    # Linear accumulators, and circular (cos/sin) accumulators for periodic axes.
    sw = zero(T); sx = zero(T); sy = zero(T)
    cxs = zero(T); sxs = zero(T); cys = zero(T); sys = zero(T)
    ax = T(2π / Nx); ay = T(2π / Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        w = abs(_topo_density(m, i, j, 1, Nx, Ny, ic2x, ic2y))
        sw += w
        if px
            θ = (i - 1) * ax; cxs += w * cos(θ); sxs += w * sin(θ)
        else
            sx += w * (i - (Nx + 1) / 2) * cx
        end
        if py
            φ = (j - 1) * ay; cys += w * cos(φ); sys += w * sin(φ)
        else
            sy += w * (j - (Ny + 1) / 2) * cy
        end
    end
    sw == 0 && return (T(NaN), T(NaN), T(NaN))
    # Circular mean back to a cell index, then to a centred coordinate.
    x = if px
        θ = atan(sxs, cxs); θ < 0 && (θ += T(2π))
        (θ / ax + 1 - (Nx + 1) / 2) * cx
    else
        sx / sw
    end
    y = if py
        φ = atan(sys, cys); φ < 0 && (φ += T(2π))
        (φ / ay + 1 - (Ny + 1) / 2) * cy
    else
        sy / sw
    end
    z = (1 - (Nz + 1) / 2) * cz
    return (T(x), T(y), T(z))
end

"""
    domainwallpos(m, mesh; comp=1) -> (x, y, z)

Domain-wall position along x as the point where the average (over y) of
magnetization component `comp` (default m_x) crosses zero — a head-to-head or
tail-to-tail wall along the long axis. Returns metres from the sample centre.
"""
function domainwallpos(m::AbstractArray{T,4}, mesh::Mesh; comp::Int = 1) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    k = 1
    # Column average of the chosen component along y, then find the zero crossing.
    prevavg = _colavg(m, comp, 1, Ny, k)
    @inbounds for i in 2:Nx
        a = _colavg(m, comp, i, Ny, k)
        if (prevavg > 0) != (a > 0)              # sign change between i-1 and i
            frac = prevavg / (prevavg - a)       # linear interp of the crossing
            xi = (i - 1) + frac
            x = (xi - (Nx + 1) / 2) * cx
            z = (k - (Nz + 1) / 2) * cz
            return (T(x), T(0), T(z))
        end
        prevavg = a
    end
    return (T(NaN), T(NaN), T(NaN))              # no crossing (single domain)
end

@inline function _colavg(m, comp, i, Ny, k)
    s = zero(eltype(m))
    @inbounds for j in 1:Ny
        s += m[comp, i, j, k]
    end
    return s / Ny
end
