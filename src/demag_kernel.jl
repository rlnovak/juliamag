# Demagnetization (magnetostatic) kernel.
#
# The demag field is a convolution  B_d(r) = -μ0 Msat Σ_r' N(r-r') · m(r')
# where N is the (symmetric, 3×3) demagnetization tensor. N depends only on the
# cell-offset r-r', so it is computed once per mesh and reused every timestep as
# an FFT convolution (that part lives in demag_field.jl).
#
# Each tensor component N_sd(offset) is obtained by brute-force integration: the
# field cell carries a uniform magnetization along axis s, which puts surface
# "magnetic charges" ±1 on its two faces perpendicular to s. We integrate the
# field those charges produce over the volume of the destination cell. The
# integration is adaptive — more sub-points for near cells, fewer for far ones —
# and staggers the source and destination sub-grids, which sharply improves
# accuracy (mumax3's approach; see Vansteenkiste et al. 2014).
#
# This is a direct port of the legacy DemagKernel.jl, rewritten with the (x,y,z)
# index convention and with the buffer/loop bugs fixed (see the commit message).

"""
    DemagKernel{T}

Precomputed demagnetization tensor for one mesh. The six independent components
of the symmetric tensor are stored as `(px, py, pz)`-sized arrays, where
`(px,py,pz) = padsize(mesh)`. Component `Kij[x,y,z]` is the tensor entry for the
cell offset `(x-1, y-1, z-1)`, wrapped for the FFT convolution.
"""
struct DemagKernel{T<:AbstractFloat}
    Kxx::Array{T,3}
    Kyy::Array{T,3}
    Kzz::Array{T,3}
    Kxy::Array{T,3}
    Kxz::Array{T,3}
    Kyz::Array{T,3}
    padsize::NTuple{3,Int}
end

# Index/component convention.
const _X, _Y, _Z = 1, 2, 3

"""
    demagkernel([T=Float64], mesh; accuracy=6.0) -> DemagKernel{T}

Compute the demagnetization tensor for `mesh`. `accuracy` controls the adaptive
sub-sampling: a cell is split into at most ~accuracy³ integration points.
"""
demagkernel(mesh::Mesh; accuracy = 6.0) = demagkernel(Float64, mesh; accuracy = accuracy)

function demagkernel(::Type{T}, mesh::Mesh; accuracy = 6.0) where {T<:AbstractFloat}
    psize = padsize(mesh)
    cellsize = mesh.cellsize
    pbc = mesh.pbc
    @assert accuracy > 0

    # Full symmetric tensor computed in Float64 for accuracy, cast to T at the end.
    K = ntuple(_ -> zeros(Float64, psize...), 6)   # xx, yy, zz, xy, xz, yz
    idx = (xx = 1, yy = 2, zz = 3, xy = 4, xz = 5, yz = 6)

    r1, r2 = kernelranges(psize, pbc)

    # Typical length scale: the smallest cell dimension.
    L = minimum(cellsize)

    for s in 0:2                       # source magnetization direction
        u = s
        v = (s + 1) % 3
        w = (s + 2) % 3

        R = zeros(3)                   # field-cell centre position

        for z in r1[_Z]:r2[_Z]
            zw = wrap(z, psize[_Z])
            zw > psize[_Z] / 2 && continue           # keep one half, mirror later
            R[_Z] = z * cellsize[_Z]

            for y in r1[_Y]:r2[_Y]
                yw = wrap(y, psize[_Y])
                yw > psize[_Y] / 2 && continue
                R[_Y] = y * cellsize[_Y]

                for x in r1[_X]:r2[_X]
                    xw = wrap(x, psize[_X])
                    xw > psize[_X] / 2 && continue
                    R[_X] = x * cellsize[_X]

                    # Sub-sampling count grows for near cells (small centre gap).
                    dx = delta(x) * cellsize[_X]
                    dy = delta(y) * cellsize[_Y]
                    dz = delta(z) * cellsize[_Z]
                    d = sqrt(dx^2 + dy^2 + dz^2)
                    d == 0 && (d = L)
                    maxSize = d / accuracy

                    nv = floor(Int, max(cellsize[v+1] / maxSize, 1.0) + 0.5)
                    nw = floor(Int, max(cellsize[w+1] / maxSize, 1.0) + 0.5)
                    nx = floor(Int, max(cellsize[_X] / maxSize, 1.0) + 0.5)
                    ny = floor(Int, max(cellsize[_Y] / maxSize, 1.0) + 0.5)
                    nz = floor(Int, max(cellsize[_Z] / maxSize, 1.0) + 0.5)
                    nv *= 2                          # stagger source vs. destination
                    nw *= 2

                    scale = 1.0 / (nv * nw * nx * ny * nz)
                    surface = cellsize[v+1] * cellsize[w+1]
                    charge = surface * scale
                    pu1 = cellsize[u+1] / 2.0        # +charge face
                    pu2 = -pu1                       # -charge face

                    B = integrate_cell(R, cellsize, u, v, w,
                                       nv, nw, nx, ny, nz, charge, pu1, pu2)

                    # Store into the independent components. Off-diagonal
                    # entries with d < s are filled by symmetry (N is symmetric).
                    xi, yi, zi = xw + 1, yw + 1, zw + 1
                    if s == 0            # source x → Nxx, Nxy, Nxz
                        K[idx.xx][xi, yi, zi] += B[_X]
                        K[idx.xy][xi, yi, zi] += B[_Y]
                        K[idx.xz][xi, yi, zi] += B[_Z]
                    elseif s == 1        # source y → Nyy, Nyz  (Nyx = Nxy)
                        K[idx.yy][xi, yi, zi] += B[_Y]
                        K[idx.yz][xi, yi, zi] += B[_Z]
                    else                 # source z → Nzz
                        K[idx.zz][xi, yi, zi] += B[_Z]
                    end
                end
            end
        end
    end

    reconstruct_symmetry!(K, idx, psize, pbc)

    Tconv(A) = convert(Array{T,3}, A)
    return DemagKernel{T}(Tconv(K[idx.xx]), Tconv(K[idx.yy]), Tconv(K[idx.zz]),
                          Tconv(K[idx.xy]), Tconv(K[idx.xz]), Tconv(K[idx.yz]), psize)
end

# Surface integral over the source cell's two charged faces, times the volume
# integral over the destination cell. Returns the field 3-vector B.
@inline function integrate_cell(R, cellsize, u, v, w,
                                nv, nw, nx, ny, nz, charge, pu1, pu2)
    cv, cw = cellsize[v+1], cellsize[w+1]
    cX, cY, cZ = cellsize[_X], cellsize[_Y], cellsize[_Z]
    Bx = By = Bz = 0.0
    pole = zeros(3)

    for i in 0:nv-1
        pole[v+1] = -cv/2 + cv/(2nv) + i*(cv/nv)
        for j in 0:nw-1
            pole[w+1] = -cw/2 + cw/(2nw) + j*(cw/nw)
            for α in 0:nx-1
                rx = R[_X] - cX/2 + cX/(2nx) + (cX/nx)*α
                for β in 0:ny-1
                    ry = R[_Y] - cY/2 + cY/(2ny) + (cY/ny)*β
                    for γ in 0:nz-1
                        rz = R[_Z] - cZ/2 + cZ/(2nz) + (cZ/nz)*γ

                        pole[u+1] = pu1
                        dxp = rx - pole[_X]; dyp = ry - pole[_Y]; dzp = rz - pole[_Z]
                        r = sqrt(dxp^2 + dyp^2 + dzp^2)
                        qr = charge / (4π * r^3)
                        bx = dxp * qr; by = dyp * qr; bz = dzp * qr

                        pole[u+1] = pu2
                        dxm = rx - pole[_X]; dym = ry - pole[_Y]; dzm = rz - pole[_Z]
                        r = sqrt(dxm^2 + dym^2 + dzm^2)
                        qr = -charge / (4π * r^3)
                        Bx += (bx + dxm*qr)          # grouped for numerical accuracy
                        By += (by + dym*qr)
                        Bz += (bz + dzm*qr)
                    end
                end
            end
        end
    end
    return (Bx, By, Bz)
end

# Fill the half of the kernel skipped by the `> size/2` guards, using the parity
# of each tensor component under reflection along each axis.
function reconstruct_symmetry!(K, idx, psize, pbc)
    px, py, pz = psize

    # Reflection along x: xx,yy,zz,yz even; xy,xz odd.
    for z in 0:pz-1, y in 0:py-1, x in (px÷2 + 1):(px-1)
        x2 = px - x
        for (c, sgn) in ((idx.xx, 1), (idx.yy, 1), (idx.zz, 1), (idx.yz, 1), (idx.xy, -1), (idx.xz, -1))
            K[c][x+1, y+1, z+1] = sgn * K[c][x2+1, y+1, z+1]
        end
    end
    # Reflection along y: xx,yy,zz,xz even; xy,yz odd.
    for z in 0:pz-1, y in (py÷2 + 1):(py-1), x in 0:px-1
        y2 = py - y
        for (c, sgn) in ((idx.xx, 1), (idx.yy, 1), (idx.zz, 1), (idx.xz, 1), (idx.xy, -1), (idx.yz, -1))
            K[c][x+1, y+1, z+1] = sgn * K[c][x+1, y2+1, z+1]
        end
    end
    # Reflection along z: xx,yy,zz,xy even; xz,yz odd.
    for z in (pz÷2 + 1):(pz-1), y in 0:py-1, x in 0:px-1
        z2 = pz - z
        for (c, sgn) in ((idx.xx, 1), (idx.yy, 1), (idx.zz, 1), (idx.xy, 1), (idx.xz, -1), (idx.yz, -1))
            K[c][x+1, y+1, z+1] = sgn * K[c][x+1, y+1, z2+1]
        end
    end

    # In 2D (single z-layer) the xz and yz components vanish identically.
    if pz == 1
        fill!(K[idx.xz], 0.0)
        fill!(K[idx.yz], 0.0)
    end
    return K
end

# Field-cell loop ranges. Padded (no PBC) axes span the symmetric window; a
# periodic axis sums over its images instead.
function kernelranges(size, pbc)
    r1 = zeros(Int, 3)
    r2 = zeros(Int, 3)
    for c in 1:3
        if pbc[c] == 0
            r1[c], r2[c] = -(size[c] - 1) ÷ 2, (size[c] - 1) ÷ 2
        else
            r1[c], r2[c] = -(size[c] * pbc[c] - 1), (size[c] * pbc[c] - 1)
        end
    end
    if size[_Z] == 1 && pbc[_Z] == 0
        r2[_Z] = 0
    end
    return r1, r2
end

# Closest distance between two cells given their centre offset (in cells): zero
# if they touch even at a corner.
function delta(d::Int)
    d = abs(d)
    d > 0 && (d -= 1)
    return Float64(d)
end

# Wrap an index into [0, max).
function wrap(number::Int, max::Int)
    number < 0 && (number += max)
    number >= max && (number -= max)
    return number
end
