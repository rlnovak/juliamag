# Dzyaloshinskii-Moriya interaction (DMI) field.
#
# The antisymmetric DM exchange favours a fixed rotation sense between
# neighbouring moments. Two continuum forms are supported:
#
# Interfacial (Néel), constant D [J/m²], for thin films / multilayers with
# broken inversion symmetry:
#
#     B_dmi = (2 D / Msat) ( ∂mz/∂x, ∂mz/∂y, -∂mx/∂x - ∂my/∂y )
#
# Bulk (Bloch), constant D [J/m²], for chiral B20 crystals:
#
#     B_dmi = -(2 D / Msat) ∇×m
#
# All in Tesla, no μ0 (mumax3 convention; μ0 enters only demag/Zeeman). Both use
# central differences, with index wrap-around on periodic axes. Between two cells of
# different D, the bond takes `dmimean` of the two (mumax3 exchAverage), so the central
# difference is (D_r m_r - D_l m_l)/2c with the D of each bond. The bonds are symmetric
# and the field is the exact gradient of the energy (with full cells; see isconservative).
#
# Free boundaries (a missing neighbour: the edge of a non-periodic axis, or an empty
# cell), as mumax3 (cuda/dmi.cu, cuda/dmibulk.cu). An empty cell (region Msat = 0 or
# fill 0) corresponds to a mumax3 cell outside the geometry (m = 0). A mumax3 region with
# Msat = 0 inside the geometry is something else: it keeps m ≠ 0 and only has A = D = 0
# bonds, so it acts as OpenBC.
#
# - Default: the chiral boundary condition 2A ∂m/∂n + D (…) = 0 (Rohart and Thiaville,
#   Phys. Rev. B 88, 184422 (2013)). mumax3 extrapolates the missing neighbour with
#   it, m' = m + c ∂m/∂n, and puts m' in both the exchange and the DMI stencil. The
#   terms linear in D cancel between the two, and what each missing neighbour adds to
#   (exchange with a Neumann boundary) + DMI is
#
#       (D²/2A Msat) P m,      P: projection on the plane of m that the DMI turns in
#                              along that axis: interfacial x: (mx, 0, mz),
#                              y: (0, my, mz); bulk x: (0, my, mz), y: (mx, 0, mz),
#                              z: (mx, my, 0)
#
#   plus nothing from the missing neighbour in the central difference. It is added
#   here, in the DMI field, so `exchange!` keeps its Neumann rule; only the sum is
#   mumax3's B_exch. The term is diagonal in the cells, so it keeps the field an exact
#   gradient.
#   The DMI of an isotropic bulk material acts along z also in a film of one cell
#   (Nz = 1): both faces count, as in mumax3, B = (D²/A Msat)(mx, my, 0).
# - openbc = true (mumax3 OpenBC): the missing neighbour contributes nothing.
#
# A cell with A = 0 has no chiral boundary condition (D²/2A diverges): it is treated
# as openbc. As in the exchange field, the prefactor divides by the region's full Msat,
# never by Msat·fill (mumax3 inv_Msat).

# Which DMI form is active. For a scalar Material these read the struct fields;
# RegionParams reports true if any region sets the respective constant. The two
# forms are mutually exclusive in a given simulation (mumax3 likewise picks one).
hasdind(m::Material) = m.Dind != 0
hasdbulk(m::Material) = m.Dbulk != 0

"""
    dmi!(B, m, mesh, mat; add=false, openbc=false)

Add (or write) the DMI field of state `m` into `B` [T]. Dispatches on which DMI
constant is set: interfacial if `mat.Dind ≠ 0`, bulk if `mat.Dbulk ≠ 0`. If both
are zero this writes zeros (unless `add=true`, when it is a no-op).

At free boundaries the chiral boundary condition of mumax3 is applied; the part of it
that mumax3 has in the exchange stencil is included here. With `openbc=true` a missing
neighbour contributes nothing (mumax3 `OpenBC`).
"""
function dmi!(B::AbstractArray{T,4}, m::AbstractArray{T,4},
              mesh::Mesh, params::AbstractParams; add::Bool = false, openbc::Bool = false) where {T}
    if hasdind(params)
        return dmi_interfacial!(B, m, mesh, params; add = add, openbc = openbc)
    elseif hasdbulk(params)
        return dmi_bulk!(B, m, mesh, params; add = add, openbc = openbc)
    else
        add || fill!(B, zero(T))
        return B
    end
end

# Neighbours of cell n along an axis of length N, periodic or not: (left, right), 0 for
# a missing one at an open edge.
@inline _dminb(n, N, p) = (n > 1 ? n - 1 : (p ? N : 0), n < N ? n + 1 : (p ? 1 : 0))

# Component c of the neighbour (i,j,k), or 0 if it is missing.
@inline _dmiv(m::AbstractArray{T}, c, ok, i, j, k) where {T} =
    ok ? (@inbounds m[c, i, j, k]) : zero(T)

# D of the bond between two cells (mumax3 engine/exchange.go exchAverage): the harmonic
# mean if the signs agree (0 if either is 0), else sign(a+b)·√(√(-ab)·|a+b|/2), which is 0
# for a = -b.
@inline function dmimean(a::T, b::T) where {T}
    a == b && return a
    (a == 0 || b == 0) && return zero(T)
    (a > 0) == (b > 0) && return 2 / (1 / a + 1 / b)
    s = a + b
    return copysign(sqrt(sqrt(-a * b) * abs(s) / 2), s)
end

# D of the bond to a neighbour, 0 if it is missing (there m counts as 0).
@inline _dbond(f, params, Dc::T, ok, i, j, k) where {T} =
    ok ? dmimean(Dc, f(params, i, j, k)) : zero(T)

# Prefactors of a magnetic cell: 2/Msat, and D²/(2A Msat) of a missing neighbour, with
# the cell's own D and A.
@inline function _dmipref(::Type{T}, D, A, Ms, openbc) where {T}
    edge = (openbc || A == 0) ? zero(T) : T(D^2 / (2 * A * Ms))
    return T(2 / Ms), edge
end

function dmi_interfacial!(B::AbstractArray{T,4}, m::AbstractArray{T,4}, mesh::Mesh,
                          params::AbstractParams; add::Bool = false, openbc::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py = isperiodic(mesh, 1), isperiodic(mesh, 2)
    i2x = T(1 / (2cx)); i2y = T(1 / (2cy))

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if isempty_cell(params, i, j, k)
            add || (B[1, i, j, k] = B[2, i, j, k] = B[3, i, j, k] = zero(T))
            continue
        end
        Dc = dind(params, i, j, k)
        pref, edge = _dmipref(T, Dc, aex(params, i, j, k), msat_region(params, i, j, k), openbc)
        il, ir = _dminb(i, Nx, px); jl, jr = _dminb(j, Ny, py)
        okl = il != 0 && !isempty_cell(params, il, j, k); okr = ir != 0 && !isempty_cell(params, ir, j, k)
        ojl = jl != 0 && !isempty_cell(params, i, jl, k); ojr = jr != 0 && !isempty_cell(params, i, jr, k)
        # D of each bond; the differences below carry it: D ∂m
        dxl = T(_dbond(dind, params, Dc, okl, il, j, k)); dxr = T(_dbond(dind, params, Dc, okr, ir, j, k))
        dyl = T(_dbond(dind, params, Dc, ojl, i, jl, k)); dyr = T(_dbond(dind, params, Dc, ojr, i, jr, k))
        dmz_dx = (dxr * _dmiv(m, 3, okr, ir, j, k) - dxl * _dmiv(m, 3, okl, il, j, k)) * i2x
        dmz_dy = (dyr * _dmiv(m, 3, ojr, i, jr, k) - dyl * _dmiv(m, 3, ojl, i, jl, k)) * i2y
        dmx_dx = (dxr * _dmiv(m, 1, okr, ir, j, k) - dxl * _dmiv(m, 1, okl, il, j, k)) * i2x
        dmy_dy = (dyr * _dmiv(m, 2, ojr, i, jr, k) - dyl * _dmiv(m, 2, ojl, i, jl, k)) * i2y
        ex = edge * (2 - okl - okr); ey = edge * (2 - ojl - ojr)   # missing neighbours

        bx = pref * dmz_dx + ex * m[1, i, j, k]
        by = pref * dmz_dy + ey * m[2, i, j, k]
        bz = pref * (-dmx_dx - dmy_dy) + (ex + ey) * m[3, i, j, k]
        if add
            B[1, i, j, k] += bx; B[2, i, j, k] += by; B[3, i, j, k] += bz
        else
            B[1, i, j, k] = bx; B[2, i, j, k] = by; B[3, i, j, k] = bz
        end
    end
    return B
end

# Bulk DMI field, mumax3 cuda/dmibulk.cu:
#   H_dmi = (2D/Msat)·(∂z my - ∂y mz, ∂x mz - ∂z mx, ∂y mx - ∂x my) = -(2D/Msat)∇×m
function dmi_bulk!(B::AbstractArray{T,4}, m::AbstractArray{T,4}, mesh::Mesh,
                   params::AbstractParams; add::Bool = false, openbc::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh, 1), isperiodic(mesh, 2), isperiodic(mesh, 3)
    i2x = T(1 / (2cx)); i2y = T(1 / (2cy)); i2z = T(1 / (2cz))

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if isempty_cell(params, i, j, k)
            add || (B[1, i, j, k] = B[2, i, j, k] = B[3, i, j, k] = zero(T))
            continue
        end
        Dc = dbulk(params, i, j, k)
        pref, edge = _dmipref(T, Dc, aex(params, i, j, k), msat_region(params, i, j, k), openbc)
        il, ir = _dminb(i, Nx, px); jl, jr = _dminb(j, Ny, py); kl, kr = _dminb(k, Nz, pz)
        okl = il != 0 && !isempty_cell(params, il, j, k); okr = ir != 0 && !isempty_cell(params, ir, j, k)
        ojl = jl != 0 && !isempty_cell(params, i, jl, k); ojr = jr != 0 && !isempty_cell(params, i, jr, k)
        oml = kl != 0 && !isempty_cell(params, i, j, kl); omr = kr != 0 && !isempty_cell(params, i, j, kr)
        # B = -pref * D ∇×m, with the D of each bond in the differences
        dxl = T(_dbond(dbulk, params, Dc, okl, il, j, k)); dxr = T(_dbond(dbulk, params, Dc, okr, ir, j, k))
        dyl = T(_dbond(dbulk, params, Dc, ojl, i, jl, k)); dyr = T(_dbond(dbulk, params, Dc, ojr, i, jr, k))
        dzl = T(_dbond(dbulk, params, Dc, oml, i, j, kl)); dzr = T(_dbond(dbulk, params, Dc, omr, i, j, kr))
        dmz_dy = (dyr * _dmiv(m, 3, ojr, i, jr, k) - dyl * _dmiv(m, 3, ojl, i, jl, k)) * i2y
        dmx_dy = (dyr * _dmiv(m, 1, ojr, i, jr, k) - dyl * _dmiv(m, 1, ojl, i, jl, k)) * i2y
        dmz_dx = (dxr * _dmiv(m, 3, okr, ir, j, k) - dxl * _dmiv(m, 3, okl, il, j, k)) * i2x
        dmy_dx = (dxr * _dmiv(m, 2, okr, ir, j, k) - dxl * _dmiv(m, 2, okl, il, j, k)) * i2x
        dmy_dz = (dzr * _dmiv(m, 2, omr, i, j, kr) - dzl * _dmiv(m, 2, oml, i, j, kl)) * i2z
        dmx_dz = (dzr * _dmiv(m, 1, omr, i, j, kr) - dzl * _dmiv(m, 1, oml, i, j, kl)) * i2z
        ex = edge * (2 - okl - okr); ey = edge * (2 - ojl - ojr); ez = edge * (2 - oml - omr)

        curlx = dmz_dy - dmy_dz
        curly = dmx_dz - dmz_dx
        curlz = dmy_dx - dmx_dy
        bx = -pref * curlx + (ey + ez) * m[1, i, j, k]
        by = -pref * curly + (ex + ez) * m[2, i, j, k]
        bz = -pref * curlz + (ex + ey) * m[3, i, j, k]
        if add
            B[1, i, j, k] += bx; B[2, i, j, k] += by; B[3, i, j, k] += bz
        else
            B[1, i, j, k] = bx; B[2, i, j, k] = by; B[3, i, j, k] = bz
        end
    end
    return B
end
