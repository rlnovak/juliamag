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
# Bulk (Bloch), constant D [J/m³], for chiral B20 crystals:
#
#     B_dmi = -(2 D / Msat) ∇×m
#
# All in Tesla, no μ0 (mumax3 convention; μ0 enters only demag/Zeeman).
#
# Both use central differences for the gradients, with a Neumann (∂m/∂n = 0)
# condition at free boundaries — the missing neighbour is taken equal to the
# central cell — and index wrap-around on periodic axes. This matches the
# boundary treatment of the exchange field. (mumax3 adds a chirality-dependent
# boundary correction; that refinement is deferred.)

"""
    dmi!(B, m, mesh, mat; add=false)

Add (or write) the DMI field of state `m` into `B` [T]. Dispatches on which DMI
constant is set: interfacial if `mat.Dind ≠ 0`, bulk if `mat.Dbulk ≠ 0`. If both
are zero this writes zeros (unless `add=true`, when it is a no-op).
"""
# Which DMI form is active. For a scalar Material these read the struct fields;
# RegionParams reports true if any region sets the respective constant. The two
# forms are mutually exclusive in a given simulation (mumax3 likewise picks one).
hasdind(m::Material) = m.Dind != 0
hasdbulk(m::Material) = m.Dbulk != 0

function dmi!(B::AbstractArray{T,4}, m::AbstractArray{T,4},
              mesh::Mesh, params::AbstractParams; add::Bool = false) where {T}
    if hasdind(params)
        return dmi_interfacial!(B, m, mesh, params; add = add)
    elseif hasdbulk(params)
        return dmi_bulk!(B, m, mesh, params; add = add)
    else
        add || fill!(B, zero(T))
        return B
    end
end

# Central-difference partial derivative of component `c` along axis `ax`
# (1=x,2=y,3=z) at cell (i,j,k), with Neumann boundaries / periodic wrap.
@inline function _deriv(m, c, ax, i, j, k, Nx, Ny, Nz, inv2c, px, py, pz)
    if ax == 1
        il = i > 1 ? i - 1 : (px ? Nx : 1)
        ir = i < Nx ? i + 1 : (px ? 1 : Nx)
        return (m[c, ir, j, k] - m[c, il, j, k]) * inv2c
    elseif ax == 2
        jl = j > 1 ? j - 1 : (py ? Ny : 1)
        jr = j < Ny ? j + 1 : (py ? 1 : Ny)
        return (m[c, i, jr, k] - m[c, i, jl, k]) * inv2c
    else
        kl = k > 1 ? k - 1 : (pz ? Nz : 1)
        kr = k < Nz ? k + 1 : (pz ? 1 : Nz)
        return (m[c, i, j, kr] - m[c, i, j, kl]) * inv2c
    end
end

function dmi_interfacial!(B::AbstractArray{T,4}, m::AbstractArray{T,4},
                          mesh::Mesh, params::AbstractParams; add::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh, 1), isperiodic(mesh, 2), isperiodic(mesh, 3)
    i2x = T(1 / (2cx)); i2y = T(1 / (2cy))

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Msc = msat(params, i, j, k)
        pref = Msc == 0 ? zero(T) : T(2 * dind(params, i, j, k) / Msc)  # Tesla, no μ0
        dmz_dx = _deriv(m, 3, 1, i, j, k, Nx, Ny, Nz, i2x, px, py, pz)
        dmz_dy = _deriv(m, 3, 2, i, j, k, Nx, Ny, Nz, i2y, px, py, pz)
        dmx_dx = _deriv(m, 1, 1, i, j, k, Nx, Ny, Nz, i2x, px, py, pz)
        dmy_dy = _deriv(m, 2, 2, i, j, k, Nx, Ny, Nz, i2y, px, py, pz)

        bx = pref * dmz_dx
        by = pref * dmz_dy
        bz = pref * (-dmx_dx - dmy_dy)
        if add
            B[1, i, j, k] += bx; B[2, i, j, k] += by; B[3, i, j, k] += bz
        else
            B[1, i, j, k] = bx; B[2, i, j, k] = by; B[3, i, j, k] = bz
        end
    end
    return B
end

# Bulk DMI field. Matches the interior formula of mumax3 (cuda/dmibulk.cu):
#   H_dmi = (2D/Msat)·(∂z my - ∂y mz, ∂x mz - ∂z mx, ∂y mx - ∂x my) = -(2D/Msat)∇×m
# verified term by term against that kernel's header. mumax3 additionally uses a
# chirality-dependent boundary extrapolation (m_C = m_A + (dm/dn)·Δ with the DMI
# boundary condition D·m + 2A·∂m = 0); here the boundary is the simpler Neumann
# rule shared with the exchange field. The two agree in the interior; near a free
# boundary the fields differ slightly (matters for edge skyrmions).
function dmi_bulk!(B::AbstractArray{T,4}, m::AbstractArray{T,4},
                   mesh::Mesh, params::AbstractParams; add::Bool = false) where {T}
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    px, py, pz = isperiodic(mesh, 1), isperiodic(mesh, 2), isperiodic(mesh, 3)
    i2x = T(1 / (2cx)); i2y = T(1 / (2cy)); i2z = T(1 / (2cz))

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Msc = msat(params, i, j, k)
        pref = Msc == 0 ? zero(T) : T(2 * dbulk(params, i, j, k) / Msc)  # Tesla, no μ0
        # B = -pref * ∇×m
        dmz_dy = _deriv(m, 3, 2, i, j, k, Nx, Ny, Nz, i2y, px, py, pz)
        dmy_dz = Nz > 1 ? _deriv(m, 2, 3, i, j, k, Nx, Ny, Nz, i2z, px, py, pz) : zero(T)
        dmx_dz = Nz > 1 ? _deriv(m, 1, 3, i, j, k, Nx, Ny, Nz, i2z, px, py, pz) : zero(T)
        dmz_dx = _deriv(m, 3, 1, i, j, k, Nx, Ny, Nz, i2x, px, py, pz)
        dmy_dx = _deriv(m, 2, 1, i, j, k, Nx, Ny, Nz, i2x, px, py, pz)
        dmx_dy = _deriv(m, 1, 2, i, j, k, Nx, Ny, Nz, i2y, px, py, pz)

        curlx = dmz_dy - dmy_dz
        curly = dmx_dz - dmz_dx
        curlz = dmy_dx - dmx_dy
        bx = -pref * curlx; by = -pref * curly; bz = -pref * curlz
        if add
            B[1, i, j, k] += bx; B[2, i, j, k] += by; B[3, i, j, k] += bz
        else
            B[1, i, j, k] = bx; B[2, i, j, k] = by; B[3, i, j, k] = bz
        end
    end
    return B
end
