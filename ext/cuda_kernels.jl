# Hand-written CUDA kernels for the per-cell terms (included by JuliaMagCUDAExt).
#
# One thread per cell, one launch per term, no temporaries — the structure of
# mumax3's cuda/*.cu kernels. The first GPU port expressed every stencil as array
# broadcasts over shifted copies (circshift); on a small mesh that cost ~180 kernel
# launches and ~70 device allocations per right-hand side and left the GPU idle half
# of the time (gpu_validation/disk_stt/juliamag/bench_rhs.jl). Each kernel here
# mirrors the arithmetic of the corresponding CPU loop in src/ (same operations in
# the same order), so CPU and GPU agree to rounding.
#
# Parameters that are uniform for a scalar Material and per-cell for region params
# are passed either as a scalar or as a (1,Nx,Ny,Nz) device array and read through
# `_pc(x, n)`, so one kernel serves both material models.

const _NTHREADS = 256

@inline _tid() = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

# Launch `f(args...)` with one thread per item, n items.
@inline function _launch(f, n::Integer, args...)
    n > 0 && @cuda threads=_NTHREADS blocks=cld(n, _NTHREADS) f(args...)
    return nothing
end

# Per-cell parameter: a scalar (uniform material) or the n-th entry of an array.
@inline _pc(x::Number, n) = x
@inline _pc(x::AbstractArray, n) = @inbounds x[n]

# Linear cell index n (1-based, x fastest) ↔ (i,j,k).
@inline function _ijk(n, Nx, Ny)
    n0 = n - Int32(1)
    i = n0 % Nx + Int32(1)
    t = n0 ÷ Nx
    j = t % Ny + Int32(1)
    k = t ÷ Ny + Int32(1)
    return i, j, k
end
@inline _lin(i, j, k, Nx, Ny) = i + Nx * ((j - Int32(1)) + Ny * (k - Int32(1)))

# Neighbour index along an axis of length N: periodic wrap, or clamp onto the cell
# itself at a free edge (the CPU rule `i > 1 ? i-1 : (p ? N : 1)`).
@inline _nbm(i, N, p) = i > Int32(1) ? i - Int32(1) : (p ? N : Int32(1))
@inline _nbp(i, N, p) = i < N ? i + Int32(1) : (p ? Int32(1) : N)

# Harmonic mean of two stiffnesses, 0 if either is 0 (src/exchange.jl harmonicmean).
@inline function _hmean(a::T, b::T) where {T}
    (a == zero(T) || b == zero(T)) && return zero(T)
    return T(2) * a * b / (a + b)
end

@inline function _store!(B, c, i, j, k, v, add)
    @inbounds if add
        B[c, i, j, k] += v
    else
        B[c, i, j, k] = v
    end
    return nothing
end

# --- Exchange (src/exchange.jl) ---------------------------------------------
# Neighbour (stiffness, m) for the exchange sum: an empty neighbour or a clamped
# free edge contributes (Ac, m_c), i.e. nothing (Neumann boundary, mumax3
# cuda/exchange.cu).
@inline function _exnb(m, Aex, Mse, Ac, mx, my, mz, clamped, i, j, k, Nx, Ny)
    T = typeof(Ac)
    clamped && return Ac, mx, my, mz
    nb = _lin(i, j, k, Nx, Ny)
    _pc(Mse, nb) == zero(T) && return Ac, mx, my, mz
    @inbounds return _hmean(Ac, _pc(Aex, nb)), m[1, i, j, k], m[2, i, j, k], m[3, i, j, k]
end

function _k_exchange!(B, m, Aex, Mse, Msr, Nx, Ny, Nz, ix2::T, iy2::T, iz2::T,
                      px, py, pz, add) where {T}
    n = _tid()
    n > Nx * Ny * Nz && return nothing
    i, j, k = _ijk(n, Nx, Ny)
    @inbounds begin
        if _pc(Mse, n) == zero(T)                     # empty cell: no field
            if !add
                B[1, i, j, k] = zero(T); B[2, i, j, k] = zero(T); B[3, i, j, k] = zero(T)
            end
            return nothing
        end
        pref = T(2) / _pc(Msr, n)                     # full region Msat (no fill)
        Ac = _pc(Aex, n)
        mx, my, mz = m[1, i, j, k], m[2, i, j, k], m[3, i, j, k]
        il = _nbm(i, Nx, px); ir = _nbp(i, Nx, px)
        jl = _nbm(j, Ny, py); jr = _nbp(j, Ny, py)
        axl, xl1, xl2, xl3 = _exnb(m, Aex, Mse, Ac, mx, my, mz, il == i, il, j, k, Nx, Ny)
        axr, xr1, xr2, xr3 = _exnb(m, Aex, Mse, Ac, mx, my, mz, ir == i, ir, j, k, Nx, Ny)
        ayl, yl1, yl2, yl3 = _exnb(m, Aex, Mse, Ac, mx, my, mz, jl == j, i, jl, k, Nx, Ny)
        ayr, yr1, yr2, yr3 = _exnb(m, Aex, Mse, Ac, mx, my, mz, jr == j, i, jr, k, Nx, Ny)
        fx = pref * (axl * (xl1 - mx) * ix2 + axr * (xr1 - mx) * ix2 +
                     ayl * (yl1 - mx) * iy2 + ayr * (yr1 - mx) * iy2)
        fy = pref * (axl * (xl2 - my) * ix2 + axr * (xr2 - my) * ix2 +
                     ayl * (yl2 - my) * iy2 + ayr * (yr2 - my) * iy2)
        fz = pref * (axl * (xl3 - mz) * ix2 + axr * (xr3 - mz) * ix2 +
                     ayl * (yl3 - mz) * iy2 + ayr * (yr3 - mz) * iy2)
        if Nz > Int32(1)
            kl = _nbm(k, Nz, pz); kr = _nbp(k, Nz, pz)
            azl, zl1, zl2, zl3 = _exnb(m, Aex, Mse, Ac, mx, my, mz, kl == k, i, j, kl, Nx, Ny)
            azr, zr1, zr2, zr3 = _exnb(m, Aex, Mse, Ac, mx, my, mz, kr == k, i, j, kr, Nx, Ny)
            fx += pref * (azl * (zl1 - mx) * iz2 + azr * (zr1 - mx) * iz2)
            fy += pref * (azl * (zl2 - my) * iz2 + azr * (zr2 - my) * iz2)
            fz += pref * (azl * (zl3 - mz) * iz2 + azr * (zr3 - mz) * iz2)
        end
    end
    _store!(B, 1, i, j, k, fx, add); _store!(B, 2, i, j, k, fy, add); _store!(B, 3, i, j, k, fz, add)
    return nothing
end

# --- Uniaxial anisotropy (src/anisotropy.jl) --------------------------------
function _k_anisotropy!(B, m, Ku, Mse, Msr, ux, uy, uz, N, add)
    n = _tid()
    n > N && return nothing
    @inbounds begin
        T = eltype(B)
        # full region Msat (no fill); nothing in an empty cell
        pref = _pc(Mse, n) == zero(T) ? zero(T) : T(2) * _pc(Ku, n) / _pc(Msr, n)
        vx, vy, vz = _pc(ux, n), _pc(uy, n), _pc(uz, n)
        b = 3 * (n - Int32(1))
        mu = m[b+1] * vx + m[b+2] * vy + m[b+3] * vz
        a = pref * mu
        if add
            B[b+1] += a * vx; B[b+2] += a * vy; B[b+3] += a * vz
        else
            B[b+1] = a * vx; B[b+2] = a * vy; B[b+3] = a * vz
        end
    end
    return nothing
end

# --- DMI (src/dmi.jl): central differences; chiral or open free boundaries -----
# Neighbour index along an axis, 0 if missing at an open edge (src/dmi.jl _dminb).
@inline _dnm(i, N, p) = i > Int32(1) ? i - Int32(1) : (p ? N : Int32(0))
@inline _dnp(i, N, p) = i < N ? i + Int32(1) : (p ? Int32(1) : Int32(0))
# Is the neighbour there: inside the mesh (index not 0) and not an empty cell.
@inline function _dok(Mse, ok, i, j, k, Nx, Ny)
    ok || return false
    return _pc(Mse, _lin(i, j, k, Nx, Ny)) != zero(eltype(Mse))
end
@inline _dv(m, c, ok, i, j, k) = ok ? (@inbounds m[c, i, j, k]) : zero(eltype(m))
# 2D/Msat and the chiral-boundary prefactor D²/(2A Msat) of a missing neighbour.
@inline function _dpref(D::T, A::T, Ms::T, openbc) where {T}
    edge = (openbc || A == zero(T)) ? zero(T) : D * D / (T(2) * A * Ms)
    return T(2) * D / Ms, edge
end

function _k_dmi_interfacial!(B, m, D, Aex, Mse, Msr, Nx, Ny, Nz, i2x::T, i2y::T, px, py,
                             openbc, add) where {T}
    n = _tid()
    n > Nx * Ny * Nz && return nothing
    i, j, k = _ijk(n, Nx, Ny)
    if _pc(Mse, n) == zero(T)                             # empty cell: no field
        if !add
            _store!(B, 1, i, j, k, zero(T), false); _store!(B, 2, i, j, k, zero(T), false)
            _store!(B, 3, i, j, k, zero(T), false)
        end
        return nothing
    end
    pref, edge = _dpref(_pc(D, n), _pc(Aex, n), _pc(Msr, n), openbc)
    il = _dnm(i, Nx, px); ir = _dnp(i, Nx, px)
    jl = _dnm(j, Ny, py); jr = _dnp(j, Ny, py)
    okl = _dok(Mse, il != Int32(0), il, j, k, Nx, Ny); okr = _dok(Mse, ir != Int32(0), ir, j, k, Nx, Ny)
    ojl = _dok(Mse, jl != Int32(0), i, jl, k, Nx, Ny); ojr = _dok(Mse, jr != Int32(0), i, jr, k, Nx, Ny)
    dmz_dx = (_dv(m, 3, okr, ir, j, k) - _dv(m, 3, okl, il, j, k)) * i2x
    dmz_dy = (_dv(m, 3, ojr, i, jr, k) - _dv(m, 3, ojl, i, jl, k)) * i2y
    dmx_dx = (_dv(m, 1, okr, ir, j, k) - _dv(m, 1, okl, il, j, k)) * i2x
    dmy_dy = (_dv(m, 2, ojr, i, jr, k) - _dv(m, 2, ojl, i, jl, k)) * i2y
    ex = edge * (Int32(2) - okl - okr); ey = edge * (Int32(2) - ojl - ojr)
    @inbounds mx, my, mz = m[1, i, j, k], m[2, i, j, k], m[3, i, j, k]
    _store!(B, 1, i, j, k, pref * dmz_dx + ex * mx, add)
    _store!(B, 2, i, j, k, pref * dmz_dy + ey * my, add)
    _store!(B, 3, i, j, k, pref * (-dmx_dx - dmy_dy) + (ex + ey) * mz, add)
    return nothing
end

function _k_dmi_bulk!(B, m, D, Aex, Mse, Msr, Nx, Ny, Nz, i2x::T, i2y::T, i2z::T, px, py, pz,
                      openbc, add) where {T}
    n = _tid()
    n > Nx * Ny * Nz && return nothing
    i, j, k = _ijk(n, Nx, Ny)
    if _pc(Mse, n) == zero(T)                             # empty cell: no field
        if !add
            _store!(B, 1, i, j, k, zero(T), false); _store!(B, 2, i, j, k, zero(T), false)
            _store!(B, 3, i, j, k, zero(T), false)
        end
        return nothing
    end
    pref, edge = _dpref(_pc(D, n), _pc(Aex, n), _pc(Msr, n), openbc)
    il = _dnm(i, Nx, px); ir = _dnp(i, Nx, px)
    jl = _dnm(j, Ny, py); jr = _dnp(j, Ny, py)
    kl = _dnm(k, Nz, pz); kr = _dnp(k, Nz, pz)
    okl = _dok(Mse, il != Int32(0), il, j, k, Nx, Ny); okr = _dok(Mse, ir != Int32(0), ir, j, k, Nx, Ny)
    ojl = _dok(Mse, jl != Int32(0), i, jl, k, Nx, Ny); ojr = _dok(Mse, jr != Int32(0), i, jr, k, Nx, Ny)
    oml = _dok(Mse, kl != Int32(0), i, j, kl, Nx, Ny); omr = _dok(Mse, kr != Int32(0), i, j, kr, Nx, Ny)
    dmz_dy = (_dv(m, 3, ojr, i, jr, k) - _dv(m, 3, ojl, i, jl, k)) * i2y
    dmx_dy = (_dv(m, 1, ojr, i, jr, k) - _dv(m, 1, ojl, i, jl, k)) * i2y
    dmz_dx = (_dv(m, 3, okr, ir, j, k) - _dv(m, 3, okl, il, j, k)) * i2x
    dmy_dx = (_dv(m, 2, okr, ir, j, k) - _dv(m, 2, okl, il, j, k)) * i2x
    dmy_dz = (_dv(m, 2, omr, i, j, kr) - _dv(m, 2, oml, i, j, kl)) * i2z
    dmx_dz = (_dv(m, 1, omr, i, j, kr) - _dv(m, 1, oml, i, j, kl)) * i2z
    ex = edge * (Int32(2) - okl - okr); ey = edge * (Int32(2) - ojl - ojr)
    ez = edge * (Int32(2) - oml - omr)
    curlx = dmz_dy - dmy_dz
    curly = dmx_dz - dmz_dx
    curlz = dmy_dx - dmx_dy
    @inbounds mx, my, mz = m[1, i, j, k], m[2, i, j, k], m[3, i, j, k]
    _store!(B, 1, i, j, k, -pref * curlx + (ey + ez) * mx, add)
    _store!(B, 2, i, j, k, -pref * curly + (ex + ez) * my, add)
    _store!(B, 3, i, j, k, -pref * curlz + (ex + ey) * mz, add)
    return nothing
end

# --- Zeeman (src/zeeman.jl) --------------------------------------------------
function _k_zeeman!(B, bx, by, bz, N, add)
    n = _tid()
    n > N && return nothing
    b = 3 * (n - Int32(1))
    @inbounds if add
        B[b+1] += bx; B[b+2] += by; B[b+3] += bz
    else
        B[b+1] = bx; B[b+2] = by; B[b+3] = bz
    end
    return nothing
end

# --- LLG torque (src/llg.jl) -------------------------------------------------
function _k_torque!(dm, m, B, γp, α, N)
    n = _tid()
    n > N && return nothing
    b = 3 * (n - Int32(1))
    @inbounds begin
        mx, my, mz = m[b+1], m[b+2], m[b+3]
        bx, by, bz = B[b+1], B[b+2], B[b+3]
        px = my*bz - mz*by
        py = mz*bx - mx*bz
        pz = mx*by - my*bx
        qx = my*pz - mz*py
        qy = mz*px - mx*pz
        qz = mx*py - my*px
        dm[b+1] = -γp * (px + α*qx)
        dm[b+2] = -γp * (py + α*qy)
        dm[b+3] = -γp * (pz + α*qz)
    end
    return nothing
end

# --- Slonczewski torque (src/stt.jl) ----------------------------------------
function _k_slonczewski!(τ, m, ppx::T, ppy::T, ppz::T, β::T, pol::T, Λ2::T, εp::T, α::T,
                         gilb::T, N, add) where {T}
    n = _tid()
    n > N && return nothing
    b = 3 * (n - Int32(1))
    @inbounds begin
        mx, my, mz = m[b+1], m[b+2], m[b+3]
        pm = ppx*mx + ppy*my + ppz*mz
        ε = pol * Λ2 / ((Λ2 + one(T)) + (Λ2 - one(T)) * pm)
        A = β * ε
        Bs = β * εp
        mxpxmFac = gilb * (A + α*Bs)
        pxmFac   = gilb * (Bs - α*A)
        pxm_x = ppy*mz - ppz*my; pxm_y = ppz*mx - ppx*mz; pxm_z = ppx*my - ppy*mx
        mxpxm_x = my*pxm_z - mz*pxm_y
        mxpxm_y = mz*pxm_x - mx*pxm_z
        mxpxm_z = mx*pxm_y - my*pxm_x
        tx = mxpxmFac*mxpxm_x + pxmFac*pxm_x
        ty = mxpxmFac*mxpxm_y + pxmFac*pxm_y
        tz = mxpxmFac*mxpxm_z + pxmFac*pxm_z
        if add
            τ[b+1] += tx; τ[b+2] += ty; τ[b+3] += tz
        else
            τ[b+1] = tx; τ[b+2] = ty; τ[b+3] = tz
        end
    end
    return nothing
end

# --- Zhang-Li torque (src/stt.jl) --------------------------------------------
function _k_zhangli!(τ, m, Nx, Ny, Nz, wx::T, wy::T, wz::T, usex, usey, usez,
                     px, py, pz, gfac::T, c1::T, c2::T, add) where {T}
    n = _tid()
    n > Nx * Ny * Nz && return nothing
    i, j, k = _ijk(n, Nx, Ny)
    @inbounds begin
        hx = zero(T); hy = zero(T); hz = zero(T)
        if usex
            il = _nbm(i, Nx, px); ir = _nbp(i, Nx, px)
            hx += wx*(m[1,ir,j,k]-m[1,il,j,k]); hy += wx*(m[2,ir,j,k]-m[2,il,j,k]); hz += wx*(m[3,ir,j,k]-m[3,il,j,k])
        end
        if usey
            jl = _nbm(j, Ny, py); jr = _nbp(j, Ny, py)
            hx += wy*(m[1,i,jr,k]-m[1,i,jl,k]); hy += wy*(m[2,i,jr,k]-m[2,i,jl,k]); hz += wy*(m[3,i,jr,k]-m[3,i,jl,k])
        end
        if usez
            kl = _nbm(k, Nz, pz); kr = _nbp(k, Nz, pz)
            hx += wz*(m[1,i,j,kr]-m[1,i,j,kl]); hy += wz*(m[2,i,j,kr]-m[2,i,j,kl]); hz += wz*(m[3,i,j,kr]-m[3,i,j,kl])
        end
        mx, my, mz = m[1,i,j,k], m[2,i,j,k], m[3,i,j,k]
        px1 = my*hz - mz*hy; py1 = mz*hx - mx*hz; pz1 = mx*hy - my*hx
        qx = my*pz1 - mz*py1; qy = mz*px1 - mx*pz1; qz = mx*py1 - my*px1
        tx = gfac * (c1*qx + c2*px1)
        ty = gfac * (c1*qy + c2*py1)
        tz = gfac * (c1*qz + c2*pz1)
    end
    _store!(τ, 1, i, j, k, tx, add); _store!(τ, 2, i, j, k, ty, add); _store!(τ, 3, i, j, k, tz, add)
    return nothing
end

# --- Normalize (src/magnetization.jl) ----------------------------------------
# Max-norm reduction: acc[1] and acc[2] ← max(acc[i], max_n |s·a_n|²) (or
# |s·(a_n − b_n)|² if diff), as unsigned bit patterns of the float type. Each warp
# reduces its 32 values with shuffles, the first warp reduces the warps' results
# from shared memory, and one thread per block makes one atomic max per slot.
# Threads past N take part with 0 (the block size is a multiple of 32, so warps are
# full).
function _k_accmaxnorm2!(acc, a, b, s, N, diff::Bool)
    n = _tid()
    v = zero(s)
    if n <= N
        o = 3 * (n - Int32(1))
        @inbounds begin
            x = a[o+1]; y = a[o+2]; z = a[o+3]
            if diff
                x -= b[o+1]; y -= b[o+2]; z -= b[o+3]
            end
        end
        x *= s; y *= s; z *= s
        v = x * x + y * y + z * z
    end
    U = eltype(acc)
    u = _warpmax(reinterpret(U, v))
    lane = CUDA.laneid()
    wid = (threadIdx().x - Int32(1)) ÷ Int32(32) + Int32(1)
    sh = CuStaticSharedArray(U, 32)
    lane == Int32(1) && (@inbounds sh[wid] = u)
    sync_threads()
    if wid == Int32(1)
        nw = blockDim().x ÷ Int32(32)
        u = lane <= nw ? (@inbounds sh[lane]) : zero(U)
        u = _warpmax(u)
        if lane == Int32(1)
            CUDA.atomic_max!(pointer(acc, 1), u)
            CUDA.atomic_max!(pointer(acc, 2), u)
        end
    end
    return nothing
end

# Max over the 32 lanes of a warp (the result is valid in lane 1).
@inline function _warpmax(u)
    d = Int32(16)
    while d > Int32(0)
        u = max(u, CUDA.shfl_down_sync(CUDA.FULL_MASK, u, d))
        d >>= Int32(1)
    end
    return u
end

function _k_normalize!(m, N)
    n = _tid()
    n > N && return nothing
    b = 3 * (n - Int32(1))
    @inbounds begin
        mx, my, mz = m[b+1], m[b+2], m[b+3]
        nr = sqrt(mx^2 + my^2 + mz^2)
        if nr > zero(nr)
            inv_n = inv(nr)
            m[b+1] = mx * inv_n; m[b+2] = my * inv_n; m[b+3] = mz * inv_n
        end
    end
    return nothing
end

# --- Minimizer helpers (src/minimizer.jl) ------------------------------------
function _k_damping_torque!(τ, m, B, γ, N)
    n = _tid()
    n > N && return nothing
    b = 3 * (n - Int32(1))
    @inbounds begin
        mx, my, mz = m[b+1], m[b+2], m[b+3]
        bx, by, bz = B[b+1], B[b+2], B[b+3]
        px = my*bz - mz*by
        py = mz*bx - mx*bz
        pz = mx*by - my*bx
        τ[b+1] = -γ * (my*pz - mz*py)
        τ[b+2] = -γ * (mz*px - mx*pz)
        τ[b+3] = -γ * (mx*py - my*px)
    end
    return nothing
end

# Cayley step in place; the per-cell |Δm|² goes to d2 (reduced on the host side).
function _k_cayley!(m, τ, d2, dt::T, N) where {T}
    n = _tid()
    n > N && return nothing
    b = 3 * (n - Int32(1))
    @inbounds begin
        tx, ty, tz = τ[b+1], τ[b+2], τ[b+3]
        t2 = dt^2 * (tx^2 + ty^2 + tz^2)
        inv_d = one(T) / (T(4) + t2)
        f = T(4) - t2
        m0x, m0y, m0z = m[b+1], m[b+2], m[b+3]
        nx = (f * m0x + T(4)*dt * tx) * inv_d
        ny = (f * m0y + T(4)*dt * ty) * inv_d
        nz = (f * m0z + T(4)*dt * tz) * inv_d
        m[b+1] = nx; m[b+2] = ny; m[b+3] = nz
        d2[n] = (nx - m0x)^2 + (ny - m0y)^2 + (nz - m0z)^2
    end
    return nothing
end

# --- Demag (src/demag_field.jl) ----------------------------------------------
# Copy the (Msat-weighted) magnetization into the data region of the padded
# (P1,P2,P3,3) buffer; the padding is never written and stays zero.
function _k_demag_pad!(P, m, S, Nx, Ny, Nz)
    n = _tid()
    n > Nx * Ny * Nz && return nothing
    i, j, k = _ijk(n, Nx, Ny)
    @inbounds begin
        s = _pc(S, n)
        P[i, j, k, 1] = s * m[1, i, j, k]
        P[i, j, k, 2] = s * m[2, i, j, k]
        P[i, j, k, 3] = s * m[3, i, j, k]
    end
    return nothing
end

# Fourier-space product Ĥ = K̂·M̂ in place, with the real symmetric kernel.
function _k_demag_kmul!(H, Kxx, Kyy, Kzz, Kxy, Kxz, Kyz, nh)
    n = _tid()
    n > nh && return nothing
    @inbounds begin
        Mx, My, Mz = H[n], H[n + nh], H[n + 2nh]
        kxx, kyy, kzz, kxy, kxz, kyz = Kxx[n], Kyy[n], Kzz[n], Kxy[n], Kxz[n], Kyz[n]
        H[n]       = kxx*Mx + kxy*My + kxz*Mz
        H[n + nh]  = kxy*Mx + kyy*My + kyz*Mz
        H[n + 2nh] = kxz*Mx + kyz*My + kzz*Mz
    end
    return nothing
end

# Copy the data region of the inverse transform out into B, times the prefactor.
function _k_demag_crop!(B, Bo, pref, Nx, Ny, Nz, add)
    n = _tid()
    n > Nx * Ny * Nz && return nothing
    i, j, k = _ijk(n, Nx, Ny)
    @inbounds begin
        v1 = pref * Bo[i, j, k, 1]; v2 = pref * Bo[i, j, k, 2]; v3 = pref * Bo[i, j, k, 3]
    end
    _store!(B, 1, i, j, k, v1, add); _store!(B, 2, i, j, k, v2, add); _store!(B, 3, i, j, k, v3, add)
    return nothing
end
