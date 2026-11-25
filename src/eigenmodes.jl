# Second variation of the energy and linear eigenmodes.
#
# Everything here is built on one linear operator on the tangent space of a state m
# (|m| = 1 in every magnetic cell). For a deviation δ ⟂ m, along great circles,
#
#     E(m + δ + …) = E(m) − V Σ Ms δ·B + ½ V Σ Ms δ·(Hδ) + …
#     H δ = P [ (m·B) δ − L(δ) ]                                        [T]
#
# with B = B_eff(m) (applied field included), L the linear field operator
# (`linearfield!`: exchange, anisotropy, DMI, demag) and P the projection on the tangent
# planes. H is the Riemannian Hessian of the energy, at any state, stationary or not.
#
# Scaled variable. H is self-adjoint in the Ms-weighted product. With w = Ms/Msref and
# u = √w δ, the operator
#
#     A u = √w H (u/√w)                                                 [T]
#
# is symmetric in the plain dot product, and d²E = V Msref (u·A u). Its eigenvalues are
# the curvatures of the energy, in tesla. The number of negative ones (the index) does
# not depend on the scaling (Sylvester): 0 at a minimum, 1 at a first-order saddle.
#
# Dynamics. At an equilibrium m0 the undamped LLG, dm/dt = −γ m×B, linearizes to
#
#     du/dt = γ J A u,       J u = m0 × u
#
# and with Gilbert damping to du/dt = γ/(1+α²) (J − α) A u. The eigenvalues of J A are
# ±iΩ (Ω in tesla, f = γΩ/2π) when A is positive definite.
#
# Vectors are tangent Cartesian fields (3,Nx,Ny,Nz), zero in empty cells: every operation
# is a broadcast or a reduction on whole arrays (CPU or GPU). The dense solvers use a
# local frame (e1, e2, m), two unknowns per magnetic cell.

using LinearAlgebra: Symmetric, Hermitian, eigen, eigvals, cholesky, issuccess, norm, qr, mul!, dot

"""
    TangentHessian(world, m)

The operator A above at the state `m`: `hessian!(y, x, H)` writes A·x for a tangent
field `x` (x ⟂ m, zero in empty cells). `H.torque` is max|m×B| over the magnetic cells
[T], which is 0 at a stationary state.
"""
struct TangentHessian{T<:AbstractFloat,W<:World{T},A<:AbstractArray{T,4},S<:AbstractArray{T,4}}
    world::W
    m::A
    b::S                    # m·B per cell [T]
    sw::S                   # √(Msat/Msref), 0 in empty cells
    isw::S                  # its inverse, 0 in empty cells
    x::A                    # scratch
    h::A
    d::S
    Msref::Float64
    torque::Float64         # max |m × B| [T]
end

function TangentHessian(w::World{T}, m::AbstractArray{T,4}) where {T}
    sz = (1, w.mesh.size...)
    ms = _msatcells(w.material, w.mesh)
    Msref = Float64(maximum(ms))
    Msref > 0 || error("TangentHessian: no magnetic cell")
    n2 = sum(abs2, Array(m); dims = 1)
    all(abs.(n2[ms .> 0] .- 1) .< 1e-3) || error("TangentHessian: |m| ≠ 1 in a magnetic cell")
    wr = T.(sqrt.(Float64.(ms) ./ Msref))
    sw = copyto!(similar(m, sz), wr)
    isw = copyto!(similar(m, sz), T.(ifelse.(wr .> 0, 1 ./ wr, 0)))
    # exactly unit vectors (the projections use m; |m|² − 1 ~ 1e-14 from a minimizer, or
    # 1e-7 from a single-precision file, makes A indefinite at that level), zero where empty
    b = similar(sw); d = similar(sw)
    mc = copy(m)
    for _ in 1:2
        _celldot!(d, mc, mc)
        mc .= ifelse.(sw .> 0, mc ./ sqrt.(d .+ (sw .== 0)), zero(T))
    end
    B = similar(m)
    effectivefield!(B, mc, w)
    _celldot!(b, mc, B)
    b .= ifelse.(sw .> 0, b, zero(T))
    x = similar(m); h = similar(m)
    _cross!(x, mc, B)
    _celldot!(d, x, x)
    d .= ifelse.(sw .> 0, d, zero(T))
    tq = sqrt(Float64(maximum(d)))
    return TangentHessian(w, mc, b, sw, isw, x, h, d, Msref, tq)
end

function _cross!(y, a, x)
    ax, ay, az = _cmp(a, 1), _cmp(a, 2), _cmp(a, 3)
    xx, xy, xz = _cmp(x, 1), _cmp(x, 2), _cmp(x, 3)
    _cmp(y, 1) .= ay .* xz .- az .* xy
    _cmp(y, 2) .= az .* xx .- ax .* xz
    _cmp(y, 3) .= ax .* xy .- ay .* xx
    return y
end

# Projection on the tangent planes, zero in empty cells.
function _tangentproject!(y, H::TangentHessian)
    _celldot!(H.d, H.m, y)
    y .= ifelse.(H.sw .> 0, y .- H.d .* H.m, zero(eltype(y)))
    return y
end

"""
    hessian!(y, x, H::TangentHessian) -> y

y = A·x, in tesla (see the top of `src/eigenmodes.jl`). `x` must be tangent.
"""
function hessian!(y, x, H::TangentHessian)
    H.x .= H.isw .* x
    linearfield!(H.h, H.x, H.world)
    y .= H.sw .* (H.b .* H.x .- H.h)
    return _tangentproject!(y, H)
end

# --- dense form ----------------------------------------------------------------------

# Orthonormal frame (e1, e2, m) with e1 × e2 = m, per cell (Duff et al., J. Comput.
# Graph. Tech. 6, 1 (2017)): continuous except across m_z = 0⁻/0⁺, which is harmless.
function _frame(mx, my, mz)
    s = copysign(one(mz), mz)
    a = -1 / (s + mz)
    b = mx * my * a
    return (1 + s * mx^2 * a, s * b, -s * mx), (b, s + my^2 * a, -my)
end

# The magnetic cells and their frames (host): cells, E[:, n] = (e1..., e2...).
function _frames(H::TangentHessian)
    m = Float64.(Array(H.m)); on = vec(Array(H.sw)) .> 0
    cells = findall(on)
    E = zeros(6, length(cells)); M = reshape(m, 3, :)
    for (n, c) in enumerate(cells)
        v = M[:, c] ./ sqrt(sum(abs2, @view M[:, c]))
        e1, e2 = _frame(v...)
        E[1:3, n] .= e1; E[4:6, n] .= e2
    end
    return cells, E
end

# Frame coordinates (2N) ↔ tangent field.
function _tofield!(x::Array{T}, u, cells, E) where {T}
    fill!(x, 0); X = reshape(x, 3, :)
    @inbounds for (n, c) in enumerate(cells), k in 1:3
        X[k, c] = u[2n-1] * E[k, n] + u[2n] * E[3+k, n]
    end
    return x
end
function _fromfield!(u, x::Array, cells, E)
    X = reshape(x, 3, :)
    @inbounds for (n, c) in enumerate(cells)
        u[2n-1] = X[1, c] * E[1, n] + X[2, c] * E[2, n] + X[3, c] * E[3, n]
        u[2n]   = X[1, c] * E[4, n] + X[2, c] * E[5, n] + X[3, c] * E[6, n]
    end
    return u
end

"""
    densehessian(H::TangentHessian) -> (A, cells, E)

The operator as a dense 2N×2N matrix (Float64, N magnetic cells) in the local frames,
column by column: 2N field evaluations. `cells` are the linear indices of the magnetic
cells and `E` their frames.
"""
function densehessian(H::TangentHessian{T}) where {T}
    cells, E = _frames(H)
    n = 2length(cells)
    A = Matrix{Float64}(undef, n, n)
    x = similar(H.m); y = similar(H.m)
    hx = Array{T}(undef, size(H.m)); hy = similar(hx)
    u = zeros(n)
    for j in 1:n
        u[j] = 1
        copyto!(x, _tofield!(hx, u, cells, E))
        hessian!(y, x, H)
        _fromfield!(@view(A[:, j]), copyto!(hy, y), cells, E)
        u[j] = 0
    end
    return A, cells, E
end

# Relative asymmetry, and the symmetric part in place.
function _symmetrize!(A)
    n = size(A, 1); big = 0.0; asym = 0.0
    @inbounds for j in 1:n, i in 1:j
        a, b = A[i, j], A[j, i]
        big = max(big, abs(a), abs(b)); asym = max(asym, abs(a - b))
        A[i, j] = A[j, i] = (a + b) / 2
    end
    return big == 0 ? 0.0 : asym / big
end

# --- lowest curvatures -----------------------------------------------------------------

_densemax() = 8000       # unknowns (2N) up to which method = :auto is dense

"""
    curvatures(world, m; nev=6, method=:auto, tol=1e-6, maxiter=2000)
        -> (; values, vectors, residuals, torque, method, asymmetry)

The `nev` lowest eigenvalues [T] of the energy Hessian at the state `m` (per unit
displacement in the Msat-weighted norm), with their modes as tangent fields u = √w δ of
unit norm. All positive: a minimum. Exactly one negative: a first-order saddle.

- `method = :dense` diagonalizes the full matrix (2N unknowns, N magnetic cells): no
  eigenvalue can be missed. `:lobpcg` is matrix-free (block size `nev` + 4, residual norm
  below `tol`, 1e-5 in Float32), for large systems; it can in principle miss an eigenvalue.
  `:auto`: dense up to 2N = $(_densemax()).
- `residuals`: ‖A u − λ u‖ [T]; an eigenvalue lies within that distance of each value.
- `torque`: max|m×B| [T] of the state. `asymmetry`: relative asymmetry of the matrix
  (dense only). A field that is not the gradient of the energy ([`isconservative`](@ref))
  is an error.
"""
function curvatures(w::World{T}, m::AbstractArray{T,4}; nev::Integer = 6, method::Symbol = :auto,
                    kwargs...) where {T}
    isconservative(w) || error("curvatures: the effective field is not the gradient of the " *
                               "energy (see `isconservative`), so it has no symmetric Hessian")
    all(in((:tol, :maxiter, :guard, :rng, :verbose)), keys(kwargs)) || error("curvatures: unknown keyword in $(keys(kwargs))")
    H = TangentHessian(w, m)
    n = 2count(>(0), Array(H.sw))
    nev = min(nev, n)
    method === :auto && (method = n <= _densemax() ? :dense : :lobpcg)
    method === :dense && return _densecurvatures(H, nev)
    method === :lobpcg && return _lobpcg(H, nev; kwargs...)
    error("unknown method $method (:auto, :dense, :lobpcg)")
end

function _densecurvatures(H::TangentHessian{T}, nev) where {T}
    A, cells, E = densehessian(H)
    asym = _symmetrize!(A)
    asym < 1e3 * eps(T) || error("curvatures: the Hessian is not symmetric (relative asymmetry $asym)")
    F = eigen(Symmetric(A), 1:nev)
    vecs = [_tofield!(zeros(T, size(H.m)), @view(F.vectors[:, k]), cells, E) for k in 1:nev]
    res = [norm(A * F.vectors[:, k] .- F.values[k] .* F.vectors[:, k]) for k in 1:nev]
    return (; values = F.values, vectors = vecs, residuals = res, torque = H.torque,
            method = :dense, asymmetry = asym)
end

# Orthonormalize the columns of S in place by the eigenvectors of the Gram matrix
# (SVQB), dropping dependent directions. Two passes. Returns the matrix (maybe narrower).
function _svqb(S)
    for _ in 1:2
        G = Symmetric(Float64.(Array(S' * S)))
        d = sqrt.(max.(abs.(G[i, i] for i in axes(G, 1)), floatmin()))
        D = Symmetric([G[i, j] / (d[i] * d[j]) for i in axes(G, 1), j in axes(G, 2)])
        F = eigen(D)
        keep = F.values .> size(G, 1) * 100 * eps(Float64(one(eltype(S)))) * maximum(F.values)
        eltype(S) === Float32 && (keep = F.values .> 1e-5 * maximum(F.values))
        Q = (F.vectors[:, keep] ./ d) ./ sqrt.(F.values[keep])'
        S = S * copyto!(similar(S, size(Q)...), eltype(S).(Q))
    end
    return S
end

_rm(S, Q) = S * copyto!(similar(S, size(Q)...), eltype(S).(Q))     # S·Q, Q a small host matrix

# Lowest eigenpairs of A by LOBPCG (Knyazev 2001) without preconditioner, with an
# orthonormal basis [X W P] at every step.
function _lobpcg(H::TangentHessian{T}, nev; tol = max(1e-6, 100 * Float64(eps(T))), maxiter::Integer = 2000, guard::Integer = 4,
                 rng = Random.MersenneTwister(1), verbose::Bool = false) where {T}
    sz = size(H.m); nn = length(H.m)
    k = nev + guard
    col(S, j) = reshape(view(S, :, j), sz)
    X = similar(H.m, nn, k)
    copyto!(X, randn(rng, T, nn, k))
    foreach(j -> _tangentproject!(col(X, j), H), 1:k)
    X = _svqb(X)
    size(X, 2) == k || error("LOBPCG: the system is too small for block size $k; use :dense")
    buf = similar(H.m); out = similar(H.m)
    function apply(S)
        AS = similar(S)
        for j in axes(S, 2)
            copyto!(buf, col(S, j))
            hessian!(out, buf, H)
            copyto!(view(AS, :, j), vec(out))
        end
        return AS
    end
    AX = apply(X)
    F = eigen(Symmetric(Float64.(Array(X' * AX))))
    X = _rm(X, F.vectors); AX = _rm(AX, F.vectors); λ = F.values
    P = nothing
    res = fill(Inf, k); it = 0
    while true
        R = AX .- X .* copyto!(similar(X, 1, k), T.(reshape(λ, 1, k)))
        res = [sqrt(_gdot(view(R, :, j), view(R, :, j))) for j in 1:k]
        verbose && it % 20 == 0 && println("  LOBPCG ", it, "  λ = ", λ[1:min(nev, 4)], "  res = ", maximum(res[1:nev]))
        (maximum(@view res[1:nev]) < tol || it >= maxiter) && break
        it += 1
        W = R
        for _ in 1:2
            W = W .- X * (X' * W)
        end
        foreach(j -> _tangentproject!(col(W, j), H), 1:size(W, 2))
        S = P === nothing ? W : hcat(W, P)
        for _ in 1:2
            S = S .- X * (X' * S)
        end
        S = _svqb(S)
        foreach(j -> _tangentproject!(col(S, j), H), 1:size(S, 2))
        AS = apply(S)
        B = hcat(X, S); AB = hcat(AX, AS)
        G = Float64.(Array(B' * AB)); M = Float64.(Array(B' * B))
        F = eigen(Symmetric((G + G') / 2), Symmetric((M + M') / 2))
        C = F.vectors[:, 1:k]; λ = F.values[1:k]
        Cs = copy(C); Cs[1:k, :] .= 0
        P = _rm(B, Cs)
        X = _rm(B, C); AX = _rm(AB, C)
        if it % 25 == 0                        # refresh: X orthonormal, AX exact
            X = _svqb(X); AX = apply(X)
            F = eigen(Symmetric(Float64.(Array(X' * AX))))
            X = _rm(X, F.vectors); AX = _rm(AX, F.vectors); λ = F.values
        end
    end
    maximum(@view res[1:nev]) < tol ||
        @warn "LOBPCG did not converge in $maxiter iterations: residual $(maximum(res[1:nev])) T"
    vecs = [Array(reshape(X[:, j], sz)) for j in 1:nev]
    return (; values = λ[1:nev], vectors = vecs, residuals = res[1:nev], torque = H.torque,
            method = :lobpcg, asymmetry = NaN, iterations = it)
end

"""
    certifysaddle(neb; image=climbing image, zerotol=1e-6 (Float32: 1e-5), nev=6, kwargs...)
        -> (; index, nzero, values, residuals, torque, along, image, method)

Order of the saddle point at an image of a relaxed [`GNEB`](@ref) band (the climbing
image; the highest one if there is none), from the lowest curvatures of the energy
([`curvatures`](@ref), whose keywords are accepted).

- `index`: number of eigenvalues below `-zerotol` [T]. 1 is a first-order saddle, the
  transition state of the path. 2 or more: not a transition state. `nev` is doubled until
  a value that is not negative is found, so the index is not truncated.
- `nzero`: eigenvalues within `±zerotol`: symmetries (rotation, translation) leave
  zero modes, which on a discrete lattice are only approximately zero. Look at `values`.
- `along`: curvature along the path tangent, u·A u [T]. The tangent is a finite difference
  between images: in a coarse band it has stiff components and `along` is above
  `values[1]`, even positive. It is not a test of the saddle; `index` is.
- `torque`: max|m×B| at the image [T]; the saddle is a stationary point only to this.
"""
function certifysaddle(neb::GNEB{T}; image::Integer = 0, zerotol = T === Float64 ? 1e-6 : 1e-5, nev::Integer = 6,
                       kwargs...) where {T}
    n = length(neb.m)
    i = image != 0 ? Int(image) : neb.climb != 0 ? neb.climb : argmax(@view neb.E[2:n-1]) + 1
    1 <= i <= n || error("no image $i")
    c = curvatures(neb.world, neb.m[i]; nev = nev, kwargs...)
    dim = 2count(>(0), Array(neb.wr))
    while all(<(-zerotol), c.values) && length(c.values) >= nev       # more negative ones?
        nev *= 2
        c = nev + 8 > dim ÷ 3 && dim <= _densemax() ?                  # too many for LOBPCG
            curvatures(neb.world, neb.m[i]; nev = nev, method = :dense) : curvatures(neb.world, neb.m[i]; nev = nev, kwargs...)
    end
    along = NaN
    if 1 < i < n
        H = TangentHessian(neb.world, neb.m[i])
        u = H.sw .* neb.t[i]
        _tangentproject!(u, H)
        along = _gdot(u, hessian!(similar(u), u, H)) / _gdot(u, u)
    end
    return (; index = count(<(-zerotol), c.values), nzero = count(x -> abs(x) <= zerotol, c.values),
            values = c.values, residuals = c.residuals, torque = c.torque, along = along, image = i,
            method = c.method)
end

# --- eigenmodes -------------------------------------------------------------------------

"""
    EigenModes

Result of [`eigenmodes`](@ref): `frequencies` [Hz], `linewidths` (decay rates Γ [1/s] of
the amplitude, e^{−Γt}, for the damping `alpha`), `modes` (complex fields δm(r),
(3,Nx,Ny,Nz), the largest cell of unit length and its largest component real:
m(t) = m0 + Re[δm e^{iωt}]),
`residuals` (‖J A u − iΩ u‖/(Ω‖u‖)), `torque` (max|m0×B| [T] of the state), `method`.
"""
struct EigenModes
    frequencies::Vector{Float64}
    linewidths::Vector{Float64}
    modes::Vector{Array{ComplexF64,4}}
    residuals::Vector{Float64}
    torque::Float64
    alpha::Float64
    method::Symbol
end

"""
    eigenmodes(world, m0; nmodes=10, method=:auto, alpha=damping, tol=1e-8, …) -> EigenModes

The `nmodes` lowest linear eigenmodes of the magnetization about the equilibrium `m0`
(relaxed tightly: `Minimizer(…; stopdm=1e-9)`; a warning is printed if max|m0×B| is above
`torquetol` = 1e-5 T). The frequencies are those of the undamped motion; the linewidths are
first order in `alpha` (diagonalized among modes of equal frequency). Spin-transfer torques
are not included (a warning is printed if the world has a current).

A state with a continuous symmetry (a texture in a periodic film, which can be translated; a
state that turns about an axis) has zero modes: the Hessian is singular, and the calculation
stops with an error. Break the symmetry, or look at the [`curvatures`](@ref).

- `method = :dense`: the full matrix, 2N unknowns (N magnetic cells). A state that is not a
  minimum is an error. A field that is not the gradient of the energy (edge-smoothed cells,
  [`isconservative`](@ref)) uses the general eigensolver, up to 2N = $(_densemax() ÷ 4).
- `method = :lanczos`: matrix-free, for large systems: the operator A⁻¹J is antisymmetric
  in the product (x, A y); its Lanczos recurrence, with full reorthogonalization, gives
  the lowest frequencies first. Every step solves A w = v by conjugate gradients
  (`cgtol`, `cgmaxiter`), which also detects a Hessian that is not positive definite. In
  Float32 the tolerances are scaled with the precision.
- `:auto`: dense up to 2N = $(_densemax() ÷ 2).
- A mode whose `residual` is above `restol` (1e-4; 1e-2 in Float32) is an error.

If `nmodes` cuts a group of modes of equal frequency, the linewidth of those returned is that of
some combination of the group: ask for the whole group.
"""
function eigenmodes(w::World{T}, m0::AbstractArray{T,4}; nmodes::Integer = 10, method::Symbol = :auto,
                    alpha = damping(w.material), torquetol = 1e-5,
                    restol = T === Float64 ? 1e-4 : 1e-2, kwargs...) where {T}
    w.current === nothing || @warn "eigenmodes: the spin-transfer torques of the world are not included"
    H = TangentHessian(w, m0)
    H.torque < torquetol || @warn "eigenmodes: the state is not an equilibrium, max|m×B| = $(H.torque) T"
    n = 2count(>(0), Array(H.sw))
    method === :auto && (method = n <= _densemax() ÷ 2 ? :dense : :lanczos)
    nmodes = min(nmodes, n ÷ 2)
    all(in((:tol, :cgtol, :cgmaxiter, :maxdim, :rng, :verbose)), keys(kwargs)) || error("eigenmodes: unknown keyword in $(keys(kwargs))")
    em = method === :dense ? _densemodes(H, nmodes, Float64(alpha)) :
         method === :lanczos ? _lanczosmodes(H, nmodes, Float64(alpha); kwargs...) :
         error("unknown method $method (:auto, :dense, :lanczos)")
    all(<(restol), em.residuals) || error("eigenmodes: a mode has a residual of $(maximum(em.residuals)). Either the " *
        "solver did not converge (maxdim, cgtol), or: $(_NOTMIN)")
    return em
end

const _NOTMIN = "The state is not a minimum, or a continuous symmetry (translation, rotation) leaves " *
                "the energy Hessian singular; see `curvatures`."

# Linewidths within groups of equal frequency: first-order perturbation theory of a degenerate
# level diagonalizes u_i*A²u_j in the group, with u_i*A u_j = c δ_ij. `S(i, j)` gives that
# matrix over c; `rot!(group, Q)` rotates the modes of the group. Ω ascending.
function _degenerate!(stretch, Ω, S, rot!; rel = 1e-7)
    i = 1
    while i <= length(Ω)
        j = i
        while j < length(Ω) && Ω[j+1] - Ω[j] < rel * Ω[j]
            j += 1
        end
        if j > i
            g = i:j
            F = eigen(Hermitian([S(a, b) for a in g, b in g]))
            stretch[g] .= F.values
            rot!(g, F.vectors)
        end
        i = j + 1
    end
    return stretch
end

_toω(Ω) = Float64(γLL) * Ω

# u (complex, frame coordinates or field) → δm, normalized.
function _modefield(re, im, H::TangentHessian)
    isw = Array(H.isw)
    δ = complex.(Float64.(isw .* re), Float64.(isw .* im))
    a = dropdims(sum(abs2, δ; dims = 1); dims = 1)
    I = argmax(a)
    c = δ[argmax(abs.(@view δ[:, I])), I]
    return δ .* (conj(c) / abs(c) / sqrt(a[I]))
end

# J in frame coordinates: e1 → e2, e2 → −e1, i.e. (J u)_{2n-1} = −u_{2n}, (J u)_{2n} = u_{2n-1}.
function _Jrows!(Y, X)
    @inbounds for j in axes(X, 2), n in 1:2:size(X, 1)
        Y[n, j] = -X[n+1, j]; Y[n+1, j] = X[n, j]
    end
    return Y
end

function _densemodes(H::TangentHessian{T}, nmodes, α) where {T}
    A, cells, E = densehessian(H)
    n = size(A, 1)
    A0 = isconservative(H.world) ? A : copy(A)
    asym = _symmetrize!(A0)
    if isconservative(H.world)
        asym < 1e3 * eps(T) || error("eigenmodes: the Hessian is not symmetric (relative asymmetry $asym)")
        F = cholesky(Symmetric(A0, :L); check = false)
        issuccess(F) || error("eigenmodes: the energy Hessian is not positive definite. $(_NOTMIN)")
        # A = L Lᵀ, y = Lᵀu: J A u = iΩ u  ⇔  (−i Lᵀ J L) y = Ω y, Hermitian.
        Anorm = maximum(sum(abs, A0; dims = 1))
        L = Matrix(F.L)
        M = L' * _Jrows!(similar(L), L)
        C = Hermitian(complex.(0.0, -(M .- M') ./ 2))
        G = eigen(C, n ÷ 2 + 1:n ÷ 2 + nmodes)        # the spectrum is ±Ω: the positive half
        Ω = G.values
        U = L' \ G.vectors
        # A singular Hessian can pass the factorization by rounding. Its zero mode then comes out as
        # a mode of very low frequency, whose energy u*A u/u*u is rounding error of A.
        soft = minimum(1 / real(dot(u, u)) for u in eachcol(U))      # u*A u = 1
        soft > 30 * Float64(eps(T)) * Anorm || error("eigenmodes: a mode of curvature $soft T, zero to rounding. $(_NOTMIN)")
        method = :dense
        AU = A0 * U                                   # u_i*A u_j = δ_ij (G.vectors unitary)
        stretch = [real(dot(x, x)) for x in eachcol(AU)]
        _degenerate!(stretch, Ω, (a, b) -> dot(view(AU, :, a), view(AU, :, b)),
                     (g, Q) -> (U[:, g] = U[:, g] * Q))
        Γ = _toω.(α / (1 + α^2) .* stretch)
    else
        n <= _densemax() ÷ 4 || error("eigenmodes: non-symmetric problem of $n unknowns; use a " *
                                      "geometry without partially filled cells or method = :lanczos")
        JA = _Jrows!(similar(A), A)
        G = eigen(JA)
        big = maximum(abs, G.values)
        sel = sort(findall(z -> imag(z) > 1e-10 * big, G.values); by = k -> imag(G.values[k]))
        # every mode must oscillate, and none may grow with or without damping
        length(sel) == n ÷ 2 || error("eigenmodes: only $(length(sel)) of $(n ÷ 2) modes oscillate. $(_NOTMIN)")
        Li = inv(G.vectors)[sel, :]                  # left vectors: rows of U⁻¹
        AV = A * G.vectors[:, sel]
        stretch = [real(sum(Li[k, :] .* AV[:, k])) for k in eachindex(sel)]
        growth = maximum(real.(G.values[sel]) .- α .* stretch)
        (all(>(0), stretch) && growth <= 1e-6 * big) ||
            error("eigenmodes: a mode grows (rate $(_toω(max(growth, 0.0))) 1/s, or negative damping). $(_NOTMIN)")
        decay = α .* stretch .- real.(G.values[sel])
        sel = sel[1:nmodes]
        Ω = imag.(G.values[sel]); U = G.vectors[:, sel]
        method = :densegeneral
        Γ = _toω.(decay[1:nmodes] ./ (1 + α^2))      # first order in α
    end
    modes = Array{ComplexF64,4}[]; res = Float64[]
    re = zeros(T, size(H.m)); im_ = zeros(T, size(H.m))
    for k in eachindex(Ω)
        u = U[:, k]
        JAu = _Jrows!(similar(u, n, 1), reshape(A * u, n, 1))[:, 1]
        push!(res, norm(JAu .- im * Ω[k] .* u) / (Ω[k] * norm(u)))
        _tofield!(re, real.(u), cells, E); _tofield!(im_, imag.(u), cells, E)
        push!(modes, _modefield(re, im_, H))
    end
    return EigenModes(_toω.(Ω) ./ 2π, Γ, modes, res, H.torque, α, method)
end

# Conjugate gradients for A w = v (A symmetric positive definite, tangent fields).
# Returns the iterations. A curvature p·Ap ≤ 0, or no convergence, is an error. The normal
# component of r, which rounding feeds, is removed at every iteration: A does not act on it,
# and it would end up as a spurious p·Ap ≤ 0.
function _cg!(w, v, H::TangentHessian, r, p, Ap; tol, maxiter)
    fill!(w, 0); copyto!(r, v); _tangentproject!(r, H); copyto!(p, r)
    rr = _gdot(r, r); r0 = rr
    r0 == 0 && return 0
    T = eltype(w)
    for it in 1:maxiter
        hessian!(Ap, p, H)
        pAp = _gdot(p, Ap)
        pAp > 0 || error("eigenmodes: a direction of curvature $(pAp / _gdot(p, p)) T was found. $(_NOTMIN)")
        a = T(rr / pAp)
        w .+= a .* p
        r .-= a .* Ap
        _tangentproject!(r, H)
        rn = _gdot(r, r)
        rn <= tol^2 * r0 && return it
        p .= r .+ T(rn / rr) .* p
        rr = rn
    end
    error("eigenmodes: conjugate gradients did not converge in $maxiter iterations (residual " *
          "$(sqrt(rr / r0)), curvature along the last direction $(_gdot(p, hessian!(Ap, p, H)) / _gdot(p, p)) T). $(_NOTMIN)")
end

# Lanczos for K = A⁻¹J, antisymmetric in (x,y)_A = x·A y: K v_j = β_j v_{j+1} − β_{j-1} v_{j-1}.
# The eigenvalues of the antisymmetric tridiagonal matrix are ±iθ, θ = 1/Ω, largest first.
# A v is recomputed for every new vector, after the reorthogonalization: carried along by
# the same linear combinations it drifts from A·v when the new vector is mostly cancelled (a
# Krylov space nearly exhausted), the basis loses its A-orthogonality and spurious modes appear.
# A single starting vector finds one copy of a degenerate frequency. So further passes
# follow, each started at random and kept A-orthogonal to the modes found (whose real and
# imaginary parts span K-invariant planes), until the lowest mode of a pass is above the
# `nmodes` already known.
function _lanczosmodes(H::TangentHessian{T}, nmodes, α; tol = max(1e-8, 100 * Float64(eps(T))),
                       cgtol = max(1e-11, 10 * Float64(eps(T))), cgmaxiter::Integer = 20_000, maxdim::Integer = max(60, 20 + 12nmodes),
                       rng = Random.MersenneTwister(1), verbose::Bool = false) where {T}
    isconservative(H.world) || error("eigenmodes: the Lanczos solver needs a symmetric Hessian " *
                                     "(no partially filled cells); use method = :dense")
    n = 2count(>(0), Array(H.sw))
    found = NamedTuple[]
    kw = (; tol, cgtol, cgmaxiter, rng, verbose)
    want = nmodes
    while 2length(found) < n
        new = _lanczospass(H, found, min(want, n ÷ 2 - length(found)), min(maxdim, n - 2length(found)); kw...)
        isempty(new) && break
        first = minimum(q.Ω for q in new)
        full = length(found) >= nmodes && first > sort([q.Ω for q in found])[nmodes]
        append!(found, new)
        full && break
        want = max(1, nmodes - length(found))
    end
    sort!(found; by = q -> q.Ω)
    k = min(nmodes, length(found))
    k == nmodes || @warn "eigenmodes: only $k of $nmodes modes were found"
    found = found[1:k]
    Ωs = [q.Ω for q in found]
    stretch = Float64[q.stretch for q in found]
    # u_i*A u_j = 2 δ_ij; A² elements from the stored images
    S(i, j) = (p = found[i]; q = found[j];
               complex(_gdot(p.Are, q.Are) + _gdot(p.Aim, q.Aim), _gdot(p.Are, q.Aim) - _gdot(p.Aim, q.Are)) / 2)
    function rot!(g, Q)
        old = found[g]
        for (c, b) in enumerate(g)
            q = old[c]
            re = fill!(similar(q.re), 0); im_ = fill!(similar(q.re), 0)
            for (r, p) in enumerate(old)
                x, y = T(real(Q[r, c])), T(imag(Q[r, c]))
                re .+= x .* p.re .- y .* p.im; im_ .+= x .* p.im .+ y .* p.re
            end
            found[b] = merge(q, (; re, im = im_))
        end
    end
    _degenerate!(stretch, Ωs, S, rot!; rel = max(1e-7, 10 * Float64(eps(T))))
    Γ = _toω.(α / (1 + α^2) .* stretch)
    modes = [_modefield(Array(q.re), Array(q.im), H) for q in found]
    return EigenModes(_toω.(Ωs) ./ 2π, Γ, modes, [q.res for q in found], H.torque, α, :lanczos)
end

# One pass: `want` modes A-orthogonal to `found`. Every mode is returned with its real and
# imaginary parts (A-orthonormal plane), their images under A, Ω, the residual
# ‖J A u − iΩ u‖/(Ω‖u‖) and the stretch u*A²u/u*Au of the linewidth.
function _lanczospass(H::TangentHessian{T}, found, want, maxdim; tol, cgtol, cgmaxiter, rng, verbose) where {T}
    mk() = similar(H.m)
    r, p, Ap, z = mk(), mk(), mk(), mk()
    function deflate!(x)
        for _ in 1:2, q in found, (f, Af) in ((q.re, q.Are), (q.im, q.Aim))
            x .-= T(_gdot(x, Af)) .* f
        end
        return _tangentproject!(x, H)
    end
    v = mk(); copyto!(v, randn(rng, T, size(H.m))); _tangentproject!(v, H)
    deflate!(v)
    Av = hessian!(mk(), v, H)
    s = _gdot(v, Av)
    s > 0 || error("eigenmodes: a direction of curvature $(s / _gdot(v, v)) T was found. $(_NOTMIN)")
    v .*= T(1 / sqrt(s)); Av .*= T(1 / sqrt(s))
    V = [v]; AV = [Av]; β = Float64[]
    ncg = 0; θ = Float64[]; Y = zeros(ComplexF64, 0, 0); err = Float64[]; invariant = false
    while true
        j = length(V)
        _cross!(z, H.m, V[j])                          # J v_j
        x = mk()
        it = _cg!(x, z, H, r, p, Ap; tol = cgtol, maxiter = cgmaxiter)
        ncg += it
        n0 = sqrt(max(_gdot(x, z), 0.0))               # A-norm of x: x·A x = x·J v
        for _ in 1:3
            deflate!(x)
            for i in 1:j                               # full reorthogonalization
                x .-= T(_gdot(x, AV[i])) .* V[i]
            end
        end
        _tangentproject!(x, H)
        Ax = hessian!(mk(), x, H)
        b2 = _gdot(x, Ax)
        bj = b2 > 0 ? sqrt(b2) : 0.0
        K = zeros(ComplexF64, j, j)                    # i times the tridiagonal matrix: Hermitian
        for i in 1:j-1
            K[i+1, i] = im * β[i]; K[i, i+1] = -im * β[i]
        end
        G = eigen(Hermitian(K))
        # (an odd dimension has one zero value, which is not a mode)
        take = [k for k in sortperm(G.values; rev = true) if G.values[k] > 1e-9 * maximum(abs, G.values)]
        θ = G.values[take]; Y = G.vectors[:, take]
        err = [bj * abs(Y[j, k]) / θ[k] for k in eachindex(θ)]
        done = length(θ) >= want && maximum(@view err[1:want]) < tol
        verbose && println("  Lanczos ", j, "  CG ", it, "  Ω = ", 1 ./ θ[1:min(end, 4)], "  err = ",
                           isempty(err) ? NaN : maximum(err[1:min(end, want)]))
        invariant = bj <= max(1e-9, 100 * Float64(eps(T))) * n0
        (done || invariant || j >= maxdim) && break
        push!(β, bj)
        x .*= T(1 / bj); Ax .*= T(1 / bj)
        push!(V, x); push!(AV, Ax)
    end
    k = min(want, length(θ))
    if k > 0 && !invariant && maximum(@view err[1:k]) >= tol
        @warn "eigenmodes: Lanczos stopped at dimension $(length(V)), error estimate $(maximum(err[1:k]))"
    end
    verbose && println("  Lanczos dimension ", length(V), ", CG iterations ", ncg)
    out = NamedTuple[]
    for q in 1:k
        # (i·tridiagonal) y = θ y ⇒ A⁻¹J u = −iθ u ⇒ J A u = −iΩ u; the conjugate has +iΩ
        re = fill!(mk(), 0); im_ = fill!(mk(), 0)
        for i in eachindex(V)
            c = conj(Y[i, q])
            re .+= T(real(c)) .* V[i]; im_ .+= T(imag(c)) .* V[i]
        end
        Are = hessian!(mk(), re, H); Aim = hessian!(mk(), im_, H)
        a, b = _gdot(re, Are), _gdot(im_, Aim)
        AuAu = _gdot(Are, Are) + _gdot(Aim, Aim)
        _cross!(z, H.m, re)
        Ω = abs((a + b) / (2 * _gdot(im_, z)))         # u*A u = −2Ω Im(u)·J Re(u)
        JAr = _cross!(mk(), H.m, Are); JAi = _cross!(mk(), H.m, Aim)
        JAr .+= T(Ω) .* im_; JAi .-= T(Ω) .* re        # J A u − iΩ u
        res = sqrt((_gdot(JAr, JAr) + _gdot(JAi, JAi)) / (_gdot(re, re) + _gdot(im_, im_))) / Ω
        a > 0 && b > 0 || error("eigenmodes: a mode without positive energy. $(_NOTMIN)")
        for (f, Af, c) in ((re, Are, a), (im_, Aim, b))
            f .*= T(1 / sqrt(c)); Af .*= T(1 / sqrt(c))
        end
        push!(out, (; Ω, re, im = im_, Are, Aim, res, stretch = AuAu / (a + b)))
    end
    return out
end

"""
    savemode(dir, em::EigenModes, k, mesh)

Write mode `k` as `dir/mode###_re.ovf` and `mode###_im.ovf`.
"""
function savemode(dir::AbstractString, em::EigenModes, k::Integer, mesh::Mesh)
    mkpath(dir)
    base = joinpath(dir, "mode" * lpad(k, 3, '0'))
    saveovf(base * "_re.ovf", real.(em.modes[k]), mesh)
    saveovf(base * "_im.ovf", imag.(em.modes[k]), mesh)
    return dir
end
