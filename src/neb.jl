# Geodesic nudged elastic band (GNEB): minimum-energy paths and energy barriers.
#
# P. F. Bessarab, V. M. Uzdin and H. Jónsson, Comput. Phys. Commun. 196, 335 (2015),
# on the product of unit spheres; tangent of G. Henkelman and H. Jónsson, J. Chem.
# Phys. 113, 9978 (2000); climbing image of Henkelman, Uberuaga and Jónsson, J. Chem.
# Phys. 113, 9901 (2000).
#
# A chain of images m⁽¹⁾ … m⁽ᴺ⁾ joins two fixed minima. Every interior image moves
# under
#     F = G − (G·τ)τ + k (L₊ − L₋) τ          (normal image)
#     F = G − 2 (G·τ)τ                        (climbing image)
# where G is the projection on the tangent planes of −∂E/∂m, τ the unit tangent of
# the path at the image and L± the geodesic distances to the neighbours. The band
# relaxes when F = 0: the images then lie on the minimum-energy path, equally spaced,
# and the climbing image sits exactly at the saddle point.
#
# Units. Forces are reduced by V·Msref (cell volume times the largest Msat), so
#     G_cell = (Msat_cell / Msref) · B_eff,⊥     [T]
# The convergence tolerance and the spring constant are then in tesla, independent
# of the system size (a uniform rotation of n cells has E ∝ n and L ∝ √n, so the
# path curvature d²E/dL² is intensive). Energies are in joules (Float64).
#
# Every per-image operation is a broadcast or a reduction over whole arrays, so the
# same code runs on CuArrays; the effective field is the World's (CPU or GPU).

"""
    GNEB(world, mA, mB; nimages=16, spring=1.0, dt=1e-13)
    GNEB(world, states::AbstractVector; nimages=16, …)

Geodesic nudged elastic band between the relaxed states `mA` and `mB` (fixed
end images), or through the given list of `states` (end states fixed, interior
ones used as waypoints of the initial path). The initial path is the per-cell
geodesic (great-circle) interpolation, `nimages` images in total, end images
included.

- `spring`: spring constant [T per unit geodesic distance]. The barrier does not
  depend on it; it only controls how evenly the images are spaced.
- `dt`: initial time-like step [s] of the optimizer: Barzilai–Borwein steps with a
  Cayley rotation of every cell, the energy minimizer's scheme applied to the whole
  chain. (It was preferred over the velocity-projection optimizer of Bessarab et
  al. 2015, which took 2–3× more steps on the Stoner–Wohlfarth problem.)

The energy-path force must be the exact gradient of the energy, which holds when
[`isconservative`](@ref)`(world)`; a warning is printed otherwise.

Relax with [`relax!`](@ref)`(neb)`; read back [`energies`](@ref), [`barrier`](@ref),
[`pathcoordinate`](@ref), [`saddle`](@ref), [`images`](@ref).
"""
mutable struct GNEB{T<:AbstractFloat,W<:World{T},A<:AbstractArray{T,4},S<:AbstractArray{T,4}}
    world::W
    m::Vector{A}            # images, 1 and end fixed
    B::Vector{A}            # effective field of each image (scratch between steps)
    F::Vector{A}            # GNEB force [T]; at the end images, G (for dE/ds)
    t::Vector{A}            # unit tangent at each image
    wr::S                   # Msat(cell)/Msref, size (1,Nx,Ny,Nz)
    mask::S                 # 1 on magnetic cells, 0 on empty ones
    d::S                    # per-cell scratch
    E::Vector{Float64}      # energy of each image [J]
    L::Vector{Float64}      # L[i]: geodesic distance image i → i+1
    Gt::Vector{Float64}     # G·τ at each image [T]: −dE/ds / (V Msref)
    spring::Float64
    climb::Int              # climbing image, 0 = none
    climbauto::Bool         # chosen by relax! (follows the top), not by the user
    Msref::Float64
    steps::Int
    rejected::Int           # steps undone by the optimizer's safeguard
    maxforce::Float64       # max over interior images and cells of |F| [T]
    # Barzilai–Borwein optimizer state
    dt::Float64             # time-like step [s]
    dt0::Float64
    dtmax::Float64          # learned stability cap on dt [s]
    bbstep::Int
    mlast::Vector{A}
    τlast::Vector{A}        # previous γF
end

function GNEB(w::World{T}, mA::AbstractArray{T,4}, mB::AbstractArray{T,4}; kwargs...) where {T}
    return GNEB(w, [mA, mB]; kwargs...)
end

function GNEB(w::World{T}, states::AbstractVector; nimages::Integer = 16, spring = 1.0,
              dt = 1e-13) where {T}
    length(states) >= 2 || error("GNEB needs at least the two end states")
    nimages >= length(states) || error("nimages must be at least the number of given states")
    nimages >= 3 || error("GNEB needs at least one free image (nimages ≥ 3)")
    spring > 0 || error("spring must be positive")
    isconservative(w) || @warn "GNEB: this world's effective field is not the exact " *
        "gradient of its energy (see `isconservative`); barriers are approximate."
    sz = (1, w.mesh.size...)
    wcpu = _msatcells(w.material, w.mesh)
    Msref = Float64(maximum(wcpu))
    Msref > 0 || error("GNEB: no magnetic cell")
    wcpu ./= T(Msref)
    m1 = states[1]
    for (k, s) in enumerate(states)      # unit vectors on every magnetic cell
        n2 = sum(abs2, Array(s); dims = 1)
        all(abs.(n2[wcpu .> 0] .- 1) .< 1e-3) ||
            error("GNEB: state $k is not normalized on the magnetic cells")
    end
    wr = copyto!(similar(m1, sz), wcpu)
    mask = copyto!(similar(m1, sz), T.(wcpu .> 0))
    # initial path on the host (done once), then copied to the array type of the states
    path = _initialpath([Array(s) for s in states], nimages)
    A = typeof(m1)
    mk() = similar(m1)::A
    ms = A[copyto!(mk(), x) for x in path]
    n = length(ms)
    neb = GNEB{T,typeof(w),A,typeof(wr)}(w, ms, [mk() for _ in 1:n], [fill!(mk(), 0) for _ in 1:n],
        [fill!(mk(), 0) for _ in 1:n], wr, mask, similar(wr),
        zeros(n), zeros(n - 1), zeros(n), Float64(spring), 0, true, Msref, 0, 0, Inf,
        Float64(dt), Float64(dt), Inf, 0, [mk() for _ in 1:n], [fill!(mk(), 0) for _ in 1:n])
    _updateends!(neb)
    _update!(neb)
    return neb
end

# End images: energy and G. Done at construction and at the start of relax! (the
# applied field of the world may have changed in between).
function _updateends!(neb::GNEB)
    for i in (1, length(neb.m))
        neb.E[i] = energy!(neb.B[i], neb.m[i], neb.world)
        _gradforce!(neb.F[i], neb.m[i], neb.B[i], neb)
    end
    return neb
end

# --- initial path ------------------------------------------------------------------

# Per-cell great-circle interpolation m(s) = cos(sθ) a + sin(sθ) p, p the unit
# direction of b − (a·b)a. For antiparallel a, b (no unique great circle) p is the
# Cartesian axis least aligned with a, projected: deterministic, so a uniform state
# rotates coherently. Computed in Float64 and normalized whatever T is: in Float32, p of
# a nearly antiparallel pair is mostly rounding noise along a, which would leave
# |m| − 1 up to several per cent.
function _slerp!(out, a, b, s)
    tol = 4 * Float64(eps(eltype(a)))          # rounding of b − c·a for unit a, b
    @inbounds for I in CartesianIndices(axes(a)[2:4])
        ax, ay, az = Float64(a[1, I]), Float64(a[2, I]), Float64(a[3, I])
        bx, by, bz = Float64(b[1, I]), Float64(b[2, I]), Float64(b[3, I])
        na = sqrt(ax^2 + ay^2 + az^2); nb = sqrt(bx^2 + by^2 + bz^2)
        if na == 0 || nb == 0                  # empty cell
            out[1, I], out[2, I], out[3, I] = a[1, I], a[2, I], a[3, I]
            continue
        end
        ax /= na; ay /= na; az /= na; bx /= nb; by /= nb; bz /= nb
        c = clamp(ax*bx + ay*by + az*bz, -1.0, 1.0)
        px, py, pz = bx - c*ax, by - c*ay, bz - c*az
        pn = sqrt(px^2 + py^2 + pz^2)
        θ = atan(pn, c)
        if pn < max(tol, 1e-6) && c < 0
            ex, ey, ez = abs(ax) <= abs(ay) && abs(ax) <= abs(az) ? (1, 0, 0) :
                         abs(ay) <= abs(az) ? (0, 1, 0) : (0, 0, 1)
            r = ex*ax + ey*ay + ez*az
            px, py, pz = ex - r*ax, ey - r*ay, ez - r*az
            pn = sqrt(px^2 + py^2 + pz^2)
        end
        if pn == 0
            out[1, I], out[2, I], out[3, I] = a[1, I], a[2, I], a[3, I]
        else
            cs, sn = cos(s*θ), sin(s*θ) / pn
            mx, my, mz = cs*ax + sn*px, cs*ay + sn*py, cs*az + sn*pz
            q = 1 / sqrt(mx^2 + my^2 + mz^2)
            out[1, I] = q*mx; out[2, I] = q*my; out[3, I] = q*mz
        end
    end
    return out
end

# nimages images through the given states, the free images split over the segments
# in proportion to their mean per-cell geodesic length.
function _initialpath(states, nimages)
    nseg = length(states) - 1
    len = [max(sqrt(_geodist2(states[s], states[s+1])), eps()) for s in 1:nseg]
    nfree = nimages - length(states)
    share = len ./ sum(len) .* nfree
    k = floor.(Int, share)
    for _ in 1:(nfree - sum(k))                   # largest remainders
        r = share .- k
        k[argmax(r)] += 1
        share[argmax(r)] = -Inf
    end
    path = [copy(states[1])]
    for s in 1:nseg
        for j in 1:k[s]
            push!(path, _slerp!(similar(states[s]), states[s], states[s+1], j / (k[s] + 1)))
        end
        push!(path, copy(states[s+1]))
    end
    return path
end

# Σ θ² over the cells (host arrays, any cell). Used only to split the initial path.
function _geodist2(a, b)
    s = 0.0
    @inbounds for I in CartesianIndices(axes(a)[2:4])
        cx = a[2, I]*b[3, I] - a[3, I]*b[2, I]
        cy = a[3, I]*b[1, I] - a[1, I]*b[3, I]
        cz = a[1, I]*b[2, I] - a[2, I]*b[1, I]
        s += Float64(atan(sqrt(cx^2 + cy^2 + cz^2), a[1, I]*b[1, I] + a[2, I]*b[2, I] + a[3, I]*b[3, I]))^2
    end
    return s
end

# --- per-image operations (broadcasts: CPU and GPU) ---------------------------------

@inline _cmp(a, c) = view(a, c:c, :, :, :)

# d = a·b per cell
function _celldot!(d, a, b)
    d .= _cmp(a, 1) .* _cmp(b, 1) .+ _cmp(a, 2) .* _cmp(b, 2) .+ _cmp(a, 3) .* _cmp(b, 3)
    return d
end

_gdot(a, b) = mapreduce((x, y) -> Float64(x) * Float64(y), +, a, b)

# Geodesic distance sqrt(Σ_magnetic cells θ²), θ = atan(|a×b|, a·b).
function _geodist(neb::GNEB, a, b)
    ax, ay, az = _cmp(a, 1), _cmp(a, 2), _cmp(a, 3)
    bx, by, bz = _cmp(b, 1), _cmp(b, 2), _cmp(b, 3)
    neb.d .= neb.mask .* atan.(sqrt.((ay .* bz .- az .* by) .^ 2 .+ (az .* bx .- ax .* bz) .^ 2 .+
                                     (ax .* by .- ay .* bx) .^ 2),
                               ax .* bx .+ ay .* by .+ az .* bz) .^ 2
    return sqrt(sum(Float64, neb.d))
end

# Projection on the tangent plane of m, written as −m×(m×X) = |m|²X − (m·X)m: exactly
# orthogonal to m even when |m| differs from 1 by rounding. (X − (m·X)m is not, and
# the Cayley step then amplifies the |m| error geometrically, ×1.6 per step on the
# test problems.) Optionally masked. Uses neb.d.
function _project!(X, m, neb::GNEB; mask::Bool = true)
    mx, my, mz = _cmp(m, 1), _cmp(m, 2), _cmp(m, 3)
    _celldot!(neb.d, m, X)
    if mask
        X .= neb.mask .* ((mx .^ 2 .+ my .^ 2 .+ mz .^ 2) .* X .- neb.d .* m)
    else
        X .= (mx .^ 2 .+ my .^ 2 .+ mz .^ 2) .* X .- neb.d .* m
    end
    return X
end

# G = (Msat/Msref) (B − (m·B) m): the reduced, projected −∂E/∂m.
function _gradforce!(G, m, B, neb::GNEB)
    G .= neb.wr .* B
    return _project!(G, m, neb; mask = false)
end

# max over cells of |a|
function _maxcellnorm(neb::GNEB, a)
    neb.d .= _cmp(a, 1) .^ 2 .+ _cmp(a, 2) .^ 2 .+ _cmp(a, 3) .^ 2
    return sqrt(Float64(maximum(neb.d)))
end

# Energy-weighted upwind tangent (Henkelman & Jónsson 2000), projected on the
# tangent planes of image i and normalized over the image.
function _tangent!(neb::GNEB{T}, i) where {T}
    t, m, mp, mm = neb.t[i], neb.m[i], neb.m[i+1], neb.m[i-1]
    Ep, E0, Em = neb.E[i+1], neb.E[i], neb.E[i-1]
    if Ep > E0 > Em
        a, b = 1.0, 0.0
    elseif Ep < E0 < Em
        a, b = 0.0, 1.0
    elseif (Ep < E0 && Em < E0) || (Ep > E0 && Em > E0)
        dmax = max(abs(Ep - E0), abs(Em - E0)); dmin = min(abs(Ep - E0), abs(Em - E0))
        a, b = Ep > Em ? (dmax, dmin) : (dmin, dmax)
    else
        a, b = 1.0, 1.0
    end
    s = max(a, b)                          # keep the weights O(1) in T
    ta, tb = T(a / s), T(b / s)
    t .= ta .* (mp .- m) .+ tb .* (m .- mm)
    _project!(t, m, neb)
    tn = sqrt(_gdot(t, t))
    (tn > 0 || isnan(tn)) ||               # NaN: a diverged step, undone by the optimizer
        error("GNEB: the path tangent vanished at image $i (coincident images?)")
    t .*= T(1 / tn)
    return t
end

# End-image tangent (one-sided), for the dE/ds of the saddle interpolation only.
function _endtangent!(neb::GNEB{T}, i) where {T}
    t = neb.t[i]
    i == 1 ? (t .= neb.m[2] .- neb.m[1]) : (t .= neb.m[i] .- neb.m[i-1])
    _project!(t, neb.m[i], neb)
    tn = sqrt(_gdot(t, t))
    tn > 0 && (t .*= T(1 / tn))
    return t
end

# Energies, fields, distances, tangents and forces of the interior images.
function _update!(neb::GNEB{T}) where {T}
    n = length(neb.m)
    for i in 2:n-1
        neb.E[i] = energy!(neb.B[i], neb.m[i], neb.world)
    end
    for i in 1:n-1
        neb.L[i] = _geodist(neb, neb.m[i], neb.m[i+1])
    end
    fmax = 0.0
    for i in 2:n-1
        t = _tangent!(neb, i)
        F = _gradforce!(neb.F[i], neb.m[i], neb.B[i], neb)
        gt = _gdot(F, t)
        neb.Gt[i] = gt
        c = i == neb.climb ? -2gt : -gt + neb.spring * (neb.L[i] - neb.L[i-1])
        F .+= T(c) .* t
        fmax = max(fmax, _maxcellnorm(neb, F))
    end
    for i in (1, n)
        neb.Gt[i] = _gdot(neb.F[i], _endtangent!(neb, i))
    end
    neb.maxforce = fmax
    return neb
end

# --- optimizers ----------------------------------------------------------------------

# Barzilai–Borwein on the whole chain: τ = γF is the descent direction, each cell is
# rotated by the Cayley step (|m| = 1 exactly), dt from the BB secant of the chain
# (alternating ss/sy and sy/yy, as the minimizer and mumax3). The rotation per step is
# capped at `maxrot` rad.
#
# Safeguard: a step that makes the force non-finite, or (optionally) raises the max
# force by more than `reject` times, is undone, and dt is capped at half the rejected
# step from then on (the cap relaxes by 2 % per accepted step). The default rejects
# only non-finite steps: BB is non-monotone by design, and where it converges a finite
# threshold costs up to 2–8× more evaluations (gpu_validation/gneb/reject_study.jl:
# ellipses, skyrmion and skyrmion lattice, thresholds 10 to 1.5; 100 changes little).
# Where it does not converge, a threshold may or may not help, with no trend: lattice,
# 21 images, k = 0.3: 10, 3 and 1.5 reach the right barrier, 100 does not; skyrmion, 11
# images: only 3 does (100 and 10 do not converge, 1.5 converges to a band that skips the
# saddle). More images fixed the latter.
function _bbstep!(neb::GNEB{T}, maxrot, reject) where {T}
    n = length(neb.m)
    γ = Float64(γLL)
    dt = min(neb.dt, neb.dtmax, maxrot / (γ * max(neb.maxforce, floatmin())))
    oldf = neb.maxforce
    for i in 2:n-1
        copyto!(neb.mlast[i], neb.m[i])
        neb.τlast[i] .= T(γ) .* neb.F[i]
        _cayley_step!(neb.m[i], neb.τlast[i], T(dt))
    end
    _update!(neb)
    if !(isfinite(neb.maxforce) && (reject == Inf || neb.maxforce <= reject * oldf))
        for i in 2:n-1
            copyto!(neb.m[i], neb.mlast[i])
        end
        _update!(neb)
        neb.rejected += 1
        neb.dtmax = neb.dt = 0.5 * dt
        neb.bbstep = 0
        return false
    end
    neb.dtmax = 1.02 * neb.dtmax
    ss = sy = yy = 0.0
    for i in 2:n-1
        neb.B[i] .= T(γ) .* neb.F[i]            # B is free until the next _update!
        a, b, c = _bb_sums(neb.m[i], neb.mlast[i], neb.τlast[i], neb.B[i])
        ss += a; sy += b; yy += c
    end
    if sy != 0 && yy != 0
        neb.dt = iseven(neb.bbstep) ? abs(ss / sy) : abs(sy / yy)
    end
    (isfinite(neb.dt) && neb.dt > 0) || (neb.dt = neb.dt0)
    neb.bbstep += 1
    return true
end

"""
    relax!(neb::GNEB; stopforce=1e-4, climbing=true, climbforce=10stopforce,
           maxsteps=50_000, maxrot=0.1, reject=Inf, verbose=false) -> neb

Relax the band until the largest GNEB force on any cell of any free image is below
`stopforce` [T]. With `climbing`, the highest-energy image becomes a climbing image
once the force is below `climbforce`, and then converges onto the saddle point; if
another image becomes the highest, it takes over. An image chosen with
[`setclimbing!`](@ref)`(neb, i)` stays the climbing image (a warning is printed if
another one ends up higher: `barrier` is then not a saddle energy). If no free image is
above both ends there is none to climb with: the path is relaxed without a climbing
image, with a warning. Either it has no barrier or the barrier falls between two images;
with `climbing`, a warning is printed when the energy halfway between two neighbouring
images shows that it may ([`midpointenergies`](@ref)).
`maxrot` caps the rotation of any cell in one step [rad]; a step that makes the
force non-finite, or raises it more than `reject` times (> 1), is undone and the step
size capped (an undone step costs two evaluations of the band and is counted in both
`neb.steps` and `neb.rejected`). If a band does not converge, use more images first; a
finite `reject` (10, 3) rescues some bands, not all. Ends after `maxsteps` (with a warning).
"""
function relax!(neb::GNEB; stopforce = 1e-4, climbing::Bool = true,
                climbforce = 10stopforce, maxsteps::Integer = 50_000, maxrot = 0.1,
                reject = Inf, verbose::Bool = false)
    reject > 1 || error("reject must be larger than 1")
    t0 = neb.steps
    _updateends!(neb)
    !climbing && neb.climb != 0 ? setclimbing!(neb, 0) : _update!(neb)
    moves = 0                                # re-selections of the climbing image
    while neb.steps - t0 < maxsteps
        n = length(neb.m)
        top = argmax(@view neb.E[2:n-1]) + 1
        hasmax = neb.E[top] > max(neb.E[1], neb.E[n])
        if climbing && hasmax && neb.climb == 0 && neb.maxforce < climbforce
            setclimbing!(neb)
            verbose && println("  GNEB step ", neb.steps, ": climbing image ", neb.climb)
        elseif neb.climb != 0 && neb.climbauto && neb.climb != top && moves < n
            setclimbing!(neb); moves += 1
            verbose && println("  GNEB step ", neb.steps, ": climbing image moved to ", neb.climb)
        end
        if neb.maxforce < stopforce && (!climbing || !hasmax || neb.climb != 0)
            climbing && !hasmax &&
                @warn "GNEB: no free image is above both ends; no climbing image was used"
            climbing && hasmax && neb.climb != top &&
                @warn "GNEB: image $top is above the climbing image $(neb.climb)"
            break
        end
        _bbstep!(neb, maxrot, reject)
        neb.steps += 1
        if verbose && neb.steps % 200 == 0
            println("  GNEB step ", neb.steps, "  max|F| = ", neb.maxforce, " T  ΔE = ",
                    barrier(neb), " J")
            flush(stdout)
        end
    end
    neb.maxforce < stopforce ||
        @warn "GNEB did not converge in $maxsteps steps: max|F| = $(neb.maxforce) T"
    climbing && _warnhidden(neb)
    return neb
end

# A maximum between two neighbouring images does not act on the forces: with too few
# images, and an energy that is flat along some mode (a translation), the images slide to
# the flat part and leave the whole transition to a single jump. The band is then relaxed,
# its barrier is wrong, and the Hermite interpolation (flat at both ends) shows nothing.
# The geodesic between two images is not the minimum-energy path, so a coarse but correct
# band can also have a midpoint above its saddle: on a flat-topped path (a wall crossing a
# wire) by 24, 7.6, 2.6, 0.6, 0.09 % of the barrier with 5, 7, 9, 13, 21 images, the barrier
# being the same. Hence a threshold of 10 %, of the smaller of the forward and backward
# barriers: the whole energy range would hide a maximum at the top of a long descent. If
# no image is above both ends, any excess beyond rounding is reported.
function _warnhidden(neb::GNEB{T}) where {T}
    Em = midpointenergies(neb)
    top, i = findmax(Em)
    Emax = maximum(neb.E)
    tol = 0.1 * (Emax - max(neb.E[1], neb.E[end])) + 8 * Float64(eps(T)) * abs(Emax)
    top > Emax + tol &&
        @warn "GNEB: the energy halfway between images $i and $(i+1) is higher than at any " *
              "image, by $(top - Emax) J: the band may skip a maximum there, and `barrier` " *
              "be too low. Check with more images."
    return neb
end

"""
    midpointenergies(neb) -> Vector{Float64}

Energies [J] halfway along the geodesics between neighbouring images (n − 1 of them).
A value far above every image energy means that the images may skip a maximum;
[`relax!`](@ref) warns when the excess is more than 10 % of the barrier (the smaller of
the forward and backward ones).
(A small excess also occurs in correct bands, because the geodesic is not the
minimum-energy path; it vanishes with more images.)
"""
function midpointenergies(neb::GNEB{T}) where {T}
    x, B = neb.mlast[1], neb.B[1]            # unused by the optimizer (end image)
    return map(1:length(neb.m)-1) do i
        a = neb.m[i]
        x .= a .+ neb.m[i+1]
        _celldot!(neb.d, x, x)
        # antiparallel (no unique midpoint) or empty cell: keep image i
        x .= ifelse.(neb.d .> T(1e-8), x ./ sqrt.(max.(neb.d, T(1e-8))), a)
        energy!(B, x, neb.world)
    end
end

"""
    setclimbing!(neb, i=:auto)

Make image `i` the climbing image; it then stays so (e.g. to climb a lower one of two
maxima). With `:auto`, the highest-energy free image, followed if another one becomes
the highest; `0` switches climbing off. The optimizer state is reset.
"""
function setclimbing!(neb::GNEB, i = :auto)
    n = length(neb.m)
    j = i === :auto ? argmax(@view neb.E[2:n-1]) + 1 : Int(i)
    (j == 0 || 2 <= j <= n - 1) || error("the climbing image must be a free image (2…$(n-1))")
    neb.climb = j
    neb.climbauto = i === :auto || j == 0
    _resetopt!(neb)
    _update!(neb)
    return j
end

function _resetopt!(neb::GNEB)
    neb.dt = min(neb.dt0, neb.dtmax); neb.bbstep = 0
    foreach(x -> fill!(x, 0), neb.τlast)
    return neb
end

# --- results ---------------------------------------------------------------------

"Energies of the images [J], end images included."
energies(neb::GNEB) = copy(neb.E)

"Path coordinate of each image: cumulative geodesic distance from the first image."
pathcoordinate(neb::GNEB) = [0.0; cumsum(neb.L)]

"The images (on the host)."
images(neb::GNEB) = [Array(m) for m in neb.m]

"""
    barrier(neb) -> ΔE [J]

Highest image energy minus the energy of the first image. With a converged climbing
image this is the barrier at the saddle point.
"""
barrier(neb::GNEB) = maximum(neb.E) - neb.E[1]

"""
    saddle(neb; samples=200) -> (s, E)

Maximum of the cubic Hermite interpolation of the energy along the path, built from
the image energies and the tangential forces dE/ds = −V·Msref·(G·τ).
"""
function saddle(neb::GNEB; samples::Integer = 200)
    s = pathcoordinate(neb)
    dE = -cellvolume(neb.world.mesh) * neb.Msref .* neb.Gt
    best = (s[1], neb.E[1])
    for i in 1:length(s)-1
        h = s[i+1] - s[i]
        for k in 0:samples
            u = k / samples
            h00 = 2u^3 - 3u^2 + 1; h10 = u^3 - 2u^2 + u; h01 = -2u^3 + 3u^2; h11 = u^3 - u^2
            E = h00 * neb.E[i] + h10 * h * dE[i] + h01 * neb.E[i+1] + h11 * h * dE[i+1]
            E > best[2] && (best = (s[i] + u * h, E))
        end
    end
    return best
end

"""
    saveimages(neb, dir; prefix="image")

Write every image as `dir/image###.ovf` and the table `dir/neb.txt` (image, path
coordinate, energy, energy minus the first image's).
"""
function saveimages(neb::GNEB, dir::AbstractString; prefix = "image")
    mkpath(dir)
    mesh = neb.world.mesh
    for (i, m) in enumerate(neb.m)
        saveovf(joinpath(dir, string(prefix, lpad(i - 1, 3, '0'), ".ovf")), Array(m), mesh)
    end
    s = pathcoordinate(neb)
    open(joinpath(dir, "neb.txt"), "w") do io
        println(io, "# image\ts ()\tE (J)\tE-E0 (J)")
        for i in eachindex(neb.E)
            println(io, i - 1, '\t', s[i], '\t', neb.E[i], '\t', neb.E[i] - neb.E[1])
        end
    end
    return dir
end
