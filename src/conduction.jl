# Electric conduction with anisotropic magnetoresistance (AMR).
#
# At every instant the potential φ obeys charge conservation in the conducting cells,
#
#     ∇·(σ(m) ∇φ) = 0,     j = −σ(m) ∇φ,
#     σ = ρ⁻¹,  ρ = ρ⊥ I + (ρ∥ − ρ⊥) m mᵀ   ⇒   σ = (1/ρ⊥) [I − (Δρ/ρ∥) m mᵀ],   Δρ = ρ∥ − ρ⊥
#
# (quasi-static: charges relax in femtoseconds). ρ∥ and ρ⊥ are given per region; a region without them
# is an insulator, a region without Msat (m = 0) conducts isotropically with ρ⊥. Contacts are regions
# held at a potential, or fed a total current (their cells share one unknown potential). All other
# boundaries are insulating, n·j = 0.
#
# Discretization: finite volumes on the cells. The current through the face between cells P and Q
# (along axis d, area A_d, spacing h_d) is
#
#     F_PQ = −A_d [ σ_dd (φ_Q − φ_P)/h_d + Σ_{e≠d} σ_de g_e ],
#
# with σ the mean of the two cells' tensors and g_e the mean of the two cells' gradients along e
# (central differences, one-sided next to an insulator). F_QP = −F_PQ to the last bit, so current is
# conserved cell by cell. A contact is a perfect conductor: its potential holds on its faces, and the
# current through a face of a contact is that of the cell on the other side, over half a cell. The off-diagonal terms (planar Hall effect) make the operator not symmetric:
# it is solved by BiCGSTAB with a Jacobi preconditioner, from the last solution.
#
# Contacts: by superposition. φ = φ₀ + Σ_c V_c ψ_c, with φ₀ the solution with the potential contacts at
# their values and the current contacts at 0, ψ_c the solution with contact c at 1 and all others at 0;
# the potentials V_c of the current contacts then follow from their currents (a small linear system).
# Without a potential contact the last current contact is the ground (0 V); its current is minus the sum
# of the others.

"""
    Contact

A region held at a potential (`kind = :potential`, volts) or fed a total current (`:current`, amperes,
positive into the sample), constant or a function of time. After a solve, `V` and `I` hold its
potential and the current it feeds into the sample.
"""
mutable struct Contact
    region::Int
    kind::Symbol
    f::Any
    value::Float64
    V::Float64
    I::Float64
end

"""
    Conduction

Resistivities per region, contacts, and the state of the solver of [`setconduction!`](@ref). Fields
of interest: `phi` (1,Nx,Ny,Nz) [V], `j` (3,Nx,Ny,Nz) [A/m²], `contacts`, `iterations` (last solve).
"""
mutable struct Conduction{A}
    rhopar::Vector{Float64}          # per region (Inf: insulator)
    rhoperp::Vector{Float64}
    contacts::Vector{Contact}
    code::Any                        # per cell: 0 insulator, 1 free, 1 + c contact c (Int32, device)
    sp::A                            # per cell 1/ρ⊥ (0 in an insulator), (1,Nx,Ny,Nz)
    an::A                            # per cell Δρ/ρ∥
    phi::A                           # potential [V]
    j::A                             # current density [A/m²]
    basis::Vector{A}                 # φ₀ and the ψ_c (warm starts)
    G::A                             # cell gradients (scratch)
    r::A; r0::A; p::A; v::A; s::A; t::A; y::A; zz::A; dg::A      # BiCGSTAB
    tol::Float64
    maxiter::Int
    iterations::Int
    ready::Bool                      # cell codes and resistivities up to date with the regions
end

"""
    setconduction!(sim; rho_par, rho_perp, regions=:magnetic, tol=1e-13, maxiter=20_000)

Electric conduction with anisotropic magnetoresistance in the cells of `regions` (`:magnetic`: every
region with Msat; or a list of region ids), with resistivities `rho_par` (m ∥ j) and `rho_perp` [Ω m].
Per region with [`setresistivity!`](@ref); contacts with [`addcontact!`](@ref). The potential is solved
at every stage of the solvers when a current is set ([`setcurrent!`](@ref): its spin torques then take
the current density of the conduction, added to `J` and the terms of [`addcurrent!`](@ref)), and at
every output. Table columns: [`q_contacts`](@ref), [`q_resistance`](@ref); fields: [`potentialmap`](@ref),
[`currentmap`](@ref).
"""
function setconduction!(w::World{T}; rho_par, rho_perp, regions = :magnetic, tol = T === Float64 ? 1e-13 : 1e-6,
                        maxiter::Integer = 20_000) where {T}
    rpar = fill(Inf, MAXREGIONS); rperp = fill(Inf, MAXREGIONS)
    ids = regions === :magnetic ? [r - 1 for r in 1:MAXREGIONS if _regionmsat(w.material, r - 1) != 0] : collect(regions)
    for id in ids
        rpar[id+1] = rho_par; rperp[id+1] = rho_perp
    end
    z = zeros(T, 1, w.mesh.size...)
    dev(a) = copyto!(similar(w._Bbuf, size(a)), a)
    sz1 = dev(z); sz3 = dev(zeros(T, 3, w.mesh.size...))
    mk() = similar(sz1)
    w.conduction = Conduction{typeof(sz1)}(rpar, rperp, Contact[], nothing, mk(), mk(), fill!(mk(), 0), fill!(similar(sz3), 0),
                                           typeof(sz1)[], similar(sz3), mk(), mk(), mk(), mk(), mk(), mk(), mk(), mk(), mk(),
                                           Float64(tol), Int(maxiter), 0, false)
    return w
end
_regionmsat(m::Material, id) = id == 0 ? m.Msat : zero(m.Msat)
_regionmsat(rp::RegionParams, id) = (id + 1 in _present(rp)) ? rp.Msat[id+1] : zero(eltype(rp))

"""
    setresistivity!(sim, region; rho_par, rho_perp=rho_par)

The resistivities [Ω m] of one region; `Inf` makes it an insulator. A lead without Msat conducts
isotropically with `rho_perp`.
"""
function setresistivity!(w::World, region::Integer; rho_par, rho_perp = rho_par)
    c = _conduction(w)
    c.rhopar[region+1] = rho_par; c.rhoperp[region+1] = rho_perp
    c.ready = false
    return w
end

"""
    addcontact!(sim, region; potential=nothing, current=nothing)

Make `region` a contact: held at `potential` [V], or fed the total `current` [A] (positive into the
sample). Each is a number or a function of time. A contact conducts as a perfect conductor.
"""
function addcontact!(w::World, region::Integer; potential = nothing, current = nothing, t = 0.0)
    c = _conduction(w)
    (potential === nothing) == (current === nothing) && throw(ArgumentError("addcontact!: give either potential or current"))
    any(k -> k.region == region, c.contacts) && throw(ArgumentError("addcontact!: region $region is already a contact"))
    kind = potential === nothing ? :current : :potential
    f = something(potential, current)
    push!(c.contacts, Contact(region, kind, f, _factor(f, t), 0.0, 0.0))
    c.ready = false
    return w
end

_conduction(w::World) = w.conduction === nothing ? throw(ArgumentError("call setconduction! first")) : w.conduction

# --- cells ----------------------------------------------------------------------------------------

# Codes and resistivities per cell from the regions (host), then to the device.
function _setupcells!(c::Conduction, w::World)
    mesh = w.mesh; n = mesh.size
    ids = _regionids(w.material, mesh)
    fl = _cellfill(w.material)
    code = zeros(Int32, 1, n...); sp = zeros(1, n...); an = zeros(1, n...)
    for I in CartesianIndices(n)
        r = ids[I]
        fl === nothing || fl[I] > 0 || continue
        isfinite(c.rhoperp[r+1]) || continue
        k = findfirst(q -> q.region == r, c.contacts)
        code[1, I] = k === nothing ? 1 : 1 + k
        sp[1, I] = 1 / c.rhoperp[r+1]
        an[1, I] = (c.rhopar[r+1] - c.rhoperp[r+1]) / c.rhopar[r+1]
    end
    for (k, q) in enumerate(c.contacts)
        any(==(1 + k), code) || error("contact $k: region $(q.region) has no conducting cell")
    end
    any(==(1), code) || error("conduction: no conducting cell outside the contacts")
    for I in CartesianIndices(n), d in 1:3                 # two contacts must not touch
        J = ntuple(e -> e == d ? I[e] + 1 : I[e], 3)
        J[d] <= n[d] || continue
        a = code[1, I]; b = code[1, J...]
        a > 1 && b > 1 && a != b && error("conduction: contacts $(a - 1) and $(b - 1) touch")
    end
    isempty(c.contacts) && error("conduction: add at least one contact (addcontact!)")
    T = eltype(c.sp)
    c.code = copyto!(similar(c.sp, Int32, size(code)), code)
    copyto!(c.sp, T.(sp)); copyto!(c.an, T.(an))
    nb = 1 + count(q -> q.kind === :current, c.contacts)
    while length(c.basis) < nb
        push!(c.basis, fill!(similar(c.sp), 0))
    end
    c.ready = true
    return c
end
_regionids(::Material, mesh) = zeros(Int, mesh.size...)
_regionids(rp::RegionParams, mesh) = Int.(rp.regions.id)

# --- the operator, CPU (the CUDA extension has the same as kernels) ---------------------------------

@inline _nb(i, n, p, s) = (k = i + s; 1 <= k <= n ? k : (p ? mod1(k, n) : 0))

# Cell gradients of φ, central differences between conducting neighbours, one-sided next to an insulator.
function _cellgradients!(G::Array{T,4}, phi::Array{T,4}, code, mesh::Mesh) where {T}
    n = mesh.size; h = mesh.cellsize; per = ntuple(d -> isperiodic(mesh, d), 3)
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        I = (i, j, k)
        for d in 1:3
            g = zero(T)
            if code[1, i, j, k] != 0 && n[d] > 1
                a = _nb(I[d], n[d], per[d], -1); b = _nb(I[d], n[d], per[d], +1)
                Ia = ntuple(e -> e == d ? a : I[e], 3); Ib = ntuple(e -> e == d ? b : I[e], 3)
                ca = a != 0 && code[1, Ia...] != 0; cb = b != 0 && code[1, Ib...] != 0
                # a contact holds its potential on its face, half a cell away
                da = T(ca && code[1, Ia...] > 1 ? h[d] / 2 : h[d]); db = T(cb && code[1, Ib...] > 1 ? h[d] / 2 : h[d])
                if code[1, i, j, k] > 1
                    g = zero(T)                                   # inside a contact: equipotential
                elseif ca && cb
                    g = (phi[1, Ib...] - phi[1, Ia...]) / (da + db)
                elseif cb
                    g = (phi[1, Ib...] - phi[1, i, j, k]) / db
                elseif ca
                    g = (phi[1, i, j, k] - phi[1, Ia...]) / da
                end
            end
            G[d, i, j, k] = g
        end
    end
    return G
end

# The face current from P to its neighbour Q along +d (both conducting), A. cP, cQ: their codes.
@inline function _faceflux(::Type{T}, d, cP, cQ, mP, mQ, spP, spQ, anP, anQ, φP, φQ, GP, GQ, area, h) where {T}
    cP > 1 && cQ > 1 && return zero(T)          # two contacts do not touch (checked at the setup)
    if cP > 1 || cQ > 1
        # a contact: its potential on the face, the tensor and the gradient of the other cell
        m_, sp_, an_, G_ = cP > 1 ? (mQ, spQ, anQ, GQ) : (mP, spP, anP, GP)
        σd = ntuple(e -> sp_ * ((e == d) - an_ * m_[d] * m_[e]), 3)
        f = σd[d] * (φQ - φP) / T(h / 2)
        for e in 1:3
            e == d && continue
            f += σd[e] * G_[e]
        end
        return -T(area) * f
    end
    # mean of the two tensors σ = sp (I − an m mᵀ)
    σd = ntuple(e -> T(0.5) * (spP * ((e == d) - anP * mP[d] * mP[e]) + spQ * ((e == d) - anQ * mQ[d] * mQ[e])), 3)
    f = σd[d] * (φQ - φP) / T(h)
    for e in 1:3
        e == d && continue
        f += σd[e] * T(0.5) * (GP[e] + GQ[e])
    end
    return -T(area) * f
end

# out[P] = Σ_faces F_out(P) for the conducting cells; with `curr`, the current density of each cell instead.
function _divergence!(out::Array{T,4}, phi::Array{T,4}, G::Array{T,4}, m::Array{T,4}, code, sp, an, mesh::Mesh;
                      curr::Union{Nothing,Array{T,4}} = nothing) where {T}
    n = mesh.size; h = mesh.cellsize; per = ntuple(d -> isperiodic(mesh, d), 3)
    area = (h[2] * h[3], h[1] * h[3], h[1] * h[2])
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        I = (i, j, k)
        if code[1, i, j, k] == 0
            curr === nothing ? (out[1, i, j, k] = 0) : (curr[1, i, j, k] = curr[2, i, j, k] = curr[3, i, j, k] = 0)
            continue
        end
        mP = (m[1, i, j, k], m[2, i, j, k], m[3, i, j, k]); GP = (G[1, i, j, k], G[2, i, j, k], G[3, i, j, k])
        s = zero(T); jc = (zero(T), zero(T), zero(T))
        for d in 1:3
            n[d] > 1 || continue
            fd = zero(T)
            for sgn in (-1, 1)
                b = _nb(I[d], n[d], per[d], sgn)
                b == 0 && continue
                Q = ntuple(e -> e == d ? b : I[e], 3)
                code[1, Q...] == 0 && continue
                mQ = (m[1, Q...], m[2, Q...], m[3, Q...]); GQ = (G[1, Q...], G[2, Q...], G[3, Q...])
                # always evaluate the face from its lower cell, so both cells get the same number
                cP = code[1, i, j, k]; cQ = code[1, Q...]
                F = sgn > 0 ? _faceflux(T, d, cP, cQ, mP, mQ, sp[1, i, j, k], sp[1, Q...], an[1, i, j, k], an[1, Q...], phi[1, i, j, k], phi[1, Q...], GP, GQ, area[d], h[d]) :
                              -_faceflux(T, d, cQ, cP, mQ, mP, sp[1, Q...], sp[1, i, j, k], an[1, Q...], an[1, i, j, k], phi[1, Q...], phi[1, i, j, k], GQ, GP, area[d], h[d])
                s += F
                fd += sgn > 0 ? F : -F          # current along +d through this face
            end
            jc = Base.setindex(jc, fd / T(2area[d]), d)
        end
        if curr === nothing
            out[1, i, j, k] = s
        else
            curr[1, i, j, k] = jc[1]; curr[2, i, j, k] = jc[2]; curr[3, i, j, k] = jc[3]
        end
    end
    return curr === nothing ? out : curr
end

# Jacobi diagonal: Σ_faces A_d σ_dd / h_d
function _jacobi!(dg::Array{T,4}, m::Array{T,4}, code, sp, an, mesh::Mesh) where {T}
    n = mesh.size; h = mesh.cellsize; per = ntuple(d -> isperiodic(mesh, d), 3)
    area = (h[2] * h[3], h[1] * h[3], h[1] * h[2])
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        I = (i, j, k); s = zero(T)
        if code[1, i, j, k] == 1
            for d in 1:3, sgn in (-1, 1)
                n[d] > 1 || continue
                b = _nb(I[d], n[d], per[d], sgn); b == 0 && continue
                Q = ntuple(e -> e == d ? b : I[e], 3)
                code[1, Q...] == 0 && continue
                σP = sp[1, i, j, k] * (1 - an[1, i, j, k] * m[d, i, j, k]^2); σQ = sp[1, Q...] * (1 - an[1, Q...] * m[d, Q...]^2)
                s += code[1, Q...] > 1 ? T(2area[d] / h[d]) * σP : T(area[d] / h[d]) * T(0.5) * (σP + σQ)
            end
        end
        dg[1, i, j, k] = s == 0 ? one(T) : s
    end
    return dg
end

# The face currents leaving contact k (its cells) into other cells: the current it feeds, A.
function _contactcurrent(phi::Array{T,4}, G::Array{T,4}, m::Array{T,4}, code, sp, an, mesh::Mesh, k) where {T}
    n = mesh.size; h = mesh.cellsize; per = ntuple(d -> isperiodic(mesh, d), 3)
    area = (h[2] * h[3], h[1] * h[3], h[1] * h[2])
    s = 0.0
    @inbounds for kk in 1:n[3], j in 1:n[2], i in 1:n[1]
        code[1, i, j, kk] == 1 + k || continue
        I = (i, j, kk)
        for d in 1:3, sgn in (-1, 1)
            n[d] > 1 || continue
            b = _nb(I[d], n[d], per[d], sgn); b == 0 && continue
            Q = ntuple(e -> e == d ? b : I[e], 3)
            (code[1, Q...] == 0 || code[1, Q...] == 1 + k) && continue
            mP = (m[1, I...], m[2, I...], m[3, I...]); mQ = (m[1, Q...], m[2, Q...], m[3, Q...])
            GP = (G[1, I...], G[2, I...], G[3, I...]); GQ = (G[1, Q...], G[2, Q...], G[3, Q...])
            cP = code[1, I...]; cQ = code[1, Q...]
            F = sgn > 0 ? _faceflux(T, d, cP, cQ, mP, mQ, sp[1, I...], sp[1, Q...], an[1, I...], an[1, Q...], phi[1, I...], phi[1, Q...], GP, GQ, area[d], h[d]) :
                          -_faceflux(T, d, cQ, cP, mQ, mP, sp[1, Q...], sp[1, I...], an[1, Q...], an[1, I...], phi[1, Q...], phi[1, I...], GQ, GP, area[d], h[d])
            s += Float64(F)
        end
    end
    return s
end

# --- generic pieces (broadcasts and reductions: CPU and GPU) ---------------------------------------

# φ in the contact cells: contact k at vals[k], the free cells untouched.
function _setcontacts!(phi, code, vals)
    for (k, v) in enumerate(vals)
        phi .= ifelse.(code .== 1 + k, eltype(phi)(v), phi)
    end
    return phi
end
_freemask(x, code) = (x .= ifelse.(code .== 1, x, zero(eltype(x))); x)
_cdot(a, b) = Float64(sum(Float64.(a) .* Float64.(b)))

# A·x on the free cells (contacts and insulators at 0): the divergence of the currents of x.
function _applyop!(y, x, c::Conduction, m, mesh)
    _cellgradients!(c.G, x, c.code, mesh)
    _divergence!(y, x, c.G, m, c.code, c.sp, c.an, mesh)
    return _freemask(y, c.code)
end

# Solve: φ free so that the divergence vanishes, with the contact cells as they are in φ. BiCGSTAB with
# Jacobi, from φ itself. Returns the iterations.
function _solvephi!(phi, c::Conduction, m, mesh, vmax)
    T = eltype(phi)
    _jacobi!(c.dg, m, c.code, c.sp, c.an, mesh)
    # r = −div(φ) on the free cells
    _cellgradients!(c.G, phi, c.code, mesh)
    _divergence!(c.r, phi, c.G, m, c.code, c.sp, c.an, mesh)
    c.r .= .-c.r; _freemask(c.r, c.code)
    # scale of the residual: the currents that the contact potentials of this solve would drive through
    # one cell, ‖diag‖·max|V|
    ref = sqrt(_cdot(c.dg, c.dg)) * vmax
    copyto!(c.r0, c.r); fill!(c.p, 0); fill!(c.v, 0)
    ρ0 = α = ω = 1.0
    it = 0
    sqrt(_cdot(c.r, c.r)) <= c.tol * ref && return 0
    while it < c.maxiter
        it += 1
        ρ1 = _cdot(c.r0, c.r)
        ρ1 == 0 && error("conduction: BiCGSTAB breakdown")
        β = (ρ1 / ρ0) * (α / ω)
        c.p .= c.r .+ T(β) .* (c.p .- T(ω) .* c.v)
        c.y .= c.p ./ c.dg; _freemask(c.y, c.code)
        _applyop!(c.v, c.y, c, m, mesh)
        α = ρ1 / _cdot(c.r0, c.v)
        c.s .= c.r .- T(α) .* c.v
        if sqrt(_cdot(c.s, c.s)) <= c.tol * ref
            phi .+= T(α) .* c.y
            return it
        end
        c.zz .= c.s ./ c.dg; _freemask(c.zz, c.code)
        _applyop!(c.t, c.zz, c, m, mesh)
        tt = _cdot(c.t, c.t)
        ω = tt == 0 ? 0.0 : _cdot(c.t, c.s) / tt
        phi .+= T(α) .* c.y .+ T(ω) .* c.zz
        c.r .= c.s .- T(ω) .* c.t
        sqrt(_cdot(c.r, c.r)) <= c.tol * ref && return it
        ρ0 = ρ1
    end
    error("conduction: BiCGSTAB did not converge in $(c.maxiter) iterations (relative residual " *
          "$(sqrt(_cdot(c.r, c.r)) / ref))")
end

# --- the solution at a state m ---------------------------------------------------------------------

"""
    solveconduction!(world, m) -> Conduction

The potential, the current density and the contact potentials and currents for the state `m`, at the
contact values of the time last set (see [`setconduction!`](@ref)).
"""
function solveconduction!(w::World, m)
    c = _conduction(w)
    c.ready || _setupcells!(c, w)
    mesh = w.mesh
    nc = length(c.contacts)
    pot = [k for k in 1:nc if c.contacts[k].kind === :potential]
    cur = [k for k in 1:nc if c.contacts[k].kind === :current]
    ground = isempty(pot) ? pop!(cur) : 0                 # without a potential contact, the last current contact is at 0 V
    # φ₀: potential contacts at their values, the others at 0
    vals = [c.contacts[k].kind === :potential ? c.contacts[k].value : 0.0 for k in 1:nc]
    vmax = max(maximum(abs, vals; init = 0.0), 1e-300)
    b0 = c.basis[1]
    _setcontacts!(b0, c.code, vals)
    it = any(!=(0), vals) ? _solvephi!(b0, c, m, mesh, vmax) : (fill!(b0, 0); 0)
    _cellgradients!(c.G, b0, c.code, mesh)
    I0 = [_contactcurrent(b0, c.G, m, c.code, c.sp, c.an, mesh, k) for k in 1:nc]
    # ψ_l: current contact l at 1 V, all others at 0
    M = zeros(length(cur), length(cur)); Iψ = zeros(nc, length(cur))
    for (l, kl) in enumerate(cur)
        ψ = c.basis[1+l]
        _setcontacts!(ψ, c.code, [k == kl ? 1.0 : 0.0 for k in 1:nc])
        it += _solvephi!(ψ, c, m, mesh, 1.0)
        _cellgradients!(c.G, ψ, c.code, mesh)
        for k in 1:nc
            Iψ[k, l] = _contactcurrent(ψ, c.G, m, c.code, c.sp, c.an, mesh, k)
        end
    end
    M .= Iψ[cur, :]
    V = isempty(cur) ? Float64[] : M \ ([c.contacts[k].value for k in cur] .- I0[cur])
    T = eltype(c.phi)
    copyto!(c.phi, b0)
    for (l, kl) in enumerate(cur)
        c.phi .+= T(V[l]) .* c.basis[1+l]
    end
    for k in 1:nc
        q = c.contacts[k]
        q.V = q.kind === :potential ? q.value : (k == ground ? 0.0 : V[findfirst(==(k), cur)])
        q.I = I0[k] + (isempty(cur) ? 0.0 : sum(Iψ[k, l] * V[l] for l in eachindex(cur)))
    end
    _cellgradients!(c.G, c.phi, c.code, mesh)
    _divergence!(c.phi, c.phi, c.G, m, c.code, c.sp, c.an, mesh; curr = c.j)
    c.iterations = it
    return c
end

# --- output and the Simulation ---------------------------------------------------------------------

for f in (:setconduction!, :setresistivity!, :addcontact!)
    @eval function $f(sim::Simulation, args...; kwargs...)
        $f(sim.world, args...; kwargs...)
        sim.hostworld === sim.world || $f(sim.hostworld, args...; kwargs...)
        return sim
    end
end
solveconduction!(sim::Simulation) = (_drive!(sim.hostworld, sim.t); solveconduction!(sim.hostworld, _hoststate(sim)))

"The electric potential of each cell [V] for the present state: (1, Nx, Ny, Nz), for `save!(sim; field = potentialmap)`."
potentialmap(sim::Simulation) = Array(solveconduction!(sim).phi)
"The current density of each cell [A/m²] for the present state, for `save!(sim; field = currentmap)`."
currentmap(sim::Simulation) = Array(solveconduction!(sim).j)

"""
    q_contacts()

Table columns: the potential [V] and the current [A] fed into the sample of every contact
([`addcontact!`](@ref)): `V1, I1, V2, I2, …`.
"""
function q_contacts(n::Integer)
    Quantity("contact", "V,A", 2n, function (w, m, t)
        c = solveconduction!(w, m)
        reduce(vcat, [[q.V, q.I] for q in c.contacts])
    end)
end

"""
    q_resistance(a=1, b=2)

Table column: the resistance (V_a − V_b)/I_a [Ω] between contacts `a` and `b` (in the order they were
added): the two-terminal resistance when the current enters at a and leaves at b.
"""
q_resistance(a::Integer = 1, b::Integer = 2) = Quantity("R", "Ohm", 1, function (w, m, t)
    c = solveconduction!(w, m)
    [(c.contacts[a].V - c.contacts[b].V) / c.contacts[a].I]
end)
