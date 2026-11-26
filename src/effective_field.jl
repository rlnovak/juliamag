# Effective field — the sum of every energy term's field.
#
# The LLG torque is driven by the total effective field
#
#     B_eff = B_exch + B_anis + B_demag + B_ext
#
# Each term writes into the same (3,Nx,Ny,Nz) buffer; the first writes, the rest
# accumulate. Which terms are present depends on the simulation, so they are
# bundled into a World that carries the mesh, material, demag plan, and applied
# field together.

"""
    World(mesh, material; demag=true, Bext=(0,0,0), openbc=false)

A complete micromagnetic problem: geometry, material, and which field terms are
active. The demag plan is built once here and reused for the whole run.

- `demag`: include the demagnetization field (the expensive term).
- `Bext`: uniform applied field [T]; can be changed between run segments.
- `openbc`: DMI at free boundaries: `false`, the chiral boundary condition (as mumax3);
  `true`, a missing neighbour contributes nothing (mumax3 `OpenBC`). See [`dmi!`](@ref).
  Set it at construction: changed after `togpu(sim)`, it would not reach the host World
  of the energy columns.
"""
mutable struct World{T<:AbstractFloat,P,M<:AbstractParams,B<:AbstractArray{T,4}}
    mesh::Mesh
    material::M                       # Material{T} or RegionParams{T}
    demagplan::P                      # DemagPlan{T,...} or nothing
    Bext::NTuple{3,T}
    _Bbuf::B                          # scratch effective-field buffer for integrators
    current::Union{Nothing,SpinCurrent{T}}   # spin-polarized current (STT), or nothing
    openbc::Bool                      # DMI boundaries: open (true) or chiral (false)
    drive::Any                        # (world, t) -> sets Bext / current at time t, or nothing
    Bterms::Vector{MapTerm{B}}        # applied field: maps × functions of time (excitation.jl)
    Jterms::Vector{MapTerm{B}}        # current density: the same
    Jbuf::Union{Nothing,B}            # the current density per cell, when there are Jterms
    conduction::Any                   # electric conduction with AMR (conduction.jl), or nothing
end

function World(mesh::Mesh, mat::AbstractParams; demag::Bool = true,
              Bext = (0, 0, 0), accuracy = 6.0, openbc::Bool = false)
    T = eltype(mat)
    # The demag kernel is purely geometric (independent of Msat); the plan's
    # μ0·Msat scaling uses a representative Msat here. For a scalar Material that
    # is exact; region-dependent Msat in the demag is handled in stage 5.
    Msref = _demag_msat(mat)
    # The kernel and its transform are computed in Float64 and only the transformed
    # kernel is rounded to T (as mumax3 does for its float32 kernel).
    plan = demag ? DemagPlan(demagkernel(Float64, mesh; accuracy = accuracy), mesh, Msref, T) : nothing
    Bbuf = zeros(T, 3, mesh.size...)
    World{T,typeof(plan),typeof(mat),typeof(Bbuf)}(mesh, mat, plan, NTuple{3,T}(Bext), Bbuf, nothing, openbc, nothing,
                                                    MapTerm{typeof(Bbuf)}[], MapTerm{typeof(Bbuf)}[], nothing, nothing)
end

"""
    setdrive!(world, f)   /   setdrive!(sim, f)

Time-dependent applied field and current: `f(world, t)` is called with the time of every
evaluation of the equation of motion (each stage of the solvers of [`TimeStepper`](@ref), as
mumax3 does) and before every output, and sets `world.Bext` ([`setexternalfield!`](@ref)) or the
current ([`setcurrent!`](@ref)). `nothing` removes it. The older integrators (`:tsit5`, `Integrator`,
`Solver`, `runthermal!`, `runcurrent!`) refuse it; `Minimizer`, `GNEB` and `eigenmodes` use the field
and the current as last set.
"""
setdrive!(w::World, f) = (w.drive = f; w)

# Whether something of the world depends on time.
_hastime(w::World) = w.drive !== nothing || any(_timedependent, w.Bterms) || any(_timedependent, w.Jterms) ||
    (w.conduction !== nothing && any(q -> !(q.f isa Real), w.conduction.contacts))
_nodrive(w::World, name) = _hastime(w) &&
    throw(ArgumentError("$name does not support a field or current that depends on time (setdrive!, " *
                        "addfield!, addcurrent! with a function); use run! or TimeStepper"))
# Set everything that depends on time to time t.
function _drive!(w::World, t)
    w.drive === nothing || w.drive(w, Float64(t))
    for q in w.Bterms
        q.c = _factor(q.f, t)
    end
    for q in w.Jterms
        q.c = _factor(q.f, t)
    end
    if w.conduction !== nothing
        for q in w.conduction.contacts
            q.value = _factor(q.f, t)
        end
    end
    return w
end

"""
    addfield!(world, map, f=1; t=0)   /   addfield!(sim, map, f=1)

Add the term `map × f(t)` to the applied field (mumax3 `B_ext.Add(mask, f)`): `map` is a
`(3, Nx, Ny, Nz)` array in tesla ([`fieldmap`](@ref), [`regionmap`](@ref)), `f` a function of time or
a number. The function is evaluated at every stage of the solvers, as the uniform part with
[`setdrive!`](@ref). Every part of the code that uses the applied field sees the terms: the effective
field, the Zeeman and total energies, the minimizer, GNEB and the eigenmodes (at the field of the present
time). [`clearfields!`](@ref) removes them.
"""
function addfield!(w::World{T}, map, f = 1.0; t = 0.0) where {T}
    q = MapTerm(_devicemap(w._Bbuf, map), f, _factor(f, t))
    push!(w.Bterms, q)
    return w
end
"Remove the terms added with [`addfield!`](@ref)."
clearfields!(w::World) = (empty!(w.Bterms); w)

"""
    addcurrent!(world, map, f=1; t=0)   /   addcurrent!(sim, map, f=1)

Add the term `map × f(t)` [A/m²] to the current density of the spin torques (mumax3 `J.Add(mask, f)`).
The current itself (its torques, fixed layer, parameters) is set first with
[`setcurrent!`](@ref), whose uniform `J` may be `(0, 0, 0)`. The Zhang-Li torque then takes the
current of each cell, as mumax3; the Slonczewski torque its z component. [`clearcurrents!`](@ref)
removes them.
"""
function addcurrent!(w::World{T}, map, f = 1.0; t = 0.0) where {T}
    w.current === nothing && throw(ArgumentError("addcurrent!: set the current and its torques first, with setcurrent! (J may be (0, 0, 0))"))
    q = MapTerm(_devicemap(w._Bbuf, map), f, _factor(f, t))
    push!(w.Jterms, q)
    w.Jbuf === nothing && (w.Jbuf = similar(w._Bbuf))
    return w
end
"Remove the terms added with [`addcurrent!`](@ref)."
clearcurrents!(w::World) = (empty!(w.Jterms); w)

# B += the terms of the applied field
function _addterms!(B::AbstractArray{T,4}, terms) where {T}
    for q in terms
        q.c == 0 || (B .+= T(q.c) .* q.map)
    end
    return B
end

"""
    appliedfield!(B, world) -> B

The applied field of each cell [T]: the uniform part and the terms of [`addfield!`](@ref), at the
time last set.
"""
appliedfield!(B::AbstractArray{T,4}, w::World{T}) where {T} = _addterms!(zeeman!(B, w.Bext), w.Bterms)

# The current density of each cell into w.Jbuf: J, the terms, and the current of the conduction for the
# state m.
function _currentdensity!(w::World{T}, m) where {T}
    J = w.current.J
    zeeman!(w.Jbuf, J)                         # (writes a uniform vector)
    _addterms!(w.Jbuf, w.Jterms)
    w.conduction === nothing || (w.Jbuf .+= solveconduction!(w, m).j)
    return w.Jbuf
end

"""
    setcurrent!(world, J; zhangli=true, slonczewski=false, fixedlayer=(0,0,1),
                thickness=<mesh thickness>, params=sttparams(world.material))
    setcurrent!(world, nothing)

Set (or remove) the spin-polarized current density `J` [A/m²] whose spin-transfer
torques every integrator adds to the LLG torque; see [`SpinCurrent`](@ref). Without `params`, each cell
takes the parameters of its region (Msat, pol, xi, lambda, epsilonPrime, alpha), as mumax3.
"""
function setcurrent!(w::World{T}, J; zhangli::Bool = true, slonczewski::Bool = false,
                     fixedlayer = (0, 0, 1), thickness = nothing, params = nothing) where {T}
    th = thickness === nothing ? w.mesh.size[3] * w.mesh.cellsize[3] : thickness
    pm = params === nothing ? sttparams(w.material) : params
    w.current = SpinCurrent(J, Material{T}(pm); zhangli = zhangli, slonczewski = slonczewski,
                            fixedlayer = fixedlayer, thickness = th, fromworld = params === nothing)
    return w
end
setcurrent!(w::World, ::Nothing) = (w.current = nothing; w)

# Representative Msat for building the demag plan's μ0·Msat prefactor.
_demag_msat(m::Material) = m.Msat
_demag_msat(rp::RegionParams) = maxmsat(rp)

"Set the uniform applied field [T]."
setexternalfield!(w::World{T}, Bext) where {T} = (w.Bext = NTuple{3,T}(Bext); w)

"""
    effectivefield!(B, m, world)

Assemble the total effective field of state `m` into `B` [T].
"""
function effectivefield!(B::AbstractArray{T,4}, m::AbstractArray{T,4}, w::World{T}) where {T}
    linearfield!(B, m, w)
    if any(!iszero, w.Bext)
        zeeman!(B, w.Bext; add = true)
    end
    _addterms!(B, w.Bterms)
    return B
end

"""
    linearfield!(B, δm, world)

The field of the self-interaction terms only (exchange, anisotropy, DMI, demag:
every term but Zeeman) for `δm`, into `B` [T]. Each of these fields is linear in its
argument, so this is the linear operator B_lin with B_eff(m) = B_lin(m) + B_ext;
`δm` need not be a unit vector field (eigenmodes, energy Hessian).
"""
function linearfield!(B::AbstractArray{T,4}, m::AbstractArray{T,4}, w::World{T}) where {T}
    exchange!(B, m, w.mesh, w.material; add = false)      # first term writes
    if hasku(w.material)
        anisotropy!(B, m, w.mesh, w.material; add = true)
    end
    if hasdmi(w.material)
        dmi!(B, m, w.mesh, w.material; add = true, openbc = w.openbc)
    end
    if w.demagplan !== nothing
        _demag!(B, m, w; add = true)
    end
    return B
end

# Demag dispatch on the material type: a scalar Material uses the fast uniform-
# Msat path (prefactor μ0·Msat, input m); a RegionParams uses the region-aware
# path (input Msat[cell]·m, prefactor μ0), which is exact for per-region Msat.
_demag!(B, m, w::World{T,P,M}; add) where {T,P,M<:Material,} =
    demagfield!(B, m, w.demagplan; add = add)
_demag!(B, m, w::World{T,P,M}; add) where {T,P,M<:RegionParams} =
    demagfield!(B, m, w.demagplan, w.material, w.mesh; add = add)
