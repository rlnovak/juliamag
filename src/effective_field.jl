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
    World(mesh, material; demag=true, Bext=(0,0,0))

A complete micromagnetic problem: geometry, material, and which field terms are
active. The demag plan is built once here and reused for the whole run.

- `demag`: include the demagnetization field (the expensive term).
- `Bext`: uniform applied field [T]; can be changed between run segments.
"""
mutable struct World{T<:AbstractFloat,P,M<:AbstractParams,B<:AbstractArray{T,4}}
    mesh::Mesh
    material::M                       # Material{T} or RegionParams{T}
    demagplan::P                      # DemagPlan{T,...} or nothing
    Bext::NTuple{3,T}
    _Bbuf::B                          # scratch effective-field buffer for integrators
    current::Union{Nothing,SpinCurrent{T}}   # spin-polarized current (STT), or nothing
end

function World(mesh::Mesh, mat::AbstractParams; demag::Bool = true,
              Bext = (0, 0, 0), accuracy = 6.0)
    T = eltype(mat)
    # The demag kernel is purely geometric (independent of Msat); the plan's
    # μ0·Msat scaling uses a representative Msat here. For a scalar Material that
    # is exact; region-dependent Msat in the demag is handled in stage 5.
    Msref = _demag_msat(mat)
    # The kernel and its transform are computed in Float64 and only the transformed
    # kernel is rounded to T (as mumax3 does for its float32 kernel).
    plan = demag ? DemagPlan(demagkernel(Float64, mesh; accuracy = accuracy), mesh, Msref, T) : nothing
    Bbuf = zeros(T, 3, mesh.size...)
    World{T,typeof(plan),typeof(mat),typeof(Bbuf)}(mesh, mat, plan, NTuple{3,T}(Bext), Bbuf, nothing)
end

"""
    setcurrent!(world, J; zhangli=true, slonczewski=false, fixedlayer=(0,0,1),
                thickness=<mesh thickness>, params=sttparams(world.material))
    setcurrent!(world, nothing)

Set (or remove) the spin-polarized current density `J` [A/m²] whose spin-transfer
torques every integrator adds to the LLG torque; see [`SpinCurrent`](@ref).
"""
function setcurrent!(w::World{T}, J; zhangli::Bool = true, slonczewski::Bool = false,
                     fixedlayer = (0, 0, 1), thickness = nothing, params = nothing) where {T}
    th = thickness === nothing ? w.mesh.size[3] * w.mesh.cellsize[3] : thickness
    pm = params === nothing ? sttparams(w.material) : params
    w.current = SpinCurrent(J, Material{T}(pm); zhangli = zhangli, slonczewski = slonczewski,
                            fixedlayer = fixedlayer, thickness = th)
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
    exchange!(B, m, w.mesh, w.material; add = false)      # first term writes
    if hasku(w.material)
        anisotropy!(B, m, w.mesh, w.material; add = true)
    end
    if hasdmi(w.material)
        dmi!(B, m, w.mesh, w.material; add = true)
    end
    if w.demagplan !== nothing
        _demag!(B, m, w; add = true)
    end
    if any(!iszero, w.Bext)
        zeeman!(B, w.Bext; add = true)
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
