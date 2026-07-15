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
mutable struct World{T<:AbstractFloat,P}
    mesh::Mesh
    material::Material{T}
    demagplan::P                     # DemagPlan{T,...} or nothing
    Bext::NTuple{3,T}
end

function World(mesh::Mesh, mat::Material{T}; demag::Bool = true,
              Bext = (0, 0, 0), accuracy = 6.0) where {T}
    plan = demag ? DemagPlan(demagkernel(T, mesh; accuracy = accuracy), mesh, mat) : nothing
    World{T,typeof(plan)}(mesh, mat, plan, NTuple{3,T}(Bext))
end

"Set the uniform applied field [T]."
setexternalfield!(w::World{T}, Bext) where {T} = (w.Bext = NTuple{3,T}(Bext); w)

"""
    effectivefield!(B, m, world)

Assemble the total effective field of state `m` into `B` [T].
"""
function effectivefield!(B::AbstractArray{T,4}, m::AbstractArray{T,4}, w::World{T}) where {T}
    exchange!(B, m, w.mesh, w.material; add = false)      # first term writes
    if w.material.Ku != 0
        anisotropy!(B, m, w.mesh, w.material; add = true)
    end
    if w.demagplan !== nothing
        demagfield!(B, m, w.demagplan; add = true)
    end
    if any(!iszero, w.Bext)
        zeeman!(B, w.Bext; add = true)
    end
    return B
end
