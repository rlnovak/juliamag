# Landau-Lifshitz-Gilbert torque.
#
# The Gilbert form of the equation of motion,
#
#     dm/dt = -γ m×B + α m×dm/dt,
#
# is solved for dm/dt to give the explicit Landau-Lifshitz form used here:
#
#     dm/dt = -γ' [ m×B + α m×(m×B) ],   γ' = γ / (1+α²)
#
# The first term is precession about B; the second is damping that spirals m
# towards B. γ is the Landau-Lifshitz gyromagnetic ratio [rad/(T·s)]. B is the
# effective field in Tesla, m is the reduced (unit) magnetization.
#
# Written over AbstractArray{T,4}, so a CuArray dispatches to the same code.

"""
    torque!(dm, m, B, alpha; gamma=γLL)

Write the LLG torque `dm/dt` of state `m` under effective field `B` into `dm`.
`dm`, `m`, `B` are `(3,Nx,Ny,Nz)`; `alpha` is the Gilbert damping.
"""
function torque!(dm::AbstractArray{T,4}, m::AbstractArray{T,4}, B::AbstractArray{T,4},
                 alpha::Real; gamma::Real = γLL) where {T}
    γ′ = T(gamma / (1 + alpha^2))
    α = T(alpha)
    @inbounds for k in axes(m, 4), j in axes(m, 3), i in axes(m, 2)
        mx, my, mz = m[1, i, j, k], m[2, i, j, k], m[3, i, j, k]
        bx, by, bz = B[1, i, j, k], B[2, i, j, k], B[3, i, j, k]

        # p = m × B  (precession)
        px = my*bz - mz*by
        py = mz*bx - mx*bz
        pz = mx*by - my*bx

        # q = m × p = m × (m × B)  (damping)
        qx = my*pz - mz*py
        qy = mz*px - mx*pz
        qz = mx*py - my*px

        dm[1, i, j, k] = -γ′ * (px + α*qx)
        dm[2, i, j, k] = -γ′ * (py + α*qy)
        dm[3, i, j, k] = -γ′ * (pz + α*qz)
    end
    return dm
end

"""
    maxtorque(dm) -> T

The largest torque magnitude over all cells, `max_cell |dm/dt|`. Used as the
convergence criterion for relaxation (in rad/s).
"""
function maxtorque(dm::AbstractArray{T,4}) where {T}
    mx = zero(T)
    @inbounds for k in axes(dm, 4), j in axes(dm, 3), i in axes(dm, 2)
        t2 = dm[1, i, j, k]^2 + dm[2, i, j, k]^2 + dm[3, i, j, k]^2
        t2 > mx && (mx = t2)
    end
    return sqrt(mx)
end
