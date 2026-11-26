# Spin-polarized current as part of the problem definition.
#
# A SpinCurrent stored in the World makes the spin-transfer torques part of the
# right-hand side every integrator sees (llgrhs!), the way mumax3 adds them in
# SetTorque (engine/torque.go: LL torque + AddSTTorque). One difference from
# mumax3 is deliberate: mumax3 adds the Zhang-Li torque whenever a current density
# is set unless DisableZhangLiTorque is enabled, and the Slonczewski torque whenever
# a fixed layer is set as well; here each torque is switched on explicitly, so a
# current-perpendicular simulation does not pick up a Zhang-Li term by default.
#
# The STT parameters (pol, xi, lambda, epsilonPrime, alpha, Msat) are uniform: they
# are taken from one Material. For a RegionParams world the default is the first
# magnetic region (the lowest region id with Msat > 0); empty cells (m = 0) receive
# no spin torque either way.

"""
    SpinCurrent(J; zhangli=true, slonczewski=false, fixedlayer=(0,0,1),
                thickness, params)

A uniform charge-current density `J = (Jx, Jy, Jz)` [A/m²] and the spin torques it
exerts:

- `zhangli`: the Zhang-Li (current-in-plane) torque, with `params.pol` and
  `params.xi`;
- `slonczewski`: the Slonczewski (spin-valve) torque from the `Jz` component, with
  the polarization `fixedlayer` of the fixed layer, the free-layer `thickness` [m],
  and `params.pol`, `params.lambda`, `params.epsilonPrime`.

`params` is the `Material` the torques take their parameters from.
"""
struct SpinCurrent{T<:AbstractFloat}
    J::NTuple{3,T}
    zhangli::Bool
    slonczewski::Bool
    fixedlayer::NTuple{3,T}
    thickness::T
    params::Material{T}
    fromworld::Bool          # the parameters are those of the world's material (per cell when they differ)
end

function SpinCurrent(J, params::Material{T}; zhangli::Bool = true, slonczewski::Bool = false,
                     fixedlayer = (0, 0, 1), thickness::Real, fromworld::Bool = false) where {T}
    zhangli || slonczewski ||
        throw(ArgumentError("SpinCurrent: enable at least one of zhangli, slonczewski"))
    params.pol != 0 ||
        throw(ArgumentError("SpinCurrent: the spin polarization `pol` of the STT parameters is 0"))
    thickness > 0 || throw(ArgumentError("SpinCurrent: thickness must be > 0"))
    p = NTuple{3,T}(fixedlayer)
    slonczewski && all(iszero, p) &&
        throw(ArgumentError("SpinCurrent: the fixed-layer polarization must not be zero"))
    SpinCurrent{T}(NTuple{3,T}(J), zhangli, slonczewski, p, T(thickness), params, fromworld)
end

"""
    sttparams(material) -> Material

The `Material` the spin torques take their parameters from: the material itself,
or for a `RegionParams` the first magnetic region (lowest id with Msat > 0).
"""
sttparams(m::Material) = m

# Whether the parameters of the spin torques (Msat, pol, xi, lambda, epsilonPrime, alpha) are the same
# in all magnetic regions.
_sttuniform(::Material) = true
function _sttuniform(rp::RegionParams)
    tabs = (rp.Msat, rp.pol, rp.xi, rp.lambda, rp.epsilonPrime, rp.alpha)
    all(t -> all(==(t[1]), t), tabs) && return true
    ref = nothing
    for r in _present(rp)
        rp.Msat[r] == 0 && continue
        v = map(t -> t[r], tabs)
        ref === nothing ? (ref = v) : (ref == v || return false)
    end
    return true
end
sttparams(p::AbstractParams) =
    throw(ArgumentError("sttparams: pass the STT parameters explicitly (params = a Material) " *
                        "for a $(nameof(typeof(p))); a Simulation takes them from its host World"))
function sttparams(rp::RegionParams{T}) where {T}
    for r in sort!(collect(_present(rp)))          # 1-based region slots in use
        rp.Msat[r] > 0 || continue
        return Material{T}(rp.Msat[r], rp.Aex[r], rp.alpha[r], rp.Ku[r], rp.anisU[r],
                           rp.Dind[r], rp.Dbulk[r], rp.pol[r], rp.xi[r], rp.lambda[r],
                           rp.epsilonPrime[r])
    end
    throw(ArgumentError("sttparams: no region with Msat > 0"))
end

"""
    applycurrent!(dm, m, world)

Add the spin-transfer torques of `world.current` (if any) to the torque `dm`.
"""
function applycurrent!(dm::AbstractArray{T,4}, m::AbstractArray{T,4}, w) where {T}
    c = w.current
    c === nothing && return dm
    if isempty(w.Jterms) && w.conduction === nothing && !(c.fromworld && !_sttuniform(w.material))
        # one current density and one set of parameters
        uniformdamping(w.material) === nothing &&
            error("spin torques with explicit `params` need a uniform damping alpha; it differs between the magnetic regions")
        c.slonczewski && slonczewskitorque!(dm, m, w.mesh, c.params, c.J[3], c.fixedlayer,
                                            c.thickness; add = true)
        c.zhangli && zhanglitorque!(dm, m, w.mesh, c.params, c.J; add = true)
    else
        # the current density and the parameters of each cell, as mumax3 (zhangli2.cu, slonczewski2.cu)
        w.Jbuf === nothing && (w.Jbuf = similar(w._Bbuf))
        J = _currentdensity!(w, m)
        par = c.fromworld ? w.material : c.params
        c.slonczewski && slonczewskitorque!(dm, m, w.mesh, par, J, c.fixedlayer, c.thickness; add = true)
        c.zhangli && zhanglitorque!(dm, m, w.mesh, par, J; add = true)
    end
    return dm
end
