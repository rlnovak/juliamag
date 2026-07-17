# Material parameters. Uniform over the whole sample for now — regions with
# different materials come later.

"""
    Material(; Msat, Aex, alpha, Ku=0.0, anisU=(0,0,1), Dind=0.0, Dbulk=0.0)

Magnetic parameters of a single, uniform material.

- `Msat`: saturation magnetization [A/m].
- `Aex`: exchange stiffness [J/m].
- `alpha`: Gilbert damping (dimensionless).
- `Ku`: uniaxial anisotropy constant [J/m³]. Zero disables anisotropy.
- `anisU`: uniaxial anisotropy easy axis; normalized on construction.
- `Dind`: interfacial (Néel) DMI constant [J/m²]. Zero disables it. Stabilizes
  Néel skyrmions and walls in thin films with broken inversion symmetry.
- `Dbulk`: bulk (Bloch) DMI constant [J/m³]. Zero disables it. Stabilizes Bloch
  skyrmions and helices in chiral B20 crystals.

# Example (Permalloy, µMAG standard problem 4)
```julia
mat = Material(Msat=8.0e5, Aex=1.3e-11, alpha=0.02)
```
"""
# Material parameters come in two flavours that the field routines treat
# uniformly through the accessor functions at the bottom of this file: a scalar
# `Material` (one material everywhere) and a per-region `RegionParams` (regions.jl
# territory). Both are `AbstractParams`; multiple dispatch selects the accessor,
# so a single-material simulation pays no lookup cost while a multi-region one
# indexes a table — and the field code is written once against the accessors.
abstract type AbstractParams end

struct Material{T<:AbstractFloat} <: AbstractParams
    Msat::T
    Aex::T
    alpha::T
    Ku::T
    anisU::NTuple{3,T}
    Dind::T
    Dbulk::T
    # Spin-transfer-torque parameters (0 disables the respective term).
    pol::T          # current spin polarization (both Zhang-Li and Slonczewski)
    xi::T           # Zhang-Li non-adiabaticity (β)
    lambda::T       # Slonczewski Slonczewski asymmetry parameter Λ
    epsilonPrime::T # Slonczewski secondary (field-like) torque ε'

    function Material{T}(Msat, Aex, alpha, Ku, anisU, Dind, Dbulk,
                         pol, xi, lambda, epsilonPrime) where {T<:AbstractFloat}
        Msat > 0 || throw(ArgumentError("Msat must be > 0, got $Msat"))
        alpha >= 0 || throw(ArgumentError("alpha must be ≥ 0, got $alpha"))
        u = NTuple{3,T}(anisU)
        n = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
        n > 0 || throw(ArgumentError("anisU must not be the zero vector"))
        new{T}(T(Msat), T(Aex), T(alpha), T(Ku), u ./ n, T(Dind), T(Dbulk),
               T(pol), T(xi), T(lambda), T(epsilonPrime))
    end
end

function Material(; Msat, Aex, alpha, Ku = nothing, anisU = (0, 0, 1),
                  Dind = nothing, Dbulk = nothing,
                  pol = nothing, xi = nothing, lambda = 1.0, epsilonPrime = nothing)
    # Optional constants default to nothing rather than 0.0 so that leaving one
    # out cannot drag an otherwise-Float32 material up to Float64.
    opt(x) = x === nothing ? Bool : typeof(float(x))
    T = promote_type(typeof(float(Msat)), typeof(float(Aex)), typeof(float(alpha)),
                     opt(Ku), opt(Dind), opt(Dbulk), opt(pol), opt(xi), opt(epsilonPrime))
    Material{T}(Msat, Aex, alpha, something(Ku, zero(T)), anisU,
                something(Dind, zero(T)), something(Dbulk, zero(T)),
                something(pol, zero(T)), something(xi, zero(T)),
                T(lambda), something(epsilonPrime, zero(T)))
end

Base.eltype(::Material{T}) where {T} = T

"Exchange length √(2A / (μ0 Msat²)) [m] — the mesh cell size should stay below this."
exchangelength(mat::Material) = sqrt(2 * mat.Aex / (μ0 * mat.Msat^2))

function Base.show(io::IO, ::MIME"text/plain", mat::Material)
    print(io, "Material\n")
    print(io, "  Msat  = ", mat.Msat, " A/m\n")
    print(io, "  Aex   = ", mat.Aex, " J/m\n")
    print(io, "  alpha = ", mat.alpha, "\n")
    if mat.Ku != 0
        print(io, "  Ku    = ", mat.Ku, " J/m³ along ", mat.anisU, "\n")
    end
    if mat.Dind != 0
        print(io, "  Dind  = ", mat.Dind, " J/m² (interfacial)\n")
    end
    if mat.Dbulk != 0
        print(io, "  Dbulk = ", mat.Dbulk, " J/m³ (bulk)\n")
    end
    print(io, "  exchange length = ", exchangelength(mat) * 1e9, " nm")
end

# --- Parameter accessors ---------------------------------------------------
# The field routines read material parameters through these accessors rather
# than reaching into a struct field, so the same code serves a scalar Material
# and a per-region RegionParams (regions.jl). For a Material the cell indices are
# ignored and the scalar is returned; the compiler inlines this to a plain field
# read, so single-material simulations pay nothing. The RegionParams methods
# (defined alongside that type) index a per-region lookup table instead.

@inline msat(m::Material, i, j, k)   = m.Msat
@inline aex(m::Material, i, j, k)    = m.Aex
@inline alphaof(m::Material, i, j, k) = m.alpha
@inline ku(m::Material, i, j, k)     = m.Ku
@inline anisu(m::Material, i, j, k)  = m.anisU
@inline dind(m::Material, i, j, k)   = m.Dind
@inline dbulk(m::Material, i, j, k)  = m.Dbulk
@inline polof(m::Material, i, j, k)  = m.pol
@inline xiof(m::Material, i, j, k)   = m.xi
@inline lambdaof(m::Material, i, j, k) = m.lambda
@inline epsprime(m::Material, i, j, k) = m.epsilonPrime

# Whether any cell has a nonzero value — lets a field skip work when a term is
# globally off. For a scalar Material this is just the field; RegionParams scans
# its table.
hasku(m::Material)    = m.Ku != 0
hasdmi(m::Material)   = m.Dind != 0 || m.Dbulk != 0
hasstt(m::Material)   = m.pol != 0

# A representative scalar damping for the LLG torque, which currently takes α as
# a single value. For a scalar Material this is exact; per-region α in the torque
# is a later refinement.
damping(m::Material) = m.alpha

# A single-material sample is all one region (0); region 0's average is the whole
# sample, any other region is empty.
average_region(m::AbstractArray, ::Material, id::Integer) =
    id == 0 ? average(m) : (eltype(m)(NaN), eltype(m)(NaN), eltype(m)(NaN))
