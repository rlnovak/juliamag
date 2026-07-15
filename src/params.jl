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
struct Material{T<:AbstractFloat}
    Msat::T
    Aex::T
    alpha::T
    Ku::T
    anisU::NTuple{3,T}
    Dind::T
    Dbulk::T

    function Material{T}(Msat, Aex, alpha, Ku, anisU, Dind, Dbulk) where {T<:AbstractFloat}
        Msat > 0 || throw(ArgumentError("Msat must be > 0, got $Msat"))
        alpha >= 0 || throw(ArgumentError("alpha must be ≥ 0, got $alpha"))
        u = NTuple{3,T}(anisU)
        n = sqrt(u[1]^2 + u[2]^2 + u[3]^2)
        n > 0 || throw(ArgumentError("anisU must not be the zero vector"))
        new{T}(T(Msat), T(Aex), T(alpha), T(Ku), u ./ n, T(Dind), T(Dbulk))
    end
end

function Material(; Msat, Aex, alpha, Ku = nothing, anisU = (0, 0, 1),
                  Dind = nothing, Dbulk = nothing)
    # Optional constants default to nothing rather than 0.0 so that leaving one
    # out cannot drag an otherwise-Float32 material up to Float64.
    opt(x) = x === nothing ? Bool : typeof(float(x))
    T = promote_type(typeof(float(Msat)), typeof(float(Aex)), typeof(float(alpha)),
                     opt(Ku), opt(Dind), opt(Dbulk))
    Material{T}(Msat, Aex, alpha, something(Ku, zero(T)), anisU,
                something(Dind, zero(T)), something(Dbulk, zero(T)))
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
