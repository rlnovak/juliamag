# Library of standard materials.
#
# Room-temperature parameters for the ferromagnets most used in micromagnetics,
# with representative literature values (Coey, "Magnetism and Magnetic Materials";
# OOMMF/mumax3 example decks; standard-problem specs). These are convenient
# starting points, NOT authoritative constants — exchange stiffness and damping
# in particular vary with sample quality, thickness, and temperature, so override
# them for quantitative work.
#
# Damping α is a nominal value; anisotropy is uniaxial K1 along +z by default
# (cubic anisotropy is not yet modelled, so cubic materials list their |K1|).

"""
    material(name; alpha=nothing, anisU=(0,0,1)) -> Material

Look up a standard material by name (case-insensitive). Known names:
`Permalloy`/`Py`, `Fe`, `Co`, `Ni`, `CoFeB`, `YIG`. Optionally override the
damping or easy axis.

# Example
```julia
py = material("Permalloy")
co = material("Co"; alpha = 0.05)
```
"""
function material(name::AbstractString; alpha = nothing, anisU = (0, 0, 1))
    key = lowercase(name)
    p = get(_MATERIALS, key, nothing)
    p === nothing && throw(ArgumentError(
        "unknown material \"$name\"; known: " * join(sort(collect(keys(_MATERIALS_CANON))), ", ")))
    a = alpha === nothing ? p.alpha : alpha
    if p.K1 == 0
        return Material(Msat = p.Msat, Aex = p.Aex, alpha = a)
    else
        return Material(Msat = p.Msat, Aex = p.Aex, alpha = a, Ku = p.K1, anisU = anisU)
    end
end

"List the known material names."
materialnames() = sort(collect(values(_MATERIALS_CANON)))

# Parameter records: Msat [A/m], Aex [J/m], K1 [J/m³] (uniaxial; 0 if none or
# cubic-only), α (nominal).
struct _MatRecord
    Msat::Float64
    Aex::Float64
    K1::Float64
    alpha::Float64
end

const _MATERIALS = Dict{String,_MatRecord}(
    # Permalloy (Ni80Fe20): the micromagnetics workhorse, soft, K≈0.
    "permalloy" => _MatRecord(8.0e5,  1.3e-11, 0.0,     0.02),
    "py"        => _MatRecord(8.0e5,  1.3e-11, 0.0,     0.02),
    # Iron (bcc): cubic K1≈4.8e4; listed as magnitude.
    "fe"        => _MatRecord(1.7e6,  2.1e-11, 4.8e4,   0.002),
    "iron"      => _MatRecord(1.7e6,  2.1e-11, 4.8e4,   0.002),
    # Cobalt (hcp): strong uniaxial anisotropy along the c-axis.
    "co"        => _MatRecord(1.4e6,  3.0e-11, 5.2e5,   0.01),
    "cobalt"    => _MatRecord(1.4e6,  3.0e-11, 5.2e5,   0.01),
    # Nickel (fcc): cubic K1<0; listed as magnitude.
    "ni"        => _MatRecord(4.9e5,  9.0e-12, 5.7e3,   0.02),
    "nickel"    => _MatRecord(4.9e5,  9.0e-12, 5.7e3,   0.02),
    # CoFeB: common in spintronics / MTJs; PMA when interfaced.
    "cofeb"     => _MatRecord(1.1e6,  1.9e-11, 0.0,     0.01),
    # YIG (Y3Fe5O12): very low damping, low Msat.
    "yig"       => _MatRecord(1.4e5,  3.6e-12, 0.0,     1.0e-4),
)

# Canonical display name per key (for error messages / listing).
const _MATERIALS_CANON = Dict(
    "permalloy" => "Permalloy", "py" => "Permalloy",
    "fe" => "Fe", "iron" => "Fe",
    "co" => "Co", "cobalt" => "Co",
    "ni" => "Ni", "nickel" => "Ni",
    "cofeb" => "CoFeB", "yig" => "YIG",
)
