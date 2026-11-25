# Exchange + DMI field and energy of JuliaMag against mumax3 at free boundaries, on the
# random states written by the scripts of make_mx3.py.
#
#   julia --project=. gpu_validation/dmi_boundary/compare_mumax3.jl DIR      (DIR/*.out)
#
# Prints, per case, the largest difference of the field over the largest field, of the
# edge cells and of the inner cells, and the relative difference of the energy. mumax3 is
# single precision: agreement is to ~1e-6.
using JuliaMag, Printf

const DIR = ARGS[1]
const D = Dict("ind" => 2.5e-3, "bulk" => 1.5e-3)

# A scalar mumax3 file (geom, regions; OVF2 binary 4, valuedim 1). loadovf reads vectors.
function scalarovf(file, dims)
    raw = read(file)
    tag = Vector{UInt8}("# Begin: Data Binary 4\n")
    r = findfirst(tag, raw)
    r === nothing && error("$file: not OVF2 binary 4")
    occursin("valuedim: 1", String(raw[1:first(r)])) || error("$file: not a scalar")
    v = reinterpret(Float32, raw[last(r)+1:last(r)+4*(prod(dims)+1)])
    v[1] == 1234567.0f0 || error("$file: bad check value")
    return reshape(Float64.(v[2:end]), dims)
end

# table.txt: header, then one row; the energy is the last column
tableE(dir) = parse(Float64, last(split(last(filter(!isempty, readlines(joinpath(dir, "table.txt")))))))

function params(name, mesh, geom, regs)
    kind = startswith(name, "ind") ? "ind" : "bulk"
    dkw = kind == "ind" ? (Dind = D[kind],) : (Dbulk = D[kind],)
    base = (Msat = 8e5, Aex = 1.3e-11, alpha = 0.02)
    all(==(1), geom) && all(==(0), regs) && return Material(; base..., dkw...)
    rp = RegionParams(mesh, Material(; base..., dkw...))
    setregion!(rp, 2; Msat = 0.0)                             # empty cells (no D, no A)
    if occursin("reg", name)
        name == "ind_regA" ? setregion!(rp, 1; Msat = 1.2e6, Aex = 2e-11, dkw...) :
                             setregion!(rp, 1; Msat = 8e5, Aex = 1.3e-11, Dind = 1e-3)
    end
    for I in CartesianIndices(geom)
        id = geom[I] == 0 ? 2 : Int(regs[I])
        id == 0 || defregioncell!(rp.regions, id, I[1], I[2], I[3])
        rp.fill[I] = geom[I] == 0 ? 1.0 : geom[I]
    end
    return rp
end

function run(name)
    dir = joinpath(DIR, name * ".out")
    m = loadovf(joinpath(dir, "m.ovf"))[1]
    Bm = Float64.(loadovf(joinpath(dir, "B_exch.ovf"))[1])
    geom = scalarovf(joinpath(dir, "geom.ovf"), size(m)[2:4])
    regs = round.(Int, scalarovf(joinpath(dir, "regions.ovf"), size(m)[2:4]))
    m = Float64.(m)
    pbc = occursin("pbcxy", name) ? (1, 1, 0) : occursin("pbcyz", name) ? (0, 1, 1) :
          occursin("pbcx", name) ? (1, 0, 0) : (0, 0, 0)
    mesh = Mesh(size(m)[2:4], (3e-9, 4e-9, 2e-9); pbc = pbc)
    p = params(name, mesh, geom, regs)
    obc = endswith(name, "openbc")
    B = similar(m)
    exchange!(B, m, mesh, p); Bx = copy(B)
    dmi!(B, m, mesh, p; add = true, openbc = obc)
    E = exchangeenergy(m, mesh, p) + dmienergy(m, mesh, p; openbc = obc)
    Em = tableE(dir)
    n = size(geom)
    isedge = falses(n)                                        # a missing neighbour on x, y or z
    for I in CartesianIndices(n), d in 1:3, s in (-1, 1)
        J = Tuple(I) .+ ntuple(k -> k == d ? s : 0, 3)
        isedge[I] |= pbc[d] == 0 && !(1 <= J[d] <= n[d]) ||
                     (checkbounds(Bool, geom, J...) && geom[J...] == 0)
    end
    partial = 0 .< geom .< 1
    Δ = dropdims(maximum(abs, B .- Bm; dims = 1); dims = 1) ./ maximum(abs, Bm)
    worst(mask) = any(mask) ? maximum(Δ[mask]) : 0.0
    sz = sum(abs, Bx .- Bm) / sum(abs, Bm)                    # how much the DMI matters
    @printf("%-16s edge %.1e  inner %.1e  partial cells %.1e   E_exch JuliaMag %+.7e mumax3 %+.7e rel %+.1e   (DMI share %.2f)\n",
            name, worst(isedge .& .!partial .& (geom .> 0)), worst(.!isedge .& .!partial .& (geom .> 0)),
            worst(partial), E, Em, E / Em - 1, sz)
end

foreach(f -> run(f[1:end-4]), sort(filter(endswith(".out"), readdir(DIR))))
