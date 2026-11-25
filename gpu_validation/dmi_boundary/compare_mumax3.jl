# Exchange + DMI and anisotropy, field and energy, of JuliaMag against mumax3, on the
# random states written by the scripts of make_mx3.py.
#
#   julia --project=. gpu_validation/dmi_boundary/compare_mumax3.jl DIR
#   (DIR: NAME.params of make_mx3.py and the NAME.out of mumax3)
#
# Prints, per case and field, the largest difference over the largest field, of the edge
# cells, the cells at a region interface, the inner cells and the partial cells, and the
# relative difference of the energy. mumax3 is single precision: agreement is to ~1e-6.
using JuliaMag, Printf

const DIR = ARGS[1]
const EMPTY = 100                                             # region id of the empty cells

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

# Column `name` of table.txt (header, then one row)
function tablecol(dir, name)
    l = filter(!isempty, readlines(joinpath(dir, "table.txt")))
    cols = split(lstrip(first(l), ['#', ' ']), '\t')
    c = findfirst(x -> startswith(x, name * " "), cols)
    c === nothing && error("$dir: no column $name in $cols")
    return parse(Float64, split(last(l), '\t')[c])
end

# NAME.params: "pbc px py pz", "openbc 0/1", "region key value..."
function readparams(file)
    pbc = (0, 0, 0); obc = false; reg = Dict{Int,Dict{Symbol,Any}}()
    for l in eachline(file)
        w = split(l)
        if w[1] == "pbc"
            pbc = Tuple(parse.(Int, w[2:4]))
        elseif w[1] == "openbc"
            obc = w[2] == "1"
        else
            v = parse.(Float64, w[3:end])
            get!(Dict{Symbol,Any}, reg, parse(Int, w[1]))[Symbol(w[2])] = length(v) == 1 ? v[1] : Tuple(v)
        end
    end
    return pbc, obc, reg
end

function params(mesh, geom, regs, reg)
    all(==(1), geom) && length(reg) == 1 && return Material(; reg[0]...)
    rp = RegionParams(mesh, Material(; reg[0]...))
    for (r, kw) in reg
        setregion!(rp, r; kw...)
    end
    setregion!(rp, EMPTY; Msat = 0.0)
    for I in CartesianIndices(geom)
        id = geom[I] == 0 ? EMPTY : Int(regs[I])
        id == 0 || defregioncell!(rp.regions, id, I[1], I[2], I[3])
        rp.fill[I] = geom[I] == 0 ? 1.0 : geom[I]
    end
    return rp
end

function run(name)
    dir = joinpath(DIR, name * ".out")
    load(q) = Float64.(loadovf(joinpath(dir, q * ".ovf"))[1])
    m = load("m"); n = size(m)[2:4]
    geom = scalarovf(joinpath(dir, "geom.ovf"), n)
    regs = round.(Int, scalarovf(joinpath(dir, "regions.ovf"), n))
    pbc, obc, reg = readparams(joinpath(DIR, name * ".params"))
    mesh = Mesh(n, (3e-9, 4e-9, 2e-9); pbc = pbc)
    p = params(mesh, geom, regs, reg)
    isedge = falses(n); isface = falses(n)                    # missing neighbour; other region
    for I in CartesianIndices(n), d in 1:3, s in (-1, 1)
        J = collect(Tuple(I)); J[d] += s
        out = !(1 <= J[d] <= n[d])
        out && pbc[d] != 0 && (J[d] = mod1(J[d], n[d]); out = false)
        isedge[I] |= out || geom[J...] == 0
        isface[I] |= !out && geom[J...] != 0 && regs[J...] != regs[I]
    end
    full = geom .== 1; partial = 0 .< geom .< 1
    function report(what, B, Bm, E, Em)
        Δ = dropdims(maximum(abs, B .- Bm; dims = 1); dims = 1) ./ maximum(abs, Bm)
        worst(mask) = any(mask) ? @sprintf("%.1e", maximum(Δ[mask])) : "   -   "
        @printf("%-16s %-4s edge %s  interface %s  inner %s  partial %s   E JuliaMag %+.7e mumax3 %+.7e rel %+.1e\n",
                name, what, worst(isedge .& full), worst(isface .& full),
                worst(.!isedge .& .!isface .& full), worst(partial), E, Em, E / Em - 1)
    end
    Bx = similar(m); exchange!(Bx, m, mesh, p)
    Bd = similar(m); dmi!(Bd, m, mesh, p; openbc = obc)
    E = exchangeenergy(m, mesh, p) + dmienergy(m, mesh, p; openbc = obc)
    report("exch", Bx .+ Bd, load("B_exch"), E, tablecol(dir, "E_exch"))
    if startswith(name, "ani")
        Ba = similar(m); anisotropy!(Ba, m, mesh, p)
        report("anis", Ba, load("B_anis"), anisotropyenergy(m, mesh, p), tablecol(dir, "E_anis"))
    end
end

for f in sort(filter(endswith(".out"), readdir(DIR)))
    name = f[1:end-4]
    isfile(joinpath(DIR, name * ".params")) ? run(name) : @warn "no $name.params: skipped"
end
