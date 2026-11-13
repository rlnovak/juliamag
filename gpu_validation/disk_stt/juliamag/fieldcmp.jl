# Load mumax3's relaxed m into JuliaMag, compute B_demag and B_exch on the same
# geometry, and compare cell by cell with mumax3's saved fields.
# Usage: julia --project=<JuliaMag> fieldcmp.jl DIR
#   DIR holds fields_es0.out/ and fields_es4.out/, the output folders of
#   ../mumax3/fields_es0.mx3 and fields_es4.mx3 (m_relaxed, B_demag, B_exch, geom).
using JuliaMag, Printf
const DIR = ARGS[1]

# scalar OVF (geom) reader: binary 4-byte, 1 component
function loadscalar(fn, nx, ny, nz)
    open(fn) do io
        while true
            l = readline(io); occursin("Begin: Data Binary", l) && break
        end
        read(io, Float32)                     # check value
        a = zeros(nx, ny, nz)
        for k in 1:nz, j in 1:ny, i in 1:nx; a[i,j,k] = read(io, Float32); end
        a
    end
end

mesh = Mesh((64,64,8), (2e-9,2e-9,2e-9))
mat  = Material(Msat=800e3, Aex=10.8e-12, alpha=0.01)
for es in (0, 4)
    d = joinpath(DIR, "fields_es$es.out")
    m, _  = loadovf(joinpath(d, "m_relaxed.ovf"))
    Bd, _ = loadovf(joinpath(d, "B_demag.ovf"))
    Bx, _ = loadovf(joinpath(d, "B_exch.ovf"))
    vol   = loadscalar(joinpath(d, "geom.ovf"), 64, 64, 8)

    rp = RegionParams(mesh, mat); setregion!(rp, 0; Msat = 0.0)
    setgeometry!(rp, Cylinder(128e-9, 32e-9); id = 1, edgesmooth = es)
    w = World(mesh, rp; demag = true)

    @printf("\n===== EdgeSmooth = %d =====\n", es)
    @printf("geometry: max|fill_JM - vol_mumax| = %.3e   (#cells vol>0: mumax %d, JM %d)\n",
            maximum(abs.(rp.fill .* (rp.regions.id .== 1) .- vol)), count(>(0), vol),
            count(i -> JuliaMag.msat(rp, Tuple(i)...) > 0, CartesianIndices(vol)))

    Bdj = zeros(size(m)); demagfield!(Bdj, m, w.demagplan, rp, mesh)
    Bxj = zeros(size(m)); JuliaMag.exchange!(Bxj, m, mesh, rp)

    inside = vol .> 0; edge = (vol .> 0) .& (vol .< 1); bulk = vol .== 1
    for (nm, Bm, Bj) in (("B_demag", Bd, Bdj), ("B_exch", Bx, Bxj))
        dif = dropdims(sqrt.(sum(abs2, Bj .- Bm; dims=1)); dims=1)
        mag = dropdims(sqrt.(sum(abs2, Bm; dims=1)); dims=1)
        scale = maximum(mag[inside])
        @printf("%-8s max|ΔB| inside=%.3e T  bulk=%.3e  edge=%s   (peak|B|=%.3e T, max rel=%.2e)\n",
                nm, maximum(dif[inside]), maximum(dif[bulk]),
                any(edge) ? @sprintf("%.3e", maximum(dif[edge])) : "n/a", scale, maximum(dif[inside])/scale)
        I = argmax(dif .* inside)
        @printf("         worst cell %s vol=%.3f  mumax=%s  JM=%s\n", Tuple(I), vol[I],
                round.(Bm[:,I], sigdigits=5), round.(Bj[:,I], sigdigits=5))
    end
    @printf("JM energies on mumax m: E_demag=%.4e  E_exch=%.4e  E_total=%.4e\n",
            demagenergy(m, w.demagplan, mesh, rp), exchangeenergy(m, mesh, rp), totalenergy(m, w))
end
