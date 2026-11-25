# energy!, linearfield!, isconservative, and the property the energy-path (GNEB) solver
# relies on: the force V·Msat·B_eff is exactly −∂E/∂m. Every term is quadratic in m
# (Zeeman linear), so a central difference E(m+εδ) − E(m−εδ) = 2ε ∇E·δ holds to
# rounding for any ε, and a non-symmetric discrete operator shows up as an O(1e-2)
# mismatch. Two discretizations are non-symmetric (exchange and DMI on edge-smoothed
# cells, a DMI constant that varies between cells); isconservative must flag exactly
# those. DMI at free boundaries is symmetric, chiral or open.
using JuliaMag, Test, Random

const _base = (Msat = 8e5, Aex = 1.3e-11, alpha = 0.02)

function _gradcases()
    cases = Tuple{String,Any,Bool}[]
    m1 = Mesh((8, 6, 3), (3e-9, 3e-9, 2e-9))
    push!(cases, ("open 3D, Ku + demag + Bext",
        World(m1, Material(; _base..., Ku = 3e5, anisU = (0.3, 0.2, 1.0)); demag = true,
              Bext = (0.02, -0.01, 0.03)), true))
    m2 = Mesh((8, 8, 2), (3e-9, 3e-9, 3e-9); pbc = (2, 2, 0))
    push!(cases, ("periodic xy film, Dind + demag",
        World(m2, Material(; _base..., Dind = 2.5e-3); demag = true, Bext = (0, 0, 0.05)), true))
    m3 = Mesh((6, 6, 4), (3e-9, 3e-9, 3e-9); pbc = (2, 2, 2))
    push!(cases, ("periodic 3D, Dbulk + Ku",
        World(m3, Material(; _base..., Dbulk = 1.5e-3, Ku = 1e5, anisU = (0, 0, 1)); demag = false), true))
    m4 = Mesh((12, 12, 2), (4e-9, 4e-9, 4e-9))
    rp = RegionParams(m4, Material(; _base...))
    setregion!(rp, 0; Msat = 0.0)
    setregion!(rp, 1; Msat = 8e5, Aex = 1.3e-11)
    setregion!(rp, 2; Msat = 1.4e6, Aex = 2.0e-11, Ku = 5e5, anisU = (0, 0, 1))
    setgeometry!(rp, Cylinder(44e-9, 1.0); id = 1)
    setgeometry!(rp, Cylinder(20e-9, 1.0); id = 2)
    push!(cases, ("regions: disc + core, empty background, demag",
        World(m4, rp; demag = true, Bext = (0.01, 0.0, 0.0)), true))
    # DMI at free boundaries
    for obc in (false, true)
        push!(cases, ("open 3D, Dind, openbc $obc",
            World(m1, Material(; _base..., Dind = 2.5e-3); demag = false, openbc = obc), true))
        push!(cases, ("open 3D, Dbulk, openbc $obc",
            World(m1, Material(; _base..., Dbulk = 1.5e-3); demag = true, openbc = obc), true))
        push!(cases, ("xy-periodic film, Dbulk (open in z), openbc $obc",
            World(m2, Material(; _base..., Dbulk = 1.5e-3); demag = false, openbc = obc), true))
        rd = RegionParams(m4, Material(; _base...))          # empty background without D
        setregion!(rd, 0; Msat = 0.0)
        setregion!(rd, 1; Msat = 8e5, Aex = 1.3e-11, Dind = 2.5e-3)
        setregion!(rd, 2; Msat = 1.4e6, Aex = 2.0e-11, Dind = 2.5e-3)
        setgeometry!(rd, Cylinder(44e-9, 1.0); id = 1)
        setgeometry!(rd, Cylinder(20e-9, 1.0); id = 2)
        push!(cases, ("disc + core, same Dind, empty background, openbc $obc",
            World(m4, rd; demag = false, openbc = obc), true))
    end
    # non-conservative discretizations
    r2 = RegionParams(m2, Material(; _base..., Dind = 2.5e-3))
    defregion!(r2, 1, (x, y, z) -> x > 0)
    setregion!(r2, 1; Msat = 8e5, Aex = 1.3e-11, Dind = 1.0e-3)
    push!(cases, ("periodic xy film, region-dependent Dind",
        World(m2, r2; demag = false), false))
    rs = RegionParams(m4, Material(; _base...))
    setregion!(rs, 0; Msat = 0.0); setregion!(rs, 1; Msat = 8e5, Aex = 1.3e-11)
    setgeometry!(rs, Cylinder(44e-9, 1.0); id = 1, edgesmooth = 4)
    push!(cases, ("edge-smoothed disc (exchange on partial cells)",
        World(m4, rs; demag = true), false))
    return cases
end

function _graderr(w, rng)
    mesh = w.mesh
    m = randommag!(rng, zeromag(Float64, mesh)); B = similar(m)
    energy!(B, m, w)
    δ = randn(rng, size(m)...)
    g = 0.0
    for k in 1:mesh.size[3], j in 1:mesh.size[2], i in 1:mesh.size[1]
        g += JuliaMag.msat(w.material, i, j, k) * (B[1,i,j,k]*δ[1,i,j,k] +
             B[2,i,j,k]*δ[2,i,j,k] + B[3,i,j,k]*δ[3,i,j,k])
    end
    g *= -cellvolume(mesh)
    rel = 0.0
    for ε in (1e-3, 1e-1)
        fd = (energy!(similar(m), m .+ ε .* δ, w) - energy!(similar(m), m .- ε .* δ, w)) / 2ε
        rel = max(rel, abs(fd - g) / abs(g))
    end
    return rel
end

@testset "energy!, linearfield!, exact gradient" begin
    rng = MersenneTwister(11)
    for (label, w, cons) in _gradcases()
        @testset "$label" begin
            mesh = w.mesh
            m = randommag!(rng, zeromag(Float64, mesh))
            B = similar(m)
            @test energy!(B, m, w) ≈ totalenergy(m, w) rtol = 1e-12
            B2 = similar(m); effectivefield!(B2, m, w)
            @test B == B2                                    # energy! leaves B_eff in B
            d1 = randn(rng, size(m)...); d2 = randn(rng, size(m)...)
            L1 = similar(m); L2 = similar(m); L12 = similar(m)
            linearfield!(L1, d1, w); linearfield!(L2, d2, w)
            linearfield!(L12, 0.7 .* d1 .- 1.3 .* d2, w)
            @test maximum(abs, L12 .- (0.7 .* L1 .- 1.3 .* L2)) <= 1e-12 * maximum(abs, L12)
            Lm = similar(m); linearfield!(Lm, m, w)
            @test maximum(abs, B .- (Lm .+ reshape(collect(w.Bext), 3, 1, 1, 1))) <=
                  1e-12 * maximum(abs, B)
            @test isconservative(w) == cons
            rel = _graderr(w, rng)
            if cons
                @test rel < 1e-9                               # exact gradient
            else
                @test rel > 1e-4                               # the flag is not vacuous
            end
        end
    end
end
