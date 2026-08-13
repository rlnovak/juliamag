@testset "Finite temperature (thermal field)" begin
    using Random, Statistics

    mesh = Mesh((40, 40, 20), (4e-9, 4e-9, 4e-9))   # 32000 cells: many samples
    mat  = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.1)
    V    = JuliaMag.cellvolume(mesh)

    # Expected per-component variance of the thermal field.
    s2(temp, dt, α, Ms) = 2 * JuliaMag.kB * temp * α / (JuliaMag.γLL * Ms * V * dt)

    @testset "variance matches the fluctuation-dissipation formula" begin
        rng = MersenneTwister(1)
        T = 300.0; dt = 1e-14
        B = zeros(3, mesh.size...)
        JuliaMag.thermalfield!(B, mesh, mat, T, dt; rng = rng)
        # 96000 samples → sample variance is within ~1% of the truth.
        @test isapprox(var(vec(B)), s2(T, dt, mat.alpha, mat.Msat); rtol = 0.03)
        @test abs(mean(vec(B))) < 0.05 * std(vec(B))     # zero mean
    end

    @testset "T = 0 gives no field" begin
        B = ones(3, mesh.size...)
        JuliaMag.thermalfield!(B, mesh, mat, 0.0, 1e-14)
        @test all(iszero, B)
    end

    @testset "scaling with T, alpha, and dt" begin
        rng() = MersenneTwister(7)
        base = zeros(3, mesh.size...); JuliaMag.thermalfield!(base, mesh, mat, 100.0, 1e-14; rng = rng())
        # Doubling T scales the field by sqrt(2) (same noise draw).
        hot = zeros(3, mesh.size...); JuliaMag.thermalfield!(hot, mesh, mat, 200.0, 1e-14; rng = rng())
        @test isapprox(std(vec(hot)) / std(vec(base)), sqrt(2); rtol = 1e-6)
        # Halving dt scales the field by sqrt(2).
        fast = zeros(3, mesh.size...); JuliaMag.thermalfield!(fast, mesh, mat, 100.0, 0.5e-14; rng = rng())
        @test isapprox(std(vec(fast)) / std(vec(base)), sqrt(2); rtol = 1e-6)
        # Quadrupling alpha scales the field by 2.
        mat4 = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.4)
        aA = zeros(3, mesh.size...); JuliaMag.thermalfield!(aA, mesh, mat4, 100.0, 1e-14; rng = rng())
        @test isapprox(std(vec(aA)) / std(vec(base)), 2.0; rtol = 1e-6)
    end

    @testset "empty (Msat=0) cells get no thermal field" begin
        smesh = Mesh((8, 8, 1), (4e-9, 4e-9, 4e-9))
        rp = RegionParams(smesh, mat)
        setregion!(rp, 0; Msat = 0.0)                 # whole sample empty
        B = ones(3, smesh.size...)
        JuliaMag.thermalfield!(B, smesh, rp, 300.0, 1e-14)
        @test all(iszero, B)
    end

    @testset "runthermal! preserves |m|=1 and advances time" begin
        smesh = Mesh((8, 8, 1), (4e-9, 4e-9, 4e-9))
        sim = Simulation(smesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5); demag = false)
        setmag!(sim, UniformConfig(0, 0, 1))
        JuliaMag.runthermal!(sim, 1e-12, 300.0; dt = 1e-14, rng = MersenneTwister(3))
        norms = [sqrt(sim.m[1,i,j,1]^2 + sim.m[2,i,j,1]^2 + sim.m[3,i,j,1]^2)
                 for i in 1:8, j in 1:8]
        @test all(n -> isapprox(n, 1; atol = 1e-6), norms)
        @test sim.t ≈ 1e-12 rtol = 1e-9
        @test all(isfinite, sim.m)
    end
end
