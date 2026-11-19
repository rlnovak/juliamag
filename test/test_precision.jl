@testset "Float32 precision" begin
    py = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02, Ku = 2e4, anisU = (1, 0, 0.3),
                  Dind = 1e-4)

    @testset "conversions" begin
        p32 = Material{Float32}(py)
        @test eltype(p32) === Float32
        @test p32.Msat == 8.0f5 && p32.Dind == 1f-4
        @test eltype(material("Permalloy"; precision = Float32)) === Float32
        @test eltype(material("Permalloy")) === Float64
        mesh = Mesh((8, 8, 2), (3e-9, 3e-9, 3e-9))
        rp = RegionParams(mesh, py); defregion!(rp, 1, Layers(mesh, 2, 3))
        setregion!(rp, 1; Msat = 1e6)
        r32 = RegionParams{Float32}(rp)
        @test eltype(r32) === Float32 && eltype(r32.fill) === Float32
        @test r32.Msat[2] == 1f6 && r32.regions === rp.regions
    end

    # Whole effective field and energies in Float32 vs Float64 on a disc with an
    # edge-smoothed rim and an empty background (exchange, anisotropy, DMI, demag,
    # Zeeman): agreement to single-precision rounding.
    mesh = Mesh((24, 24, 3), (2e-9, 2e-9, 2e-9))
    function disc(mat)
        rp = RegionParams(mesh, mat)
        setregion!(rp, 0; Msat = 0.0)
        setgeometry!(rp, Cylinder(40e-9, 6e-9); id = 1, edgesmooth = 4)
        rp
    end
    rp64 = disc(py); rp32 = RegionParams{Float32}(rp64)
    m64 = randommag!(zeromag(mesh)); clearempty!(m64, rp64); m32 = Float32.(m64)
    w64 = World(mesh, rp64; Bext = (0.01, 0, 0.02)); w32 = World(mesh, rp32; Bext = (0.01, 0, 0.02))

    @testset "effective field and energies" begin
        B64 = similar(m64); effectivefield!(B64, m64, w64)
        B32 = similar(m32); effectivefield!(B32, m32, w32)
        @test eltype(B32) === Float32
        @test maximum(abs.(B32 .- B64)) < 1e-5 * maximum(abs.(B64))
        E64 = totalenergy(m64, w64); E32 = totalenergy(m32, w32)
        @test E32 isa Float32
        @test abs(E32 - E64) < 1e-5 * abs(E64)
        a64 = average_magnet(m64, rp64); a32 = average_magnet(m32, rp32)
        @test all(abs.(a32 .- a64) .< 1e-6)
    end

    @testset "Simulation relax + run in Float32" begin
        s64 = Simulation(mesh, rp64); setmag!(s64, VortexConfig(mesh))
        s32 = Simulation(mesh, rp32); setmag!(s32, VortexConfig(mesh))
        @test eltype(s32.m) === Float32
        # The small disc relaxes to a near-uniform state whose soft mode (rotation
        # of m_y) converges slowly: stop tightly enough that both precisions settle.
        relax!(s64; stopdm = 1e-6); relax!(s32; stopdm = 1e-6)
        @test all(abs.(average(s32) .- average(s64)) .< 1e-4)
        E64 = totalenergy(s64.m, s64.world); E32 = totalenergy(s32.m, s32.world)
        @test abs(E32 - E64) < 2e-5 * abs(E64)
        savequantities!(s64, q_time(), q_m(); every = 5e-12)
        savequantities!(s32, q_time(), q_m(); every = 5e-12)
        setexternalfield!(s64.world, (0.02, 0, 0)); setexternalfield!(s32.world, (0.02, 0, 0))
        run!(s64, 2e-11); run!(s32, 2e-11)
        @test all(isfinite, s32.m)
        @test all(abs.(average(s32) .- average(s64)) .< 1e-3)
    end
end
