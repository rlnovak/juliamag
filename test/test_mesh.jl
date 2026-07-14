@testset "Mesh" begin
    mesh = Mesh((160, 40, 1), (3.125e-9, 3.125e-9, 3e-9))

    @test size(mesh) == (160, 40, 1)
    @test length(mesh) == 160 * 40 * 1
    @test cellvolume(mesh) ≈ 3.125e-9 * 3.125e-9 * 3e-9
    @test volume(mesh) ≈ length(mesh) * cellvolume(mesh)
    @test all(worldsize(mesh) .≈ (500e-9, 125e-9, 3e-9))
    @test !isperiodic(mesh, 1)

    @testset "constructor validation" begin
        @test_throws ArgumentError Mesh((0, 4, 1), (1e-9, 1e-9, 1e-9))
        @test_throws ArgumentError Mesh((4, 4, 1), (0.0, 1e-9, 1e-9))
        @test_throws ArgumentError Mesh((4, 4, 1), (1e-9, 1e-9, 1e-9); pbc = (-1, 0, 0))
    end

    @testset "constructor does not mutate its arguments" begin
        cs = [5e-9, 5e-9, 2e-9]
        Mesh((8, 8, 1), cs)
        @test cs == [5e-9, 5e-9, 2e-9]
    end

    @testset "padsize" begin
        # Large N: pad to 2N. Small N (≤ 5): pad to 2N-1.
        @test padsize(Mesh((160, 40, 1), (1e-9, 1e-9, 1e-9))) == (320, 80, 1)
        @test padsize(Mesh((8, 8, 4), (1e-9, 1e-9, 1e-9))) == (16, 16, 7)
        @test padsize(Mesh((3, 8, 2), (1e-9, 1e-9, 1e-9))) == (5, 16, 3)

        # A periodic axis wraps around, so it needs no padding at all.
        @test padsize(Mesh((8, 8, 4), (1e-9, 1e-9, 1e-9); pbc = (1, 0, 0))) == (8, 16, 7)
        @test padsize(Mesh((8, 8, 4), (1e-9, 1e-9, 1e-9); pbc = (2, 3, 0))) == (8, 8, 7)
    end

    @testset "padsize is always ≥ 2N-1 for a linear convolution" begin
        for n in 1:12
            m = Mesh((n, n, n), (1e-9, 1e-9, 1e-9))
            @test all(padsize(m) .>= 2 .* size(m) .- 1)
        end
    end

    @testset "dataregion fits inside the padded array" begin
        mesh = Mesh((8, 8, 4), (1e-9, 1e-9, 1e-9))
        r = dataregion(mesh)
        @test length.(r) == size(mesh)
        @test all(last.(r) .<= padsize(mesh))
    end
end
