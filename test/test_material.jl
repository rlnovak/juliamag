@testset "Material" begin
    # Permalloy, as used in µMAG standard problem 4.
    py = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)

    @test py.Msat == 8.0e5
    @test py.Aex == 1.3e-11
    @test py.alpha == 0.02
    @test py.Ku == 0.0
    @test eltype(py) === Float64

    # Permalloy's exchange length is ~5.7 nm; the standard problem's 3.125 nm
    # cell sits comfortably below it.
    @test exchangelength(py) ≈ 5.69e-9 rtol = 0.01

    @testset "anisotropy axis is normalized" begin
        mat = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02, Ku = 5.0e5, anisU = (0, 0, 3))
        @test all(mat.anisU .≈ (0.0, 0.0, 1.0))

        mat2 = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02, Ku = 5.0e5, anisU = (1, 1, 0))
        @test sum(abs2, mat2.anisU) ≈ 1.0
    end

    @testset "validation" begin
        @test_throws ArgumentError Material(Msat = 0.0, Aex = 1.3e-11, alpha = 0.02)
        @test_throws ArgumentError Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = -0.1)
        @test_throws ArgumentError Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02,
                                            Ku = 1.0, anisU = (0, 0, 0))
    end

    @testset "Float32 material" begin
        mat = Material(Msat = 8.0f5, Aex = 1.3f-11, alpha = 0.02f0)
        @test eltype(mat) === Float32
    end
end
