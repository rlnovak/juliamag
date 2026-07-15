@testset "Demag kernel" begin
    # The self-cell (zero offset) sits at padded index (1,1,1). The trace of the
    # self-demag tensor is exactly 1 for any single cell (sum rule Nxx+Nyy+Nzz=1),
    # and for a cube each diagonal term is 1/3 by symmetry.
    @testset "cube: self-demag factors are 1/3" begin
        mesh = Mesh((4, 4, 4), (5e-9, 5e-9, 5e-9))
        K = demagkernel(mesh)
        nxx = -K.Kxx[1, 1, 1]      # N = -K here (see convention note below)
        nyy = -K.Kyy[1, 1, 1]
        nzz = -K.Kzz[1, 1, 1]
        @test nxx ≈ 1/3 rtol = 1e-3
        @test nyy ≈ 1/3 rtol = 1e-3
        @test nzz ≈ 1/3 rtol = 1e-3
        @test nxx + nyy + nzz ≈ 1.0 rtol = 1e-3
    end

    @testset "sum rule holds for a non-cubic cell" begin
        mesh = Mesh((4, 4, 4), (4e-9, 6e-9, 10e-9))
        K = demagkernel(mesh)
        tr = -(K.Kxx[1, 1, 1] + K.Kyy[1, 1, 1] + K.Kzz[1, 1, 1])
        @test tr ≈ 1.0 rtol = 1e-3
        # The cell is longest along z, so it demagnetizes *least* along z
        # (a bar magnetized along its long axis has the smallest demag factor)
        # and most along its shortest axis, x.
        @test -K.Kzz[1, 1, 1] < -K.Kyy[1, 1, 1] < -K.Kxx[1, 1, 1]
    end

    @testset "thin cell: Nzz is the largest demag factor" begin
        # A flat cell (thin along z) demagnetizes most strongly along z.
        # The aspect ratio is kept moderate: the brute-force integrator's point
        # count scales with (long edge / short edge)³, so a 50:1 cell would take
        # hundreds of millions of sub-points per cell on the CPU.
        mesh = Mesh((2, 2, 2), (12e-9, 12e-9, 3e-9))
        K = demagkernel(mesh)
        nxx, nyy, nzz = -K.Kxx[1,1,1], -K.Kyy[1,1,1], -K.Kzz[1,1,1]
        @test nzz > nxx
        @test nzz > nyy
        @test nxx + nyy + nzz ≈ 1.0 rtol = 1e-3
    end

    @testset "off-diagonal self term is zero" begin
        mesh = Mesh((4, 4, 4), (5e-9, 5e-9, 5e-9))
        K = demagkernel(mesh)
        @test abs(K.Kxy[1, 1, 1]) < 1e-9
        @test abs(K.Kxz[1, 1, 1]) < 1e-9
        @test abs(K.Kyz[1, 1, 1]) < 1e-9
    end

    @testset "2D mesh: xz and yz components vanish" begin
        mesh = Mesh((8, 8, 1), (5e-9, 5e-9, 5e-9))
        K = demagkernel(mesh)
        @test all(iszero, K.Kxz)
        @test all(iszero, K.Kyz)
        @test size(K.Kxx) == padsize(mesh)
    end

    @testset "diagonal parity under reflection" begin
        # Nxx is even in every offset; the reconstructed half must mirror the
        # computed half. Check a reflected pair along x.
        mesh = Mesh((6, 6, 2), (5e-9, 5e-9, 5e-9))
        K = demagkernel(mesh)
        px = padsize(mesh)[1]
        for x in 2:(px ÷ 2)
            x2 = px - x + 2                       # reflection partner (1-based)
            @test K.Kxx[x, 2, 1] ≈ K.Kxx[x2, 2, 1] rtol = 1e-9
            @test K.Kxy[x, 2, 1] ≈ -K.Kxy[x2, 2, 1] rtol = 1e-9   # xy is odd in x
        end
    end

    @testset "element type propagates" begin
        mesh = Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9))
        @test eltype(demagkernel(Float32, mesh).Kxx) === Float32
        @test eltype(demagkernel(mesh).Kxx) === Float64
    end
end
