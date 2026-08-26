@testset "Voronoi grains" begin
    using Statistics

    mesh = Mesh((80, 80, 2), (4e-9, 4e-9, 4e-9))     # 320×320 nm, 2 layers
    py   = material("Permalloy")
    gs   = 40e-9

    @testset "tessellation is reproducible and seed-dependent" begin
        a = RegionParams(mesh, py); voronoi!(a, gs, 100; seed = 1)
        b = RegionParams(mesh, py); voronoi!(b, gs, 100; seed = 1)
        c = RegionParams(mesh, py); voronoi!(c, gs, 100; seed = 2)
        @test a.regions.id == b.regions.id          # same seed → identical
        @test a.regions.id != c.regions.id          # different seed → different
    end

    @testset "grains are columnar (constant through z)" begin
        rp = RegionParams(mesh, py); voronoi!(rp, gs, 100; seed = 5)
        @test rp.regions.id[:, :, 1] == rp.regions.id[:, :, 2]
    end

    @testset "region ids stay within range" begin
        rp = RegionParams(mesh, py); voronoi!(rp, gs, 16; seed = 7)
        @test all(r -> 0 <= r < 16, rp.regions.id)
    end

    @testset "more grains with a smaller grain size" begin
        # Boundary-cell density grows as the grains shrink.
        function boundary_frac(grainsize)
            rp = RegionParams(mesh, py); voronoi!(rp, grainsize, 200; seed = 3)
            id = rp.regions.id[:, :, 1]
            b = 0
            for j in 2:79, i in 2:79
                r = id[i, j]
                (id[i-1,j] != r || id[i+1,j] != r || id[i,j-1] != r || id[i,j+1] != r) && (b += 1)
            end
            b / (78 * 78)
        end
        @test boundary_frac(20e-9) > boundary_frac(60e-9)
    end

    @testset "random anisotropy: uniform axes, normalized" begin
        rp = RegionParams(mesh, py)
        randomanisotropy!(rp, 200; Ku = 5e5, seed = 11)
        axes = [rp.anisU[r+1] for r in 0:199]
        @test all(a -> isapprox(sqrt(a[1]^2 + a[2]^2 + a[3]^2), 1; atol = 1e-9), axes)
        zs = [a[3] for a in axes]
        @test abs(mean(zs)) < 0.1                    # z-component averages ~0
        @test isapprox(std(zs), 1/sqrt(3); atol = 0.1)   # uniform-on-sphere spread
        @test all(r -> rp.Ku[r+1] == 5e5, 0:199)
    end

    @testset "a polycrystal builds and its field is finite" begin
        rp = RegionParams(mesh, py)
        voronoi!(rp, gs, 64; seed = 2)
        randomanisotropy!(rp, 64; Ku = 5e5, seed = 2)
        w = World(mesh, rp; demag = true)
        m = uniform(mesh, (1, 0, 0))
        B = similar(m); effectivefield!(B, m, w)
        @test all(isfinite, B)
    end
end
