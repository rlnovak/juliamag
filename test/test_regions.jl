@testset "Regions" begin
    @testset "default region 0 fills the mesh" begin
        mesh = Mesh((8, 8, 2), (5e-9, 5e-9, 5e-9))
        r = Regions(mesh)
        @test size(r) == (8, 8, 2)
        @test all(r.id .== 0)
        @test regionlist(r) == [0]
        @test regionvolume(r, 0) == 8 * 8 * 2
    end

    @testset "defregion! paints cells inside a cuboid" begin
        mesh = Mesh((10, 10, 1), (5e-9, 5e-9, 5e-9))   # 50 × 50 nm
        r = Regions(mesh)
        # A 20 × 20 nm cuboid at the centre covers the central 4 × 4 cells.
        defregion!(r, 1, Cuboid(20e-9, 20e-9, 1e6))
        @test regionvolume(r, 1) == 16          # 4 × 4 central cells
        @test regionvolume(r, 0) == 100 - 16
        @test sort(regionlist(r)) == [0, 1]
        # centre cells are region 1, corners are region 0
        @test r[5, 5, 1] == 1
        @test r[1, 1, 1] == 0
    end

    @testset "cylinder region" begin
        mesh = Mesh((20, 20, 1), (5e-9, 5e-9, 5e-9))   # 100 × 100 nm
        r = Regions(mesh)
        defregion!(r, 2, Cylinder(80e-9, 1e6))         # 80 nm diameter disc
        n = regionvolume(r, 2)
        # A disc of radius 40 nm on a 100 nm square: area ratio ≈ π·40²/100² ≈ 0.50,
        # so roughly half of the 400 cells.
        @test 170 < n < 230
        @test r[10, 10, 1] == 2                 # centre is inside
        @test r[1, 1, 1] == 0                   # corner is outside the disc
    end

    @testset "later paints overwrite earlier (build outward)" begin
        mesh = Mesh((10, 10, 1), (4e-9, 4e-9, 4e-9))
        r = Regions(mesh)
        defregion!(r, 1, Universe())            # everything region 1
        defregion!(r, 2, Cuboid(16e-9, 16e-9, 1e6))  # centre → region 2
        @test regionvolume(r, 1) + regionvolume(r, 2) == 100
        @test r[5, 5, 1] == 2
        @test r[1, 1, 1] == 1
    end

    @testset "multilayer: different region per z-layer" begin
        mesh = Mesh((4, 4, 4), (5e-9, 5e-9, 5e-9))     # 4 layers
        r = Regions(mesh)
        defregion!(r, 1, Layers(mesh, 1, 3))    # bottom two layers
        defregion!(r, 2, Layers(mesh, 3, 5))    # top two layers
        @test regionvolume(r, 1) == 4 * 4 * 2
        @test regionvolume(r, 2) == 4 * 4 * 2
        @test r[1, 1, 1] == 1                   # bottom layer
        @test r[1, 1, 4] == 2                   # top layer
        @test sort(regionlist(r)) == [1, 2]
    end

    @testset "defregioncell! sets a single cell" begin
        mesh = Mesh((6, 6, 1), (5e-9, 5e-9, 5e-9))
        r = Regions(mesh)
        defregioncell!(r, 3, 2, 4, 1)
        @test r[2, 4, 1] == 3
        @test regionvolume(r, 3) == 1
    end

    @testset "validation" begin
        r = Regions(Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9)))
        @test_throws ArgumentError defregion!(r, 256, Universe())
        @test_throws ArgumentError defregion!(r, -1, Universe())
    end

    @testset "cellcenter matches the shape/config convention" begin
        mesh = Mesh((11, 1, 1), (5e-9, 5e-9, 5e-9))    # odd → centre cell at 6
        @test cellcenter(mesh, 6, 1, 1)[1] ≈ 0.0       # centre cell at origin
        @test cellcenter(mesh, 1, 1, 1)[1] ≈ -5 * 5e-9 # leftmost cell
        @test cellcenter(mesh, 11, 1, 1)[1] ≈ 5 * 5e-9 # rightmost cell
    end
end
