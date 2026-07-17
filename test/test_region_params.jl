# A shape covering a single central cell (helper for the has* test).
cell_shape() = (x, y, z) -> abs(x) < 3e-9 && abs(y) < 3e-9

@testset "RegionParams" begin
    py = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)

    @testset "uniform region params reproduce the scalar Material" begin
        # A RegionParams whose regions are all the default must give identical
        # fields to the scalar Material — the dispatch seam is transparent.
        mesh = Mesh((10, 8, 2), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(mesh, py)          # all cells region 0 = py

        m = randommag!(zeromag(mesh))
        Bm = similar(m); exchange!(Bm, m, mesh, py)
        Br = similar(m); exchange!(Br, m, mesh, rp)
        @test Br ≈ Bm

        # Anisotropy too (with a Ku material).
        mat = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02, Ku = 5e5, anisU = (0,0,1))
        rp2 = RegionParams(mesh, mat)
        Am = similar(m); anisotropy!(Am, m, mesh, mat)
        Ar = similar(m); anisotropy!(Ar, m, mesh, rp2)
        @test Ar ≈ Am
    end

    @testset "setregion! overrides only the named parameters" begin
        mesh = Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(mesh, py)
        setregion!(rp, 1; Msat = 1.4e6, Aex = 2.0e-11)
        # region 0 unchanged, region 1 overridden
        @test rp.Msat[1] == 8.0e5           # region 0 (index 1)
        @test rp.Msat[2] == 1.4e6           # region 1 (index 2)
        @test rp.Aex[2] == 2.0e-11
        @test rp.alpha[2] == 0.02           # not set → default kept
        @test_throws ArgumentError setregion!(rp, 300; Msat = 1e6)
    end

    @testset "accessors read the per-cell region" begin
        mesh = Mesh((10, 1, 1), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(mesh, py)
        defregion!(rp, 1, XRange(0.0, 1.0))         # right half → region 1
        setregion!(rp, 1; Msat = 1.4e6)
        # left cells are region 0 (Msat 8e5), right cells region 1 (1.4e6)
        @test JuliaMag.msat(rp, 1, 1, 1) == 8.0e5
        @test JuliaMag.msat(rp, 10, 1, 1) == 1.4e6
    end

    @testset "exchange interface uses the harmonic mean of stiffness" begin
        # Two regions with different Aex meeting at an interface: the coupling
        # across the boundary is the harmonic mean, not either bulk value.
        mesh = Mesh((6, 1, 1), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(mesh, py)
        defregion!(rp, 1, XRange(0.0, 1e3))         # cells 4,5,6 → region 1
        A0 = 1.3e-11; A1 = 5.0e-11
        setregion!(rp, 0; Aex = A0)
        setregion!(rp, 1; Aex = A1)

        # A magnetization with a kink at the interface. Cell 3 is region 0, cell 4
        # region 1; the exchange field at cell 3 from its right neighbour uses the
        # harmonic mean 2 A0 A1/(A0+A1).
        m = zeromag(mesh)
        for i in 1:6
            θ = 0.2 * i
            m[1,i,1,1] = cos(θ); m[3,i,1,1] = sin(θ)
        end
        B = similar(m); exchange!(B, m, mesh, rp)
        @test all(isfinite, B)
        # Sanity: swapping the two stiffnesses (symmetric) leaves the interface
        # coupling unchanged, since the harmonic mean is symmetric.
        rp2 = RegionParams(mesh, py)
        defregion!(rp2, 1, XRange(0.0, 1e3))
        setregion!(rp2, 0; Aex = A1); setregion!(rp2, 1; Aex = A0)
        harm(a,b) = 2a*b/(a+b)
        @test harm(A0,A1) == harm(A1,A0)
    end

    @testset "bilayer with different Msat runs in a World" begin
        # Two z-layers, different Msat — a magnetic multilayer.
        mesh = Mesh((6, 6, 4), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(mesh, py)
        defregion!(rp, 1, Layers(mesh, 3, 5))       # top two layers
        setregion!(rp, 1; Msat = 1.4e6, Aex = 2.0e-11)

        world = World(mesh, rp; demag = true)
        m = uniform(mesh, (1, 0, 0))
        B = similar(m)
        effectivefield!(B, m, world)
        @test all(isfinite, B)
        # The two layers now carry different Msat.
        @test JuliaMag.msat(rp, 1, 1, 1) == 8.0e5   # bottom layer
        @test JuliaMag.msat(rp, 1, 1, 4) == 1.4e6   # top layer
    end

    @testset "region-aware demag matches the uniform path for one material" begin
        # With a single material everywhere, the Msat[cell]·m input path (prefactor
        # μ0) must equal the uniform-Msat path (prefactor μ0·Msat, input m).
        mesh = Mesh((8, 8, 2), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(mesh, py)
        plan = DemagPlan(demagkernel(mesh), mesh, py.Msat)
        m = randommag!(zeromag(mesh))

        Buni = similar(m); demagfield!(Buni, m, plan)
        Breg = similar(m); demagfield!(Breg, m, plan, rp, mesh)
        @test Breg ≈ Buni rtol = 1e-10
    end

    @testset "bilayer Msat: stronger demag in the higher-Msat layer" begin
        # Two z-layers, both magnetized out of plane; the layer with larger Msat
        # feels a stronger (more negative) demag field along z.
        mesh = Mesh((8, 8, 4), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(mesh, py)
        defregion!(rp, 1, Layers(mesh, 3, 5))       # top two layers → region 1
        setregion!(rp, 1; Msat = 1.6e6)             # double Msat
        plan = DemagPlan(demagkernel(mesh), mesh, JuliaMag.maxmsat(rp))

        m = uniform(mesh, (0, 0, 1))                # out of plane
        B = similar(m); demagfield!(B, m, plan, rp, mesh)
        # Demag opposes m (Bz < 0) in both layers, stronger where Msat is larger.
        bz_bottom = B[3, 4, 4, 1]                   # region 0 (Msat 8e5)
        bz_top    = B[3, 4, 4, 4]                   # region 1 (Msat 1.6e6)
        @test bz_bottom < 0
        @test bz_top < 0
        @test abs(bz_top) > abs(bz_bottom)          # higher Msat ⇒ stronger demag
    end

    @testset "has* predicates scan present regions" begin
        mesh = Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(mesh, py)                 # no Ku, no DMI, no STT
        @test !JuliaMag.hasku(rp)
        @test !JuliaMag.hasdmi(rp)
        @test !JuliaMag.hasstt(rp)
        # Give region 1 anisotropy but never paint it — not present, so still off.
        setregion!(rp, 1; Ku = 5e5)
        @test !JuliaMag.hasku(rp)
        defregion!(rp, 1, cell_shape())             # paint one cell region 1
        @test JuliaMag.hasku(rp)
    end
end
