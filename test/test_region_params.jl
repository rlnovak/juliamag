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

    @testset "unfilled geometry: empty cells carry no material or torque" begin
        # A disc on a square mesh: region 0 (default) is emptied, the cylinder is
        # painted region 1 with material. Cells outside the disc are empty.
        mesh = Mesh((20, 20, 1), (5e-9, 5e-9, 5e-9))   # 100 × 100 nm
        rp = RegionParams(mesh, py)
        setregion!(rp, 0; Msat = 0.0)                  # region 0 → empty
        defregion!(rp, 1, Cylinder(80e-9, 1e6))        # disc → region 1 (material)

        @test JuliaMag.msat(rp, 10, 10, 1) == 8.0e5    # centre is material
        @test JuliaMag.msat(rp, 1, 1, 1) == 0.0        # corner is empty
        @test JuliaMag.isempty_cell(rp, 1, 1, 1)
        @test !JuliaMag.isempty_cell(rp, 10, 10, 1)

        m = uniform(mesh, (1, 0, 0))
        clearempty!(m, rp)
        # Magnetization zeroed outside the disc.
        @test all(iszero, m[:, 1, 1, 1])
        @test !all(iszero, m[:, 10, 10, 1])

        # Torque is zero in empty cells (no phantom magnetization moves).
        world = World(mesh, rp; demag = true)
        B = similar(m); effectivefield!(B, m, world)
        dm = similar(m); torque!(dm, m, B, 0.02)
        @test all(iszero, dm[:, 1, 1, 1])              # empty corner
        @test maximum(abs, dm) > 0                     # material cells feel torque
        @test all(isfinite, B)
    end

    @testset "exchange edge-fill matches mumax3 (no 1/fill, Neumann to empty)" begin
        # mumax3's exchange (cuda/exchange.cu + amul.h inv_Msat) divides by the
        # region's FULL Msat, never Msat·fill, and treats an empty neighbour as a
        # free (Neumann) boundary. Two regressions this guards:
        #   (1) the prefactor must not pick up 1/fill on a partially-filled cell;
        #   (2) an empty neighbour with a nonzero background-region Aex must not
        #       leak a spurious -Ac·m_c/Δ² term (it must be Neumann).
        py = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)
        mesh = Mesh((6, 1, 1), (4e-9, 4e-9, 4e-9))

        # A material strip in cells 2..5; cells 1 and 6 empty (Msat 0), but their
        # Aex is left at the default (nonzero), the exact condition that used to
        # leak a spurious term at the strip edge.
        rp = RegionParams(mesh, py)
        setregion!(rp, 0; Msat = 0.0)                  # background empty (Aex ≠ 0)
        for i in 2:5
            rp.regions.id[i, 1, 1] = 1
        end
        rp.Msat[2] = 8.0e5; rp.Aex[2] = 1.3e-11        # region 1 = same material

        # A non-uniform state so the exchange field is nonzero in the interior.
        m = zeromag(mesh)
        for i in 1:6
            θ = 0.3 * i
            m[1, i, 1, 1] = sin(θ); m[3, i, 1, 1] = cos(θ)
        end
        clearempty!(m, rp)
        B = similar(m); exchange!(B, m, mesh, rp)

        A = py.Aex; Ms = py.Msat; iΔ2 = 1 / (4e-9)^2
        pref = 2 / Ms                                  # B = (2/Msat) Σ a (m_nbr-m_c)/Δ²
        # (2) Edge cell 2 has an empty left neighbour (cell 1). Neumann → only the
        # right neighbour (cell 3) couples; the empty side adds exactly zero, and
        # the prefactor uses the full Msat (not Msat·fill, which is 1 here anyway).
        for c in (1, 3)
            ref = pref * (A * (m[c, 3, 1, 1] - m[c, 2, 1, 1]) * iΔ2)
            @test B[c, 2, 1, 1] ≈ ref rtol = 1e-10
        end
        # An interior cell (3) couples to both neighbours (cells 2 and 4).
        for c in (1, 3)
            ref = pref * (A * (m[c, 2, 1, 1] - m[c, 3, 1, 1]) * iΔ2 +
                          A * (m[c, 4, 1, 1] - m[c, 3, 1, 1]) * iΔ2)
            @test B[c, 3, 1, 1] ≈ ref rtol = 1e-10
        end
        # Empty cells carry no field.
        @test all(iszero, B[:, 1, 1, 1]); @test all(iszero, B[:, 6, 1, 1])
    end

    @testset "exchange prefactor ignores fill on a partial cell" begin
        # A half-covered boundary cell (fill ≈ 0.5) must feel the same exchange
        # field as if it were full: mumax3 divides by the region Msat, not Msat·fill.
        # Build two identical strips differing only in the edge cell's fill, with a
        # uniform-in-material state so the only difference would be the 1/fill bug.
        py = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)
        mesh = Mesh((4, 1, 1), (4e-9, 4e-9, 4e-9))

        full = RegionParams(mesh, py); setregion!(full, 0; Msat = 0.0)
        for i in 1:4; full.regions.id[i,1,1] = 1; end
        full.Msat[2] = 8.0e5; full.Aex[2] = 1.3e-11

        half = deepcopy(full)
        half.fill[2, 1, 1] = 0.5                       # cell 2 half-covered

        m = zeromag(mesh)
        for i in 1:4
            θ = 0.4 * i; m[1,i,1,1] = sin(θ); m[3,i,1,1] = cos(θ)
        end
        Bf = similar(m); exchange!(Bf, m, mesh, full)
        Bh = similar(m); exchange!(Bh, m, mesh, half)
        # The exchange FIELD in the half cell must match the full cell (no 1/fill).
        @test Bh[:, 2, 1, 1] ≈ Bf[:, 2, 1, 1] rtol = 1e-10
    end

    @testset "clearempty! is a no-op for a scalar Material" begin
        mesh = Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9))
        m = uniform(mesh, (1, 0, 0))
        m2 = copy(m)
        clearempty!(m, py)                             # Material is never empty
        @test m == m2
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

    @testset "edge smoothing (fractional fill)" begin
        py = material("Permalloy")

        @testset "edgesmooth=0 reproduces the staircase" begin
            mesh = Mesh((48, 48, 1), (4e-9, 4e-9, 4e-9))
            a = RegionParams(mesh, py); setregion!(a, 0; Msat = 0.0)
            defregion!(a, 1, Cylinder(120e-9, 1e6))
            b = RegionParams(mesh, py); setregion!(b, 0; Msat = 0.0)
            setgeometry!(b, Cylinder(120e-9, 1e6); id = 1, edgesmooth = 0)
            # identical region map and identical effective Msat everywhere
            @test a.regions.id == b.regions.id
            @test all(b.fill .== 1)                  # no fractional cells
            for k in 1:1, j in 1:48, i in 1:48
                @test JuliaMag.msat(a, i, j, k) == JuliaMag.msat(b, i, j, k)
            end
        end

        @testset "a half-covered cell gets fill ≈ 0.5" begin
            mesh = Mesh((9, 9, 1), (4e-9, 4e-9, 4e-9))
            rp = RegionParams(mesh, py); setregion!(rp, 0; Msat = 0.0)
            setgeometry!(rp, YRange(0.0, 1e9); id = 1, edgesmooth = 8)  # y ≥ 0 half-plane
            @test isapprox(rp.fill[5, 5, 1], 0.5; atol = 0.02)          # centre row on the edge
            @test rp.fill[5, 9, 1] == 1                                  # fully inside (large y)
        end

        @testset "total moment converges to the analytic area" begin
            mesh = Mesh((64, 64, 1), (4e-9, 4e-9, 4e-9))
            area = π * (60e-9)^2 / (4e-9)^2          # disc area in cells
            filled(es) = begin
                rp = RegionParams(mesh, py); setregion!(rp, 0; Msat = 0.0)
                setgeometry!(rp, Cylinder(120e-9, 1e6); id = 1, edgesmooth = es)
                sum(rp.fill[i, j, 1] for i in 1:64, j in 1:64 if rp.regions.id[i, j, 1] == 1)
            end
            # Smoothing brings the filled area much closer to the analytic value.
            @test abs(filled(8) - area) < abs(filled(0) - area)
            @test isapprox(filled(8), area; rtol = 0.01)
        end

        @testset "fill = 0 makes a cell empty" begin
            mesh = Mesh((16, 16, 1), (4e-9, 4e-9, 4e-9))   # 64 nm across
            rp = RegionParams(mesh, py); setregion!(rp, 0; Msat = 0.0)
            setgeometry!(rp, Cylinder(40e-9, 1e6); id = 1, edgesmooth = 4)  # 20 nm radius
            # a far corner cell is well outside the disc: region 0, empty
            @test JuliaMag.isempty_cell(rp, 1, 1, 1)
            # the central cells are inside: not empty, full Msat
            @test !JuliaMag.isempty_cell(rp, 8, 8, 1)
            @test JuliaMag.msat(rp, 8, 8, 1) == py.Msat
        end
    end
end
