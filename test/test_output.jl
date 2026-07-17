@testset "Trackers and output" begin
    py = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)

    @testset "vortex core position and polarity" begin
        mesh = Mesh((41, 41, 1), (4e-9, 4e-9, 4e-9))
        # Core at the centre (translate 0).
        m = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1))
        x, y, z, pol = vortexcore(m, mesh)
        @test abs(x) < 4e-9 && abs(y) < 4e-9        # core near the centre
        @test pol > 0.9                             # +z polarity

        # Move the core to +8 cells in x.
        off = 8 * 4e-9
        m2 = setconfig(mesh, translate(VortexConfig(mesh; pol = -1), off, 0, 0))
        x2, y2, z2, pol2 = vortexcore(m2, mesh)
        @test x2 ≈ off atol = 4e-9                  # tracked to the shifted core
        @test pol2 < -0.9                           # -z polarity
    end

    @testset "topological charge of a skyrmion is ±1" begin
        mesh = Mesh((60, 60, 1), (3e-9, 3e-9, 3e-9))
        mp = setconfig(mesh, NeelSkyrmionConfig(mesh; charge = 1, pol = 1))
        Q = topologicalcharge(mp, mesh)
        @test abs(abs(Q) - 1) < 0.15                # ≈ ±1 (discretization)

        # A uniform state has zero charge.
        @test abs(topologicalcharge(uniform(mesh, (0,0,1)), mesh)) < 1e-6
    end

    @testset "skyrmion position tracks the core" begin
        mesh = Mesh((60, 60, 1), (3e-9, 3e-9, 3e-9))
        m = setconfig(mesh, NeelSkyrmionConfig(mesh; charge = 1, pol = 1))
        x, y, z = skyrmionpos(m, mesh)
        @test abs(x) < 5e-9 && abs(y) < 5e-9        # centred

        off = 15 * 3e-9
        m2 = setconfig(mesh, translate(NeelSkyrmionConfig(mesh), off, 0, 0))
        x2, _, _ = skyrmionpos(m2, mesh)
        @test x2 ≈ off atol = 6e-9
    end

    @testset "domain-wall position" begin
        mesh = Mesh((80, 8, 1), (4e-9, 4e-9, 4e-9))
        # Head-to-head wall centred: mx = +1 left, -1 right.
        m = setconfig(mesh, TwoDomainConfig(mesh, (1,0,0), (0,1,0), (-1,0,0)))
        x, y, z = domainwallpos(m, mesh)
        @test abs(x) < 6e-9                         # wall near the centre
    end

    @testset "DataTable: columns, header, rows, write" begin
        mesh = Mesh((8, 8, 1), (5e-9, 5e-9, 5e-9))
        world = World(mesh, py; demag = false, Bext = (0.1, 0, 0))
        m = uniform(mesh, (1, 0, 0))

        tbl = DataTable(columns = [q_time(), q_m(), q_energy(), q_Bext()])
        tableadd!(tbl, q_maxtorque())

        hdr = tableheader(tbl)
        @test hdr[1] == "t (s)"
        @test "mx" in hdr && "my" in hdr && "mz" in hdr        # 3 components
        @test "B_extx (T)" in hdr

        tablesave!(tbl, world, m, 0.0)
        tablesave!(tbl, world, m, 1e-12)
        @test length(tbl.rows) == 2
        @test length(tbl.rows[1]) == length(hdr)               # row width = header
        # time column, m column (uniform +x → mx=1)
        @test tbl.rows[1][1] == 0.0
        @test tbl.rows[1][2] ≈ 1.0                              # mx
        @test tbl.rows[2][1] ≈ 1e-12

        path = joinpath(mktempdir(), "table.txt")
        writetable(tbl, path)
        lines = readlines(path)
        @test startswith(lines[1], "#")
        @test length(lines) == 3                                # header + 2 rows
    end

    @testset "per-region magnetization average" begin
        mesh = Mesh((8, 8, 2), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(mesh, py)
        defregion!(rp, 1, Layers(mesh, 2, 3))       # top layer → region 1
        m = zeromag(mesh)
        m[1, :, :, 1] .= 1.0                         # bottom layer +x
        m[2, :, :, 2] .= 1.0                         # top layer +y

        a0 = average_region(m, rp, 0)               # bottom layer
        a1 = average_region(m, rp, 1)               # top layer
        @test a0[1] ≈ 1.0 && a0[2] ≈ 0.0
        @test a1[1] ≈ 0.0 && a1[2] ≈ 1.0

        # For a scalar Material, region 0 is the whole sample.
        @test all(average_region(m, py, 0) .≈ average(m))
    end
end
