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

        # mumax3 ext_corepos searches every z-layer: a core present only in the
        # top layer (bottom layer flat, m_z = 0) must still be found there.
        mesh3 = Mesh((41, 41, 2), (4e-9, 4e-9, 4e-9))
        m3 = setconfig(mesh3, translate(VortexConfig(mesh3; pol = 1), off, 0, 0))
        m3[1, :, :, 1] .= 1.0; m3[2, :, :, 1] .= 0.0; m3[3, :, :, 1] .= 0.0
        x3, y3, z3, pol3 = vortexcore(m3, mesh3)
        @test x3 ≈ off atol = 4e-9
        @test abs(y3) < 4e-9
        @test z3 ≈ 2e-9                             # centre of the top layer
        @test pol3 > 0.9
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

    @testset "skyrmion position is circular on a periodic axis" begin
        # Roll a centred skyrmion by half the stripe so it straddles the x wrap
        # seam. A linear centroid would report the middle; the circular centroid
        # on the periodic axis must follow the roll to the seam at ±L/2.
        Nx, cx = 60, 3e-9
        L = Nx * cx
        mesh = Mesh((Nx, 60, 1), (cx, 3e-9, 3e-9); pbc = (1, 0, 0))
        m0 = setconfig(mesh, NeelSkyrmionConfig(mesh; charge = 1, pol = 1))
        xc0, _, _ = skyrmionpos(m0, mesh)
        @test abs(xc0) < 5e-9                       # centred to start

        m = circshift(m0, (0, Nx ÷ 2, 0, 0))        # shift by L/2 along x
        xc, _, _ = skyrmionpos(m, mesh)
        # Now at the seam: |xc| ≈ L/2, i.e. near ±L/2, not near 0.
        @test min(abs(xc - L/2), abs(xc + L/2)) < 4e-9
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

    @testset "magnet average weights the cell fill (mumax3 table m)" begin
        # mumax3 engine/average.go sAverageMagnet: <m> = Σ fill·m / Σ fill.
        mesh = Mesh((16, 16, 2), (2e-9, 2e-9, 2e-9))
        disk(es) = (rp = RegionParams(mesh, py); setregion!(rp, 0; Msat = 0.0);
                    setgeometry!(rp, Cylinder(32e-9, 8e-9); id = 1, edgesmooth = es); rp)
        rp = disk(4)
        # Cells outside the shape keep region 0 (Msat = 0) with fill 1, so select
        # the magnet's cells by region, then split them by fill.
        disk1 = rp.regions.id .== 1
        full = disk1 .& (rp.fill .== 1)
        part = disk1 .& (rp.fill .< 1)
        @test any(part)
        m = zeromag(mesh)
        for k in 1:2, j in 1:16, i in 1:16
            full[i, j, k] && (m[1, i, j, k] = 1.0)     # interior +x
            part[i, j, k] && (m[2, i, j, k] = 1.0)     # fractional edge +y
        end
        W = count(full) + sum(rp.fill[part])
        a = average_magnet(m, rp)
        @test a[1] ≈ count(full) / W
        @test a[2] ≈ sum(rp.fill[part]) / W
        @test a[3] == 0
        # The unweighted region average counts edge cells fully — differs here.
        b = average_region(m, rp, 1)
        @test b[1] ≈ count(full) / (count(full) + count(part))
        @test !isapprox(a[1], b[1]; rtol = 1e-3)
        # q_m reports the magnet average, not the box average (which the empty
        # corners would dilute).
        w = World(mesh, rp; demag = false)
        @test q_m()(w, m, 0.0) ≈ collect(a)
        # Staircase geometry (fill 0/1): identical to the region average.
        rp0 = disk(0)
        m0 = randommag(mesh); clearempty!(m0, rp0)
        @test all(average_magnet(m0, rp0) .≈ average_region(m0, rp0, 1))
        # Scalar Material: plain mesh average.
        @test all(average_magnet(m0, py) .≈ average(m0))
    end
end
