@testset "Material library and Simulation wrapper" begin
    @testset "material library" begin
        py = material("Permalloy")
        @test py.Msat == 8.0e5
        @test py.Aex == 1.3e-11
        @test py.Ku == 0.0
        @test material("Py").Msat == py.Msat        # alias
        @test material("permalloy").Msat == py.Msat # case-insensitive

        co = material("Co")
        @test co.Msat == 1.4e6
        @test co.Ku > 0                             # uniaxial anisotropy

        # override damping
        @test material("Co"; alpha = 0.05).alpha == 0.05

        @test "Permalloy" in materialnames()
        @test "CoFeB" in materialnames()
        @test_throws ArgumentError material("Unobtainium")
    end

    @testset "Simulation: build, set state, save, relax, run" begin
        m = Mesh((16, 16, 1), (5e-9, 5e-9, 5e-9))
        sim = Simulation(m, material("Permalloy"); demag = true)
        @test JuliaMag.mesh(sim) === m

        setmag!(sim, VortexConfig(m; circ = 1, pol = 1))
        @test average(sim)[3] > 0                   # vortex core +z

        savequantities!(sim, q_time(), q_m(), q_vortexcore(); every = 50e-12)
        @test sim.table.autosave == 50e-12

        # A short relax + run; just check the table fills and time advances.
        relax!(sim; stopdm = 1e-5)
        run!(sim, 2e-10)
        @test sim.t ≈ 2e-10 rtol = 1e-6
        # autosave every 50 ps over 200 ps ⇒ 1 initial + 4 steps = 5 rows.
        @test length(sim.table.rows) >= 4
        @test all(isfinite, sim.table.rows[end])
    end

    @testset "Simulation: no-autosave run saves start and end" begin
        m = Mesh((8, 8, 1), (5e-9, 5e-9, 5e-9))
        sim = Simulation(m, material("Permalloy"); demag = false, Bext = (0.05, 0, 0))
        setmag!(sim, UniformConfig(1, 0, 0))
        savequantities!(sim, q_time(), q_m())        # every = 0
        run!(sim, 1e-11)
        @test length(sim.table.rows) == 2            # start + end
        @test sim.table.rows[1][1] == 0.0
        @test sim.table.rows[2][1] ≈ 1e-11 rtol = 1e-6
    end

    @testset "Simulation with a region material runs" begin
        m = Mesh((8, 8, 4), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(m, material("Permalloy"))
        defregion!(rp, 1, Layers(m, 3, 5))
        setregion!(rp, 1; Msat = material("CoFeB").Msat, Aex = material("CoFeB").Aex)
        sim = Simulation(m, rp; demag = true)
        setmag!(sim, UniformConfig(1, 0, 0))
        savequantities!(sim, q_time(), q_m(), q_m_region(0), q_m_region(1))
        savenow!(sim)
        @test length(sim.table.rows) == 1
        @test all(isfinite, sim.table.rows[1])
    end
end
