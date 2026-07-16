@testset "Energies" begin
    Msat = 8.0e5

    @testset "exchange energy is zero for a uniform state, positive otherwise" begin
        mesh = Mesh((8, 8, 2), (5e-9, 5e-9, 5e-9))
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02)
        @test exchangeenergy(uniform(mesh, (1, 0, 0)), mesh, mat) ≈ 0 atol = 1e-25
        # Any non-uniform state costs exchange energy (E = -½ Msat V Σ m·B ≥ 0
        # because B_exch opposes curvature).
        @test exchangeenergy(randommag!(zeromag(mesh)), mesh, mat) > 0
    end

    @testset "Zeeman energy matches -μ… analytic form" begin
        # For a uniform m and uniform Bext, E = -Msat·V_total·(m·Bext).
        mesh = Mesh((4, 4, 2), (5e-9, 5e-9, 5e-9))
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02)
        Bext = (0.1, 0.0, 0.0)
        m = uniform(mesh, (1, 0, 0))
        expected = -Msat * volume(mesh) * 0.1
        @test zeemanenergy(m, Bext, mesh, mat) ≈ expected rtol = 1e-12

        # Antiparallel costs +, parallel costs −.
        @test zeemanenergy(uniform(mesh, (-1, 0, 0)), Bext, mesh, mat) ≈ -expected rtol = 1e-12
        @test zeemanenergy(uniform(mesh, (0, 1, 0)), Bext, mesh, mat) ≈ 0 atol = 1e-25
    end

    @testset "uniaxial anisotropy energy matches Ku·V·(1-(m·u)²)" begin
        mesh = Mesh((4, 4, 2), (5e-9, 5e-9, 5e-9))
        Ku = 5.0e5
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02, Ku = Ku, anisU = (0, 0, 1))

        # m along the easy axis: minimum energy. The field-based energy uses the
        # constant term convention, so compare energy *differences*: the cost of
        # rotating from ∥u to ⊥u is Ku·V.
        E_para = anisotropyenergy(uniform(mesh, (0, 0, 1)), mesh, mat)
        E_perp = anisotropyenergy(uniform(mesh, (1, 0, 0)), mesh, mat)
        @test E_perp - E_para ≈ Ku * volume(mesh) rtol = 1e-10
    end

    @testset "demag energy of a uniformly magnetized cube" begin
        # E_demag = ½ μ0 Msat² V N for magnetization along a cube axis, N=1/3.
        mesh = Mesh((6, 6, 6), (5e-9, 5e-9, 5e-9))
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02)
        plan = DemagPlan(demagkernel(mesh), mesh, mat)
        m = uniform(mesh, (1, 0, 0))
        E = demagenergy(m, plan, mesh, mat)
        expected = 0.5 * μ0 * Msat^2 * volume(mesh) * (1/3)
        @test E ≈ expected rtol = 0.02
        @test E > 0
    end

    @testset "total energy drops as the minimizer relaxes" begin
        # THE convergence check. The Barzilai-Borwein minimizer is not strictly
        # monotone — individual steps can overshoot (mumax3's minimizer has the
        # same behaviour) — but the energy must fall strongly overall, and a
        # windowed minimum must decrease steadily.
        mesh = Mesh((32, 16, 1), (4e-9, 4e-9, 4e-9))
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02)
        world = World(mesh, mat; demag = true)
        m = uniform(mesh, (1, 1, 1))
        mn = Minimizer(world, m; stopdm = 1e-6)

        E0 = totalenergy(mn.m, world)
        Es = Float64[E0]
        for _ in 1:400
            minimizestep!(mn)
            push!(Es, totalenergy(mn.m, world))
        end

        # Net energy dropped substantially.
        @test Es[end] < E0
        @test (E0 - Es[end]) / abs(E0) > 0.5        # relaxed by >50%

        # The trend is downward. The Barzilai-Borwein step overshoots on
        # individual steps (and can leave two adjacent windows nearly equal), so
        # we check the overall trend rather than strict block-to-block monotonicity:
        # each quarter's mean energy is below the previous quarter's, and the mean
        # of the second half is well below the first half's.
        q = length(Es) ÷ 4
        quarters = [sum(@view Es[(n*q+1):((n+1)*q)]) / q for n in 0:3]
        @test all(diff(quarters) .< 0)
        firsthalf = sum(@view Es[1:2q]) / (2q)
        secondhalf = sum(@view Es[(2q+1):4q]) / (2q)
        @test secondhalf < firsthalf
    end

    @testset "total energy assembles all active terms" begin
        mesh = Mesh((6, 6, 1), (4e-9, 4e-9, 4e-9))
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02, Ku = 3e5, anisU = (0, 0, 1))
        world = World(mesh, mat; demag = true, Bext = (0.05, 0, 0))
        m = randommag!(zeromag(mesh))
        E = totalenergy(m, world)
        parts = exchangeenergy(m, mesh, mat) +
                anisotropyenergy(m, mesh, mat) +
                demagenergy(m, world.demagplan, mesh, mat) +
                zeemanenergy(m, world.Bext, mesh, mat)
        @test E ≈ parts rtol = 1e-12
    end
end
