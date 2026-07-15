using LinearAlgebra: norm, ⋅

@testset "Dynamics" begin

    @testset "torque: precession is ⊥ to both m and B" begin
        mesh = Mesh((1, 1, 1), (5e-9, 5e-9, 5e-9))
        m = uniform(mesh, (1, 0, 0))
        B = zeromag(mesh); B[:, 1, 1, 1] = [0, 0, 1.0]     # field along z
        dm = similar(m)
        torque!(dm, m, B, 0.0)                              # no damping
        d = dm[:, 1, 1, 1]
        # Undamped torque is -γ m×B, perpendicular to both.
        @test abs(d ⋅ m[:, 1, 1, 1]) < 1e-3 * norm(d)
        @test abs(d ⋅ B[:, 1, 1, 1]) < 1e-3 * norm(d)
    end

    @testset "torque: damping drives m toward B" begin
        mesh = Mesh((1, 1, 1), (5e-9, 5e-9, 5e-9))
        m = uniform(mesh, (1, 0, 0))
        B = zeromag(mesh); B[:, 1, 1, 1] = [0, 0, 1.0]
        dm = similar(m)
        torque!(dm, m, B, 0.1)                              # with damping
        # The damping component pulls m's z up toward B.
        @test dm[3, 1, 1, 1] > 0
    end

    @testset "Larmor precession frequency (undamped macrospin)" begin
        # A single spin in a uniform field, no demag/exchange/anisotropy,
        # precesses at ω = γ B with no change in the component along B.
        B0 = 0.1                                             # Tesla, along z
        mesh = Mesh((1, 1, 1), (5e-9, 5e-9, 5e-9))
        mat = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.0)
        world = World(mesh, mat; demag = false, Bext = (0, 0, B0))
        m = uniform(mesh, (1, 0, 0))                         # start ⊥ to B

        solver = Solver(world, m; dt = 1e-13, maxerr = 1e-7, maxdt = 1e-12)
        ω = γLL * B0
        period = 2π / ω

        # Integrate a quarter period: mx should rotate from 1 toward 0, my grow.
        runtime!(solver, period / 4)
        @test solver.m[3, 1, 1, 1] ≈ 0 atol = 1e-3          # ‖-component conserved
        @test abs(solver.m[1, 1, 1, 1]) < 0.1               # rotated out of x
        @test abs(norm(solver.m[:, 1, 1, 1]) - 1) < 1e-6    # |m| preserved

        # After a full period it returns near the start.
        runtime!(solver, 3 * period / 4)
        @test solver.m[1, 1, 1, 1] ≈ 1.0 atol = 2e-2
        @test solver.m[2, 1, 1, 1] ≈ 0.0 atol = 2e-2
    end

    @testset "damped macrospin relaxes to the field direction" begin
        B0 = 0.1
        mesh = Mesh((1, 1, 1), (5e-9, 5e-9, 5e-9))
        mat = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.5)   # heavy damping
        world = World(mesh, mat; demag = false, Bext = (0, 0, B0))
        m = uniform(mesh, (1, 0, 0))
        solver = Solver(world, m; dt = 1e-13, maxerr = 1e-6)

        relax!(solver; stoptorque = 1e5)
        # m ends up aligned with B (+z).
        @test solver.m[3, 1, 1, 1] > 0.999
        @test abs(norm(solver.m[:, 1, 1, 1]) - 1) < 1e-6
    end

    @testset "|m| stays normalized over a many-cell run" begin
        # A smooth (nearly uniform) state, not a cell-by-cell random one: random
        # magnetization on a 5 nm grid has an enormous exchange field and no
        # explicit solver survives it without relaxing first. A small tilt is a
        # realistic starting point.
        mesh = Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9))
        mat = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.1)
        world = World(mesh, mat; demag = true)
        m = uniform(mesh, (1, 0, 0))
        m[3, :, :, :] .= 0.1                    # small out-of-plane tilt
        normalize!(m)
        solver = Solver(world, m; dt = 1e-15, maxerr = 1e-5)

        for _ in 1:50
            step!(solver)
        end
        for k in 1:1, j in 1:4, i in 1:4
            @test abs(norm(solver.m[:, i, j, k]) - 1) < 1e-6
        end
        @test solver.t > 0
        @test solver.step == 50
    end

    @testset "adaptive step control accepts and rejects" begin
        mesh = Mesh((2, 2, 1), (5e-9, 5e-9, 5e-9))
        mat = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)
        world = World(mesh, mat; demag = false, Bext = (0, 0, 0.1))
        m = uniform(mesh, (1, 0, 0))
        # Start with a wildly too-large step and a tight tolerance: the
        # controller must reject and shrink until the error fits.
        solver = Solver(world, m; dt = 1e-9, maxerr = 1e-8, maxdt = 1e-9)
        step!(solver)
        @test solver.nfail > 0            # at least one rejected trial
        @test solver.dt < 1e-9            # step was shrunk
    end
end
