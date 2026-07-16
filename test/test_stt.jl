using LinearAlgebra: norm, ⋅, ×

@testset "Spin-transfer torque" begin
    Msat = 8.0e5

    @testset "Zhang-Li torque is on the LLG scale (γ carried inside)" begin
        # Regression guard for the γLL bug: the STT must be comparable to the LLG
        # torque for a realistic current, not ~1e11 times smaller. In a vortex
        # texture with j = 1e12 A/m², the Zhang-Li torque should reach ~1e9 rad/s
        # or more — the same order as the LLG torque, so the current actually
        # moves the texture (std problem 5).
        mesh = Mesh((32, 32, 4), (100e-9/32, 100e-9/32, 10e-9/4))
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.1, pol = 1.0, xi = 0.05)
        m = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1))
        τ = zeromag(mesh)
        zhanglitorque!(τ, m, mesh, mat, (1e12, 0, 0); add = false)
        @test maxtorque(τ) > 1e8
    end

    @testset "Zhang-Li vanishes without polarization or current, and for uniform m" begin
        mesh = Mesh((16, 4, 1), (4e-9, 4e-9, 4e-9))
        # pol = 0 ⇒ no torque
        mat0 = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02)
        τ = zeromag(mesh)
        zhanglitorque!(τ, randommag!(zeromag(mesh)), mesh, mat0, (1e12, 0, 0))
        @test all(iszero, τ)

        # pol set but J = 0 ⇒ no torque
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02, pol = 1.0, xi = 0.05)
        τ = zeromag(mesh)
        zhanglitorque!(τ, randommag!(zeromag(mesh)), mesh, mat, (0, 0, 0); add = false)
        @test all(iszero, τ)

        # uniform m ⇒ (u·∇)m = 0 ⇒ no torque
        τ = zeromag(mesh)
        zhanglitorque!(τ, uniform(mesh, (1, 0, 0)), mesh, mat, (1e12, 0, 0); add = false)
        @test maximum(abs, τ) < 1e-6
    end

    @testset "Zhang-Li drives a non-uniform texture" begin
        # A gradient along x with current along x gives a non-zero torque.
        mesh = Mesh((32, 1, 1), (4e-9, 4e-9, 4e-9))
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02, pol = 0.5, xi = 0.1)
        m = zeromag(mesh)
        for i in 1:32                       # smooth rotation in the x-z plane
            θ = π * (i - 1) / 31
            m[1, i, 1, 1] = cos(θ)
            m[3, i, 1, 1] = sin(θ)
        end
        τ = zeromag(mesh)
        zhanglitorque!(τ, m, mesh, mat, (1e12, 0, 0); add = false)
        @test maximum(abs, τ) > 0            # the wall feels a torque
        # torque is perpendicular to m at each cell (both terms are cross products with m)
        for i in 3:30
            mi = m[:, i, 1, 1]; ti = τ[:, i, 1, 1]
            @test abs(ti ⋅ mi) < 1e-6 * (norm(ti) + eps())
        end
    end

    @testset "Slonczewski torque is ⊥ m and vanishes when m ∥ p" begin
        mesh = Mesh((4, 4, 1), (4e-9, 4e-9, 4e-9))
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02, pol = 0.5, lambda = 1.5)
        thickness = 3e-9
        p = (0, 0, 1)

        # m ⊥ p: nonzero torque, perpendicular to m
        m = uniform(mesh, (1, 0, 0))
        τ = zeromag(mesh)
        slonczewskitorque!(τ, m, mesh, mat, 1e12, p, thickness; add = false)
        @test maximum(abs, τ) > 0
        for I in CartesianIndices((4, 4, 1))
            ti = τ[:, I]; mi = m[:, I]
            @test abs(ti ⋅ mi) < 1e-6 * norm(ti)
        end

        # m ∥ p: p×m = 0 ⇒ torque = 0
        m = uniform(mesh, (0, 0, 1))
        τ = zeromag(mesh)
        slonczewskitorque!(τ, m, mesh, mat, 1e12, p, thickness; add = false)
        @test maximum(abs, τ) < 1e-3        # numerically ~0
    end

    @testset "Slonczewski vanishes without polarization or current" begin
        mesh = Mesh((4, 4, 1), (4e-9, 4e-9, 4e-9))
        mat0 = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02)   # pol = 0
        τ = zeromag(mesh)
        slonczewskitorque!(τ, uniform(mesh, (1, 0, 0)), mesh, mat0, 1e12, (0, 0, 1), 3e-9)
        @test all(iszero, τ)

        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02, pol = 0.5)
        τ = zeromag(mesh)
        slonczewskitorque!(τ, uniform(mesh, (1, 0, 0)), mesh, mat, 0.0, (0, 0, 1), 3e-9)
        @test all(iszero, τ)
    end

    @testset "STT adds onto an existing (LLG) torque" begin
        mesh = Mesh((8, 4, 1), (4e-9, 4e-9, 4e-9))
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02, pol = 0.5, xi = 0.1)
        m = randommag!(zeromag(mesh))
        base = randommag!(zeromag(mesh))          # pretend LLG torque

        τ = copy(base)
        zhanglitorque!(τ, m, mesh, mat, (1e12, 0, 0); add = true)
        stt = zeromag(mesh)
        zhanglitorque!(stt, m, mesh, mat, (1e12, 0, 0); add = false)
        @test τ ≈ base .+ stt
    end

    @testset "Slonczewski β prefactor matches ħJ/(qe t Msat)" begin
        # Check the overall magnitude against the analytic prefactor for a simple
        # geometry: m ⊥ p, Λ=1 (ε = pol/2), ε'=0, α small.
        mesh = Mesh((2, 2, 1), (4e-9, 4e-9, 4e-9))
        pol = 0.4
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.0, pol = pol, lambda = 1.0)
        Jz = 5e12; thickness = 2e-9
        m = uniform(mesh, (1, 0, 0)); p = (0, 0, 1)
        τ = zeromag(mesh)
        slonczewskitorque!(τ, m, mesh, mat, Jz, p, thickness; add = false)

        # γLL is carried inside JuliaMag's torque (mumax applies it in the integrator).
        β = JuliaMag.γLL * (JuliaMag.ħ / JuliaMag.qe) * Jz / (thickness * Msat)
        ε = pol * 1.0 / (2.0 + 0.0)          # Λ=1: ε = pol/2
        A = β * ε
        # α=0, ε'=0 ⇒ τ = A·m×(p×m). |m×(p×m)| = 1 for m⊥p.
        @test maximum(abs, τ) ≈ abs(A) rtol = 1e-6
    end
end
