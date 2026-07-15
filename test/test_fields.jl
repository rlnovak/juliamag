using LinearAlgebra: norm

@testset "Fields" begin
    py = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)

    @testset "exchange" begin
        mesh = Mesh((10, 8, 4), (4e-9, 4e-9, 4e-9))

        @testset "vanishes for a uniform state" begin
            for dir in ((1, 0, 0), (0, 0, 1), (1, 1, 1))
                m = uniform(mesh, dir)
                B = similar(m)
                exchange!(B, m, mesh, py)
                @test maximum(abs, B) < 1e-6   # ∇²(const) = 0, up to roundoff
            end
        end

        @testset "matches an analytic spin wave" begin
            # m = (sin(qx), 0, cos(qx)) has ∇²m = -q² (sin, 0, cos) exactly on the
            # continuum, so the discrete Laplacian's field should be
            # -(2A/μ0 Msat) q² m in the interior, up to the O(Δ²) stencil error.
            Nx = 64
            L = 400e-9
            cx = L / Nx
            mesh1 = Mesh((Nx, 1, 1), (cx, 4e-9, 4e-9))
            q = 2π / L                       # one full period across the sample
            m = zeromag(mesh1)
            for i in 1:Nx
                x = (i - 0.5) * cx
                m[1, i, 1, 1] = sin(q * x)
                m[3, i, 1, 1] = cos(q * x)
            end
            B = similar(m)
            exchange!(B, m, mesh1, py)

            pref = 2 * py.Aex / py.Msat        # Tesla field, no μ0 (mumax3 convention)
            # Compare in the interior only; boundaries use the Neumann condition.
            for i in 4:Nx-3, c in (1, 3)
                expected = -pref * q^2 * m[c, i, 1, 1]
                @test B[c, i, 1, 1] ≈ expected rtol = 1e-2
            end
        end

        @testset "Neumann boundary conserves total field of a domain wall" begin
            # ∇² is a sum of differences; with Neumann walls the exchange field
            # summed over all cells and components must be zero (no net torque
            # source from a free boundary).
            m = randommag!(zeromag(mesh))
            B = similar(m)
            exchange!(B, m, mesh, py)
            @test abs(sum(B)) < 1e-3 * maximum(abs, B) * length(B)
        end
    end

    @testset "anisotropy" begin
        mesh = Mesh((4, 4, 2), (5e-9, 5e-9, 5e-9))
        mat = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02,
                       Ku = 5.0e5, anisU = (0, 0, 1))

        @testset "zero when m ∥ easy axis, and aligned with u" begin
            m = uniform(mesh, (0, 0, 1))
            B = similar(m)
            anisotropy!(B, m, mesh, mat)
            # m·u = 1, so B = (2Ku/μ0Msat) u, purely along z.
            expected = 2 * mat.Ku / mat.Msat
            @test all(B[3, :, :, :] .≈ expected)
            @test maximum(abs, B[1:2, :, :, :]) < 1e-9
        end

        @testset "zero when m ⊥ easy axis" begin
            m = uniform(mesh, (1, 0, 0))
            B = similar(m)
            anisotropy!(B, m, mesh, mat)
            @test maximum(abs, B) < 1e-9    # m·u = 0
        end

        @testset "no anisotropy ⇒ zero field" begin
            m = uniform(mesh, (1, 1, 1))
            B = similar(m)
            anisotropy!(B, m, mesh, py)     # py has Ku = 0
            @test all(iszero, B)
        end
    end

    @testset "zeeman" begin
        mesh = Mesh((4, 4, 2), (5e-9, 5e-9, 5e-9))
        m = uniform(mesh, (1, 0, 0))
        B = similar(m)
        Bext = (-24.6e-3, 4.3e-3, 0.0)      # standard problem 4, field 1
        zeeman!(B, Bext)
        @test all(B[1, :, :, :] .≈ Bext[1])
        @test all(B[2, :, :, :] .≈ Bext[2])
        @test all(B[3, :, :, :] .≈ Bext[3])
    end

    @testset "add=true accumulates" begin
        mesh = Mesh((6, 6, 1), (4e-9, 4e-9, 4e-9))
        mat = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02,
                       Ku = 5.0e5, anisU = (0, 0, 1))
        m = randommag!(zeromag(mesh))

        # Assemble in one buffer with add.
        B = similar(m)
        exchange!(B, m, mesh, mat; add = false)
        anisotropy!(B, m, mesh, mat; add = true)
        zeeman!(B, (0.0, 0.0, 0.1); add = true)

        # Compare against separate buffers summed by hand.
        Be = similar(m); exchange!(Be, m, mesh, mat)
        Ba = similar(m); anisotropy!(Ba, m, mesh, mat)
        Bz = similar(m); zeeman!(Bz, (0.0, 0.0, 0.1))
        @test B ≈ Be .+ Ba .+ Bz
    end
end
