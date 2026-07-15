@testset "DMI" begin
    @testset "vanishes for a uniform state (both forms)" begin
        mesh = Mesh((8, 8, 2), (4e-9, 4e-9, 4e-9))
        for (kw, val) in ((:Dind, 1e-3), (:Dbulk, 1e-3))
            mat = Material(; Msat = 8e5, Aex = 1.3e-11, alpha = 0.02, kw => val)
            for dir in ((1, 0, 0), (0, 0, 1), (1, 1, 1))
                m = uniform(mesh, dir)
                B = similar(m)
                dmi!(B, m, mesh, mat)
                @test maximum(abs, B) < 1e-6      # ∇(const) = 0
            end
        end
    end

    @testset "field is antisymmetric in D" begin
        mesh = Mesh((8, 8, 1), (4e-9, 4e-9, 4e-9))
        m = randommag!(zeromag(mesh))

        for kw in (:Dind, :Dbulk)
            matp = Material(; Msat = 8e5, Aex = 1.3e-11, alpha = 0.02, kw => 2e-3)
            matm = Material(; Msat = 8e5, Aex = 1.3e-11, alpha = 0.02, kw => -2e-3)
            Bp = similar(m); dmi!(Bp, m, mesh, matp)
            Bm = similar(m); dmi!(Bm, m, mesh, matm)
            @test Bp ≈ -Bm
        end
    end

    @testset "interfacial field matches its analytic form" begin
        # Impose a smooth mz(x) ramp and check B_x = (2D/μ0 Msat) ∂mz/∂x in the
        # interior, where the central difference is exact for a linear field.
        Nx = 40
        cx = 2e-9
        mesh = Mesh((Nx, 1, 1), (cx, 4e-9, 4e-9))
        D = 3e-3
        Msat = 8e5
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02, Dind = D)

        # mz linear in x, mx making up the unit norm; slope small enough to stay
        # on the sphere across the sample.
        slope = 0.005 / cx
        m = zeromag(mesh)
        for i in 1:Nx
            x = (i - Nx/2) * cx
            mz = slope * x
            m[3, i, 1, 1] = mz
            m[1, i, 1, 1] = sqrt(1 - mz^2)
        end

        B = similar(m); dmi!(B, m, mesh, mat)
        pref = 2 * D / Msat
        for i in 3:Nx-2
            @test B[1, i, 1, 1] ≈ pref * slope rtol = 1e-6      # B_x = pref ∂mz/∂x
        end
    end

    @testset "bulk field equals -(2D/Msat) ∇×m" begin
        # A helical state m = (cos(qx), sin(qx), 0) has ∇×m = (0, 0, q cos(qx)),
        # so B = -(2D/Msat)(0,0,q cos qx). Check the z-component in the interior.
        Nx = 64
        L = 256e-9
        cx = L / Nx
        mesh = Mesh((Nx, 1, 1), (cx, 4e-9, 4e-9))
        D = 1e-3
        Msat = 8e5
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02, Dbulk = D)
        q = 2π / L

        m = zeromag(mesh)
        for i in 1:Nx
            x = (i - 0.5) * cx
            m[1, i, 1, 1] = cos(q * x)
            m[2, i, 1, 1] = sin(q * x)
        end

        B = similar(m); dmi!(B, m, mesh, mat)
        pref = 2 * D / Msat
        for i in 4:Nx-3
            x = (i - 0.5) * cx
            expected = -pref * q * cos(q * x)
            @test B[3, i, 1, 1] ≈ expected rtol = 2e-2
            @test abs(B[1, i, 1, 1]) < 1e-6 * abs(pref * q)   # curl_x = curl_y = 0
            @test abs(B[2, i, 1, 1]) < 1e-6 * abs(pref * q)
        end
    end

    @testset "no DMI ⇒ zero field, and effective field unaffected" begin
        mesh = Mesh((6, 6, 1), (4e-9, 4e-9, 4e-9))
        mat = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02)   # no DMI
        m = randommag!(zeromag(mesh))
        B = similar(m); dmi!(B, m, mesh, mat)
        @test all(iszero, B)

        # A world with no DMI gives the same effective field as before.
        w = World(mesh, mat; demag = false)
        Beff = similar(m); effectivefield!(Beff, m, w)
        Bex = similar(m); exchange!(Bex, m, mesh, mat)
        @test Beff ≈ Bex
    end

    @testset "DMI enters the effective field when set" begin
        mesh = Mesh((6, 6, 1), (4e-9, 4e-9, 4e-9))
        mat = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02, Dind = 2e-3)
        m = randommag!(zeromag(mesh))
        w = World(mesh, mat; demag = false)

        Beff = similar(m); effectivefield!(Beff, m, w)
        Bex = similar(m); exchange!(Bex, m, mesh, mat)
        Bd = similar(m); dmi!(Bd, m, mesh, mat)
        @test Beff ≈ Bex .+ Bd
    end

    @testset "material without DMI still constructs and keeps its type" begin
        mat = Material(Msat = 8.0f5, Aex = 1.3f-11, alpha = 0.02f0)
        @test mat.Dind == 0
        @test mat.Dbulk == 0
        @test eltype(mat) === Float32
    end
end
