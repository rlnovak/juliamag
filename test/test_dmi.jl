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

    @testset "bulk field matches mumax3 term by term (interior, 3D state)" begin
        # mumax3 (cuda/dmibulk.cu):
        #   H_x = (2D/Msat)(∂z my - ∂y mz)
        #   H_y = (2D/Msat)(∂x mz - ∂z mx)
        #   H_z = (2D/Msat)(∂y mx - ∂x my)
        # Impose an analytic m(x,y,z) and compare central differences directly.
        Nx = Ny = Nz = 8
        c = 3e-9
        mesh = Mesh((Nx, Ny, Nz), (c, c, c); pbc = (1, 1, 1))   # periodic → no boundary terms
        D = 1.5e-3; Msat = 8e5
        mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02, Dbulk = D)

        kx = 2π / (Nx*c); ky = 2π / (Ny*c); kz = 2π / (Nz*c)
        m = zeromag(mesh)
        f(i, j, k) = ((i-0.5)*c, (j-0.5)*c, (k-0.5)*c)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            x, y, z = f(i, j, k)
            v = (sin(kx*x), sin(ky*y), sin(kz*z))
            n = sqrt(sum(abs2, v)) + 1e-9
            m[1,i,j,k], m[2,i,j,k], m[3,i,j,k] = v ./ n
        end

        B = similar(m); dmi!(B, m, mesh, mat)
        pref = 2 * D / Msat
        wrap(a, N) = mod(a - 1, N) + 1
        dcen(comp, ax, i, j, k) = begin
            step = (ax == 1, ax == 2, ax == 3)
            ip = (wrap(i + step[1], Nx), wrap(j + step[2], Ny), wrap(k + step[3], Nz))
            im = (wrap(i - step[1], Nx), wrap(j - step[2], Ny), wrap(k - step[3], Nz))
            spacing = 2 * c
            (m[comp, ip...] - m[comp, im...]) / spacing
        end
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            Hx = pref * (dcen(2,3,i,j,k) - dcen(3,2,i,j,k))   # ∂z my - ∂y mz
            Hy = pref * (dcen(3,1,i,j,k) - dcen(1,3,i,j,k))   # ∂x mz - ∂z mx
            Hz = pref * (dcen(1,2,i,j,k) - dcen(2,1,i,j,k))   # ∂y mx - ∂x my
            @test B[1,i,j,k] ≈ Hx rtol = 1e-6 atol = 1e-4
            @test B[2,i,j,k] ≈ Hy rtol = 1e-6 atol = 1e-4
            @test B[3,i,j,k] ≈ Hz rtol = 1e-6 atol = 1e-4
        end
    end

    @testset "helical ground state has near-zero total torque" begin
        # With exchange + bulk DMI and PBC, the equilibrium is a helix of period
        # L = 4πA/D. Seeded at that period, the net torque should be small
        # (exchange and DMI balance) — far smaller than for a wrong period.
        A = 1.3e-11; D = 3e-3; Msat = 8e5
        L = 4π * A / D
        c = L / 32
        mesh = Mesh((32, 1, 1), (c, 5e-9, 5e-9); pbc = (1, 0, 0))
        mat = Material(Msat = Msat, Aex = A, alpha = 0.5, Dbulk = D)
        world = World(mesh, mat; demag = false)

        # Bloch helix in the x-y…z plane: m = (0, sin(qx), cos(qx)), q = 2π/L.
        q = 2π / L
        function helix(period)
            qq = 2π / period
            m = zeromag(mesh)
            for i in 1:32
                x = (i - 0.5) * c
                m[2, i, 1, 1] = sin(qq * x)
                m[3, i, 1, 1] = cos(qq * x)
            end
            m
        end

        B = similar(zeromag(mesh)); dm = similar(B)
        mgood = helix(L)
        effectivefield!(B, mgood, world); torque!(dm, mgood, B, mat.alpha)
        tgood = maxtorque(dm)

        mbad = helix(L / 2)          # wrong period
        effectivefield!(B, mbad, world); torque!(dm, mbad, B, mat.alpha)
        tbad = maxtorque(dm)

        @test tgood < tbad           # the correct-period helix is closer to equilibrium
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
