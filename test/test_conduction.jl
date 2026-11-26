using JuliaMag, Test, Random

@testset "conduction with AMR" begin
    ρpar, ρperp = 210e-9, 200e-9
    Δρ = ρpar - ρperp
    h = 2e-9
    # a strip of 100 × 10 cells; contacts: the first and the last column
    function strip(θ; nx = 200, ny = 10, contact1 = (; current = 1e-3), contact2 = (; potential = 0.0), mesh = Mesh((nx, ny, 1), (h, h, h)),
                   extra = rp -> nothing, T = Float64)
        rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.1, pol = 0.5))
        defregion!(rp, 1, XRange(-Inf, -(nx / 2 - 1) * h)); defregion!(rp, 2, XRange((nx / 2 - 1) * h, Inf))
        extra(rp)
        w = World(mesh, T === Float64 ? rp : RegionParams{T}(rp); demag = false)
        setconduction!(w; rho_par = ρpar, rho_perp = ρperp)
        addcontact!(w, 1; contact1...); addcontact!(w, 2; contact2...)
        m = uniform(mesh, (cos(θ), sin(θ), 0))
        return w, m, mesh
    end

    @testset "isotropic: R of the discrete strip" begin
        w, m, mesh = strip(0.0)
        setresistivity!(w, 0; rho_par = ρperp); setresistivity!(w, 1; rho_par = ρperp); setresistivity!(w, 2; rho_par = ρperp)
        c = solveconduction!(w, m)
        R = (c.contacts[1].V - c.contacts[2].V) / c.contacts[1].I
        @test c.contacts[1].I ≈ 1e-3 rtol = 1e-12
        @test c.contacts[2].I ≈ -1e-3 rtol = 1e-7                 # what enters leaves (to the tolerance of the solver)
        @test R ≈ ρperp * 198h / (10h * h) rtol = 1e-8              # the sample between the faces of the contacts
    end

    @testset "uniform m at angle θ: AMR and planar Hall field in the middle" begin
        for θ in (0.0, π / 6, π / 4, π / 3, π / 2)
            w, m, mesh = strip(θ)
            c = solveconduction!(w, m)
            φ = c.phi; jx = 1e-3 / (10h * h)
            Ex = -(φ[1, 101, 5, 1] - φ[1, 99, 5, 1]) / 2h
            Ey = -(φ[1, 100, 9, 1] - φ[1, 100, 2, 1]) / 7h
            @test Ex / jx ≈ ρperp + Δρ * cos(θ)^2 rtol = 1e-7
            @test Ey / jx ≈ Δρ * sin(θ) * cos(θ) atol = 1e-7 * ρperp
            # the current through every cross-section between the contacts is the contact current
            Ix = [sum(c.j[1, i, :, 1]) * h * h for i in 3:198]
            @test all(x -> isapprox(x, 1e-3; rtol = 1e-7), Ix)
            # the electrodes short the Hall field over a few widths (decay ~ exp(−πx/W)); far from them j ∥ x
            @test maximum(abs, c.j[2, 90:110, :, 1]) < 1e-9 * jx
            # m → −m: the same
            c2 = solveconduction!(w, -m)
            @test c2.phi ≈ c.phi rtol = 1e-9
        end
    end

    @testset "contacts: potential, current, ground, functions of time" begin
        θ = 0.4
        w1, m, = strip(θ; contact1 = (; potential = 0.3), contact2 = (; potential = -0.2))
        c1 = solveconduction!(w1, m); R = 0.5 / c1.contacts[1].I
        w2, = strip(θ; contact1 = (; current = c1.contacts[1].I), contact2 = (; current = -c1.contacts[1].I))
        c2 = solveconduction!(w2, m)                                    # the second is the ground
        @test c2.contacts[2].V == 0
        @test c2.contacts[1].V - c2.contacts[2].V ≈ 0.5 rtol = 1e-7
        @test maximum(abs.((c2.phi .- c2.contacts[1].V) .- (c1.phi .- 0.3))) < 1e-8
        w3, = strip(θ; contact1 = (; current = t -> 1e-3 * (1 + 1e9 * t)))
        JuliaMag._drive!(w3, 1e-9)
        @test solveconduction!(w3, m).contacts[1].I ≈ 2e-3 rtol = 1e-9
        @test_throws ArgumentError addcontact!(w3, 2; potential = 1.0)
        @test_throws ArgumentError addcontact!(w3, 3)
    end

    @testset "spin torques from the conduction" begin
        θ = 0.3
        w, m, mesh = strip(θ)
        Random.seed!(2); mt = normalize!(m .+ 0.2 .* randn(size(m)...))
        setcurrent!(w, (0, 0, 0); zhangli = true)
        τ = llgrhs!(similar(mt), mt, w)
        c = solveconduction!(w, mt)
        u = World(mesh, w.material; demag = false); setcurrent!(u, (0, 0, 0)); addcurrent!(u, Array(c.j))
        @test τ ≈ llgrhs!(similar(mt), mt, u) rtol = 1e-12
        # with Δρ = 0 the current is uniform between the contacts: the torque of a uniform J there
        setresistivity!(w, 0; rho_par = ρperp); setresistivity!(w, 1; rho_par = ρperp); setresistivity!(w, 2; rho_par = ρperp)
        τ0 = llgrhs!(similar(mt), mt, w)
        v = World(mesh, w.material; demag = false); setcurrent!(v, (1e-3 / (10h * h), 0, 0))
        τu = llgrhs!(similar(mt), mt, v)
        @test τ0[:, 10:190, :, :] ≈ τu[:, 10:190, :, :] rtol = 1e-7
    end

    Rof(c) = (c.contacts[1].V - c.contacts[2].V) / c.contacts[1].I

    @testset "Float32 world: solved in Float64" begin
        θ = 0.5
        w, m, = strip(θ); c = solveconduction!(w, m)
        w32, = strip(θ; T = Float32); c32 = solveconduction!(w32, Float32.(m))
        @test c32.contacts[2].I ≈ -1e-3 rtol = 1e-9                     # no charge lost
        @test maximum(abs, c32.j .- c.j) < 1e-6 * maximum(abs, c.j)      # (m rounded to Float32)
        τ = llgrhs!(similar(Float32.(m)), Float32.(m), (setcurrent!(w32, (0, 0, 0)); w32))
        @test all(isfinite, τ)
    end

    @testset "islands, inconsistent contacts" begin
        cut(rp) = (setregion!(rp, 5; Msat = 0.0); defregion!(rp, 5, XRange(-h, h)))           # an insulating column
        w, m, = strip(0.2; extra = cut)
        @test_throws ErrorException solveconduction!(w, m)          # contact 1 not connected to the potential contact
        w, m, = strip(0.2; extra = cut, contact1 = (; potential = 1.0))
        c = solveconduction!(w, m)
        @test abs(c.contacts[1].I) < 1e-13 && abs(c.contacts[2].I) < 1e-13
        # an insulating block (6 × 8 cells) over the strip; a 2 × 2 island inside it, and a cell inside it
        # that touches a conducting cell at a corner only
        cellat(xc, yc) = (x, y, z) -> abs(x - xc) < h / 2 && abs(y - yc) < h / 2
        block(rp) = (setregion!(rp, 5; Msat = 0.0); defregion!(rp, 5, (x, y, z) -> abs(x) < 3h && y > -3h))
        dent(rp) = (block(rp); defregion!(rp, 0, cellat(-2.5h, -2.5h)))
        hole(rp) = (block(rp); defregion!(rp, 0, (x, y, z) -> abs(x) < h && abs(y - 2h) < h))
        corner(rp) = (dent(rp); defregion!(rp, 0, cellat(-1.5h, -1.5h)))
        for (f, ref) in ((hole, block), (corner, dent))
            R0 = Rof(solveconduction!(strip(0.2; extra = ref)[1:2]...))
            w, m, = strip(0.2; extra = f)
            c = @test_logs (:warn, r"not connected") solveconduction!(w, m)
            @test Rof(c) ≈ R0 rtol = 1e-10
        end
        # contacts fed currents only: they must sum to zero, and there must be two
        w, m, = strip(0.2; contact1 = (; current = 1e-3), contact2 = (; current = 5e-3))
        @test_throws ErrorException solveconduction!(w, m)
        mesh = Mesh((40, 4, 1), (h, h, h)); rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1e-11, alpha = 0.1))
        defregion!(rp, 1, XRange(-Inf, -19h)); w = World(mesh, rp; demag = false)
        setconduction!(w; rho_par = ρpar, rho_perp = ρperp); addcontact!(w, 1; current = 1e-3)
        @test_throws ErrorException solveconduction!(w, uniform(mesh, (1, 0, 0)))
        # contacts at nearly the same potential: the accuracy follows the difference
        w1, m, = strip(0.4; contact1 = (; potential = 1.0), contact2 = (; potential = 0.0))
        w2, = strip(0.4; contact1 = (; potential = 1.0), contact2 = (; potential = 0.999))
        c1 = solveconduction!(w1, m); R1 = 1 / c1.contacts[1].I
        @test 0.001 / solveconduction!(w2, m).contacts[1].I ≈ R1 rtol = 1e-9
    end

    @testset "changes of the regions; leads without Msat; contacts carry no j" begin
        block(rp) = (setregion!(rp, 5; Msat = 0.0); defregion!(rp, 5, (x, y, z) -> abs(x) < 10h && y > -2h))
        w, m, = strip(0.3); R1 = Rof(solveconduction!(w, m))
        block(w.material)                                         # after the first solve
        R2 = Rof(solveconduction!(w, m))
        @test R2 > 1.05R1
        @test R2 ≈ Rof(solveconduction!(strip(0.3; extra = block)[1:2]...)) rtol = 1e-8       # (warm against cold start)
        # a lead without Msat is isotropic even where m ≠ 0 (World API, m not cleared)
        lead(rp) = (setregion!(rp, 6; Msat = 0.0); defregion!(rp, 6, XRange(-50h, 50h)))
        w, m, mesh = strip(0.0; extra = lead)
        setresistivity!(w, 6; rho_par = ρpar, rho_perp = ρperp)
        wi, = strip(0.0; extra = lead); setresistivity!(wi, 6; rho_par = ρperp, rho_perp = ρperp)
        @test Rof(solveconduction!(w, m)) ≈ Rof(solveconduction!(wi, m)) rtol = 1e-10
        c = solveconduction!(w, m)
        @test all(==(0), c.j[:, 1, :, 1]) && all(==(0), c.j[:, end, :, 1])
    end

    @testset "partially filled cells" begin
        # a strip 10.5 cells wide on a 12-cell grid: the edge cells are a quarter full (edge smoothing)
        nx, ny = 100, 12
        mesh = Mesh((nx, ny, 1), (h, h, h))
        rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.1))
        setregion!(rp, 0; Msat = 0.0)
        setgeometry!(rp, (x, y, z) -> abs(y) < 5.25h; id = 1, edgesmooth = 8)
        @test count(f -> 0 < f < 1, rp.fill) == 2nx
        defregion!(rp, 2, XRange(-Inf, -(nx / 2 - 1) * h)); defregion!(rp, 3, XRange((nx / 2 - 1) * h, Inf))
        w = World(mesh, rp; demag = false)
        setconduction!(w; rho_par = ρperp, rho_perp = ρperp)
        addcontact!(w, 2; current = 1e-3); addcontact!(w, 3; potential = 0.0)
        c = solveconduction!(w, uniform(mesh, (1, 0, 0)))
        @test Rof(c) ≈ ρperp * (nx - 2) * h / (10.5h * h) rtol = 1e-9
        j = 1e-3 / (10.5h * h)                                   # the current density of the material, edge cells too
        @test all(x -> isapprox(x, j; rtol = 1e-8), c.j[1, 3:nx-2, :, 1])
    end

    @testset "3D: m out of plane" begin
        mesh = Mesh((60, 4, 3), (h, h, h))
        w, m, = strip(0.0; nx = 60, ny = 4, mesh = mesh)
        m = uniform(mesh, (0, 0, 1)); c = solveconduction!(w, m)
        jx = 1e-3 / (4h * 3h)
        @test (c.phi[1, 29, 2, 2] - c.phi[1, 31, 2, 2]) / 2h / jx ≈ ρperp rtol = 1e-8
        @test maximum(abs, c.j[2:3, 3:58, :, :]) < 1e-9 * jx
        m = uniform(mesh, (1, 1, 1) ./ sqrt(3)); c = solveconduction!(w, m)
        Ez = -(c.phi[1, 30, 2, 3] - c.phi[1, 30, 2, 1]) / 2h
        @test Ez / jx ≈ Δρ / 3 rtol = 1e-6                        # E = ρ j: the z row of ρ times x̂
    end

    @testset "Simulation: table and maps" begin
        mesh = Mesh((40, 8, 1), (h, h, h))
        rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5))
        defregion!(rp, 1, XRange(-Inf, -19h)); defregion!(rp, 2, XRange(19h, Inf))
        sim = Simulation(mesh, rp; demag = false, Bext = (0, 0.05, 0), verbose = false)
        setconduction!(sim; rho_par = ρpar, rho_perp = ρperp)
        addcontact!(sim, 1; current = 1e-3); addcontact!(sim, 2; potential = 0.0)
        setmag!(sim, UniformConfig(1, 0, 0))
        tableadd!(sim, q_resistance()); tableadd!(sim, q_contacts(2)); tableautosave!(sim, 1e-11)
        run!(sim, 1e-10)
        R = getindex.(sim.table.rows, 5)
        @test R[1] ≈ ρpar * 38h / (8h * h) rtol = 1e-6                  # m ∥ j at the start
        @test R[end] < R[1] - 0.2 * Δρ * 38h / (8h * h)                  # m turned towards y
        @test size(potentialmap(sim)) == (1, 40, 8, 1) && size(currentmap(sim)) == (3, 40, 8, 1)
    end
end
