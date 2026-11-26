using JuliaMag, Test, Random

@testset "conduction with AMR" begin
    ρpar, ρperp = 210e-9, 200e-9
    Δρ = ρpar - ρperp
    h = 2e-9
    # a strip of 100 × 10 cells; contacts: the first and the last column
    function strip(θ; nx = 200, ny = 10, contact1 = (; current = 1e-3), contact2 = (; potential = 0.0), mesh = Mesh((nx, ny, 1), (h, h, h)))
        rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.1, pol = 0.5))
        defregion!(rp, 1, XRange(-Inf, -(nx / 2 - 1) * h)); defregion!(rp, 2, XRange((nx / 2 - 1) * h, Inf))
        w = World(mesh, rp; demag = false)
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
