using JuliaMag, Test, Random

erf_(x) = ccall((:erf, Base.libm_name), Float64, (Float64,), x)

@testset "field and current in space and time" begin
    mesh = Mesh((8, 6, 2), (3e-9, 3e-9, 3e-9))
    mat = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.1, pol = 0.5, xi = 0.05, lambda = 1.2, epsilonPrime = 0.02)

    @testset "maps" begin
        f = fieldmap(mesh, (x, y, z) -> (x, y, z))
        @test f[:, 1, 1, 1] ≈ collect(JuliaMag.cellcenter(mesh, 1, 1, 1))
        @test f[:, 8, 6, 2] ≈ [10.5e-9, 7.5e-9, 1.5e-9]
        @test fieldmap(mesh) do x, y, z; (x, y, z); end == f                  # the do-block form
        @test fieldmap(Float32, mesh) do x, y, z; (x, y, z); end == Float32.(f)
        s = fieldmap(mesh, XRange(0, Inf), (0, 0, 2))
        @test s[3, :, :, :] == [i > 4 ? 2.0 : 0.0 for i in 1:8, j in 1:6, k in 1:2]
        rp = RegionParams(mesh, mat); defregion!(rp, 3, YRange(0, Inf))
        r = regionmap(rp, 3, (1, 0, 0))
        @test r[1, :, :, :] == [j > 3 ? 1.0 : 0.0 for i in 1:8, j in 1:6, k in 1:2] && all(iszero, r[2:3, :, :, :])
        @test_throws ArgumentError regionmap(mat, 1, (1, 0, 0))
        @test_throws ArgumentError fieldmap(mesh, (x, y, z) -> (NaN, 0, 0))
        w = World(mesh, mat)
        @test_throws DimensionMismatch addfield!(w, zeros(3, 2, 2, 2))
        @test_throws ArgumentError addcurrent!(w, zeros(3, 8, 6, 2))     # setcurrent! first
    end

    @testset "a uniform map is the uniform field, bit for bit" begin
        Random.seed!(3); m0 = randommag!(Random.default_rng(), zeromag(Float64, mesh))
        a = World(mesh, mat; Bext = (0.01, -0.02, 0.05))
        b = World(mesh, mat); addfield!(b, fieldmap(mesh, (x, y, z) -> (0.01, -0.02, 0.05)))
        @test effectivefield!(similar(m0), m0, a) == effectivefield!(similar(m0), m0, b)
        ma = copy(m0); mb = copy(m0)
        advanceto!(TimeStepper(a, ma), 2e-12); advanceto!(TimeStepper(b, mb), 2e-12)
        @test ma == mb
        @test zeemanenergy(m0, b) ≈ zeemanenergy(m0, a) rtol = 1e-14
    end

    @testset "energies with a nonuniform field, regions and partial cells" begin
        rp = RegionParams(mesh, mat); setregion!(rp, 0; Msat = 0.0); setregion!(rp, 1; Msat = 6e5, Aex = 1e-11)
        setgeometry!(rp, Ellipse(22e-9, 16e-9); id = 1, edgesmooth = 3)
        w = World(mesh, rp; Bext = (0.02, 0, 0))
        g = fieldmap(mesh, (x, y, z) -> (sin(x / 5e-9), 0.3, cos(y / 4e-9)))
        addfield!(w, g, t -> 0.07 + 1e9 * t; t = 1e-11)
        @test w.Bterms[1].c ≈ 0.08
        Random.seed!(5); m = clearempty!(randommag!(Random.default_rng(), zeromag(Float64, mesh)), rp)
        Bapp = appliedfield!(similar(m), w)
        @test Bapp ≈ 0.08 .* g .+ reshape([0.02, 0, 0], 3, 1, 1, 1)
        V = cellvolume(mesh)
        Ez = -V * sum(rp.Msat[rp.regions.id[i, j, k]+1] * rp.fill[i, j, k] * sum(m[:, i, j, k] .* Bapp[:, i, j, k]) for i in 1:8, j in 1:6, k in 1:2)
        @test zeemanenergy(m, w) ≈ Ez rtol = 1e-13
        @test energy!(similar(m), m, w) ≈ totalenergy(m, w) rtol = 1e-12
        # the field is the gradient of the energy: E(m + εδ) − E(m − εδ) against −V Σ Ms δ·B
        w2 = World(mesh, RegionParams(mesh, mat); demag = false); addfield!(w2, g, 0.05)
        m2 = normalize!(randommag!(Random.default_rng(), zeromag(Float64, mesh)))
        δ = 1e-6 .* randn(size(m2)...)
        B = effectivefield!(similar(m2), m2, w2)
        dE = totalenergy(m2 .+ δ, w2) - totalenergy(m2 .- δ, w2)
        @test dE ≈ -2V * 8e5 * sum(δ .* B) rtol = 1e-6
    end

    @testset "time: a spin in a pulse, order of the solvers" begin
        one = Mesh((1, 1, 1), (3e-9, 3e-9, 3e-9))
        pulse(t) = 0.3 * exp(-((t - 1e-11) / 3e-12)^2)
        exact(t) = γLL * 0.3 * 3e-12 * sqrt(π) / 2 * (erf_((t - 1e-11) / 3e-12) - erf_(-1e-11 / 3e-12))
        for (s, p, dts) in ((:rk4, 4, (4e-13, 2e-13)), (:rk45, 5, (4e-13, 2e-13)))
            err = map(dts) do dt
                w = World(one, Material(Msat = 8e5, Aex = 0.0, alpha = 0.0); demag = false)
                addfield!(w, fieldmap(one, (x, y, z) -> (0, 0, 1)), pulse)
                m = uniform(one, (1, 0, 0)); advanceto!(TimeStepper(w, m; solver = s, fixdt = dt), 2e-11)
                abs(rem2pi(atan(m[2], m[1]) - exact(2e-11), RoundNearest))
            end
            # the phase is a quadrature of B(t), which these methods do to an order ≥ their own; stages at
            # wrong times would give order 1
            @test log2(err[1] / err[2]) > p - 0.3
        end
        # two uncoupled cells in different fields precess at their own frequencies
        two = Mesh((2, 1, 1), (3e-9, 3e-9, 3e-9))
        w = World(two, Material(Msat = 8e5, Aex = 0.0, alpha = 0.0); demag = false)
        addfield!(w, fieldmap(two, (x, y, z) -> (0, 0, x > 0 ? 0.2 : 0.1)))
        m = uniform(two, (1, 0, 0)); advanceto!(TimeStepper(w, m; solver = :rk45, maxerr = 1e-9), 3e-12)
        @test atan(m[2, 1, 1, 1], m[1, 1, 1, 1]) ≈ γLL * 0.1 * 3e-12 rtol = 1e-6
        @test atan(m[2, 2, 1, 1], m[1, 2, 1, 1]) ≈ γLL * 0.2 * 3e-12 rtol = 1e-6
        # a function of time is refused by the old solvers, a constant term is not
        sim = Simulation(two, Material(Msat = 8e5, Aex = 0.0, alpha = 0.5); demag = false, verbose = false)
        addfield!(sim, fieldmap(two, (x, y, z) -> (0, 0, 0.1)), 2.0)
        runthermal!(sim, 1e-13, 0.0)
        addfield!(sim, fieldmap(two, (x, y, z) -> (0, 0, 0.1)), t -> 1.0)
        @test_throws ArgumentError runthermal!(sim, 1e-13, 0.0)
        @test_throws ArgumentError Integrator(sim.world, sim.m)
        clearfields!(sim); @test isempty(sim.world.Bterms)
    end

    @testset "current per cell" begin
        Random.seed!(7); m = normalize!(randommag!(Random.default_rng(), zeromag(Float64, mesh)))
        J0 = (1e12, -3e11, 2e12)
        a = setcurrent!(World(mesh, mat; demag = false), J0; zhangli = true, slonczewski = true, fixedlayer = (0.3, 0, 1))
        b = setcurrent!(World(mesh, mat; demag = false), (0, 0, 0); zhangli = true, slonczewski = true, fixedlayer = (0.3, 0, 1))
        addcurrent!(b, fieldmap(mesh, (x, y, z) -> J0))
        @test llgrhs!(similar(m), m, b) ≈ llgrhs!(similar(m), m, a) rtol = 1e-14
        # the current of each cell, as mumax3: no torque where J is 0, the uniform torque where it is J0
        c = setcurrent!(World(mesh, mat; demag = false), (0, 0, 0); zhangli = true, slonczewski = true, fixedlayer = (0.3, 0, 1))
        addcurrent!(c, fieldmap(mesh, XRange(0, Inf), J0), t -> 2.0)
        τa = JuliaMag.applycurrent!(fill!(similar(m), 0), m, a)
        τc = JuliaMag.applycurrent!(fill!(similar(m), 0), m, c)
        @test all(iszero, τc[:, 1:4, :, :])
        @test τc[:, 6:7, :, :] ≈ 2 .* τa[:, 6:7, :, :] rtol = 1e-13
        # the parameters of each cell's region (as mumax3): region 1 with its own Msat, pol, xi, lambda, ε′, alpha
        rp = RegionParams(mesh, mat); setregion!(rp, 1; Msat = 4e5, pol = 0.8, xi = 0.2, lambda = 1.5, epsilonPrime = 0.05, alpha = 0.3)
        defregion!(rp, 1, YRange(0, Inf))
        mat1 = Material(Msat = 4e5, Aex = 1.3e-11, alpha = 0.3, pol = 0.8, xi = 0.2, lambda = 1.5, epsilonPrime = 0.05)
        for (Jm, uniformJ) in ((nothing, J0), (fieldmap(mesh, (x, y, z) -> J0), (0, 0, 0)))
            r = setcurrent!(World(mesh, rp; demag = false), uniformJ; zhangli = true, slonczewski = true, fixedlayer = (0.3, 0, 1))
            Jm === nothing || addcurrent!(r, Jm)
            τr = JuliaMag.applycurrent!(fill!(similar(m), 0), m, r)
            for (par, rows) in ((mat, 1:3), (mat1, 4:6))
                u = setcurrent!(World(mesh, par; demag = false), J0; zhangli = true, slonczewski = true, fixedlayer = (0.3, 0, 1))
                τu = JuliaMag.applycurrent!(fill!(similar(m), 0), m, u)
                @test τr[:, :, rows, :] ≈ τu[:, :, rows, :] rtol = 1e-13
            end
        end
        # explicit parameters stay uniform, and need one alpha
        e = setcurrent!(World(mesh, rp; demag = false), J0; params = mat)
        @test_throws ErrorException llgrhs!(similar(m), m, e)
        # solvers without spin torques refuse a current
        @test_throws ArgumentError Solver(a, copy(m))
        sim = Simulation(mesh, mat; verbose = false); setcurrent!(sim, (0, 0, 0)); addcurrent!(sim, fieldmap(mesh, XRange(0, Inf), J0), 0.5)
        @test_throws ArgumentError runthermal!(sim, 1e-13, 10.0)
        @test q_J()(sim.world, sim.m, 0.0) ≈ [J0...] ./ 4
        addfield!(sim, fieldmap(mesh, (x, y, z) -> (0, 0, 1)), t -> 1e9 * t); JuliaMag._drive!(sim.world, 2e-10)
        @test q_Bext()(sim.world, sim.m, 0.0) ≈ [0, 0, 0.2]
    end
end
