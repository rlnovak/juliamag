using JuliaMag, Test, Random

w2(mesh) = World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5); demag = false)
norm3(m) = sqrt.(sum(abs2, m; dims = 1))
_withpol(rp) = (r = deepcopy(rp); fill!(r.pol, 0.5); r)

@testset "driver: damping per region, drive, relaxation" begin
    @testset "damping per region" begin
        mesh = Mesh((2, 1, 1), (3e-9, 3e-9, 3e-9))
        rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 0.0, alpha = 0.1)); setregion!(rp, 1; alpha = 0.5)
        defregioncell!(rp, 1, 2, 1, 1)
        w = World(mesh, rp; demag = false, Bext = (0, 0, 0.5))
        m = uniform(mesh, (1, 0, 0)); ts = TimeStepper(w, m; solver = :rk4, fixdt = 1e-13); advanceto!(ts, 1e-11)
        for (i, a) in ((1, 0.1), (2, 0.5))
            w1 = World(Mesh((1, 1, 1), mesh.cellsize), Material(Msat = 8e5, Aex = 0.0, alpha = a); demag = false, Bext = (0, 0, 0.5))
            m1 = uniform(w1.mesh, (1, 0, 0)); advanceto!(TimeStepper(w1, m1; solver = :rk4, fixdt = 1e-13), 1e-11)
            @test m[:, i, 1, 1] ≈ m1[:, 1, 1, 1] rtol = 1e-14
        end
        @test JuliaMag.uniformdamping(rp) === nothing
        @test_throws ErrorException JuliaMag.damping(rp)
        # a region that is not there, or has no moment, does not count
        r2 = RegionParams(mesh, Material(Msat = 8e5, Aex = 0.0, alpha = 0.1)); setregion!(r2, 5; alpha = 0.7)
        @test JuliaMag.uniformdamping(r2) == 0.1
        setregion!(r2, 0; Msat = 0.0); setregion!(r2, 1; alpha = 0.3, Msat = 8e5); defregioncell!(r2, 1, 2, 1, 1)
        @test JuliaMag.uniformdamping(r2) == 0.3 == JuliaMag.damping(r2)     # the magnet, not the empty region 0
        # a current takes the alpha of each cell (as mumax3); with explicit parameters, alpha must be uniform
        wc = setcurrent!(World(mesh, _withpol(rp); demag = false), (1e12, 0, 0))
        @test all(isfinite, llgrhs!(similar(m), m, wc))
        @test_throws ErrorException llgrhs!(similar(m), m, setcurrent!(World(mesh, _withpol(rp); demag = false), (1e12, 0, 0); params = Material(Msat = 8e5, Aex = 0.0, alpha = 0.1, pol = 0.5)))
    end
    @testset "drive: order of the solvers" begin
        mesh = Mesh((1, 1, 1), (3e-9, 3e-9, 3e-9))
        ϕ = γLL * 0.3 * sin(2π * 2e10 * 2e-11) / (2π * 2e10)
        for (s, p, dts) in ((:heun, 2, (4e-14, 2e-14)), (:rk23, 3, (2e-13, 1e-13)), (:rk4, 4, (4e-13, 2e-13)), (:rk45, 5, (4e-13, 2e-13)))
            err = map(dts) do dt
                w = World(mesh, Material(Msat = 8e5, Aex = 0.0, alpha = 0.0); demag = false)
                setdrive!(w, (w, t) -> setexternalfield!(w, (0, 0, 0.3 * cos(2π * 2e10 * t))))
                m = uniform(mesh, (1, 0, 0)); ts = TimeStepper(w, m; solver = s, fixdt = dt); advanceto!(ts, 2e-11)
                abs(rem2pi(atan(m[2], m[1]) - ϕ, RoundNearest))
            end
            @test abs(log2(err[1] / err[2]) - p) < 0.35
        end
        sim = Simulation(mesh, Material(Msat = 8e5, Aex = 0.0, alpha = 0.1); demag = false, verbose = false)
        setdrive!(sim, (w, t) -> nothing)
        @test_throws ArgumentError run!(sim, 1e-12; solver = :tsit5)
        @test_throws ArgumentError runthermal!(sim, 1e-13, 10.0)
        @test_throws ArgumentError Integrator(sim.world, sim.m)
        setdrive!(sim, nothing); setmag!(sim, UniformConfig(1, 0, 0)); setexternalfield!(sim, (0, 0.1, 0))
        run!(sim, 1e-14); setsolver!(sim; precess = false); m0 = copy(sim.m); run!(sim, 1e-13)
        @test abs(sim.m[3] - m0[3]) < 1e-6 && sim.m[2] > m0[2] + 1e-4
        @test_throws ArgumentError run!(sim, 1e-12; solver = :tsit5)
    end
    @testset "Minimizer(mumax = true)" begin
        # mumax3's test uniaxialanisotropy-minimize: from (-1,-2,-3) it ends in +y (a Barzilai–Borwein
        # step of negative sign), and the minimum itself is the same
        mesh = Mesh((64, 64, 1), (4e-9, 4e-9, 2e-9))
        w = World(mesh, Material(Msat = 1.1e6, Aex = 1.3e-11, alpha = 1.0, Ku = -0.5e6, anisU = (1, 0, 0)))
        for (kw, y) in (((; mumax = true), 1), ((;), -1))
            m = uniform(mesh, (-1, -2, -3) ./ sqrt(14)); mn = Minimizer(w, m; kw...); minimize!(mn)
            @test average(m)[2] ≈ y atol = 1e-3
            @test maximum(abs.(norm3(m) .- 1)) < 1e-14
        end
        # with a Slonczewski current the state differs from the one without
        mk(J) = setcurrent!(World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5, pol = 0.5, lambda = 1.0)), J; zhangli = false, slonczewski = true, fixedlayer = (0, 0.3, 1))
        mesh = Mesh((16, 16, 1), (4e-9, 4e-9, 2e-9))
        a = uniform(mesh, (1, 0.2, 0) ./ sqrt(1.04)); b = copy(a)
        wj = mk((0, 0, 5e10)); minimize!(Minimizer(wj, a; mumax = true)); minimize!(Minimizer(mk((0, 0, 0)), b; mumax = true))
        @test 1e-3 < maximum(abs.(a .- b)) < 0.5
        τ = JuliaMag.lltorque!(similar(a), a, effectivefield!(similar(a), a, wj), wj.material; precess = false)
        @test maxtorque(τ) / γLL > 1e-3                       # balanced by the spin torque
        @test maxtorque(JuliaMag.applycurrent!(τ, a, wj)) / γLL < 1e-5
        m = uniform(mesh, (1, 0, 0)); m[:, 1, 1, 1] .= NaN
        @test_throws ErrorException minimize!(Minimizer(w2(mesh), m); maxsteps = 20)
    end
    @testset "relaxllg!" begin
        mesh = Mesh((16, 16, 1), (4e-9, 4e-9, 4e-9))
        mk() = Simulation(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02); verbose = false)
        sim = mk(); setmag!(sim, VortexConfig(mesh)); relaxllg!(sim)
        s2 = mk(); setmag!(s2, VortexConfig(mesh)); relax!(s2; stopdm = 1e-9)
        @test maximum(abs.(sim.m .- s2.m)) < 1e-3
        @test sim.t == 0
        @test totalenergy(sim.m, sim.world) ≈ totalenergy(s2.m, s2.world) rtol = 1e-8
        setmag!(sim, VortexConfig(mesh)); relaxllg!(sim; threshold = 1e-3)
        dm = similar(sim.m); JuliaMag.lltorque!(dm, sim.m, effectivefield!(similar(sim.m), sim.m, sim.world), sim.world.material; precess = false)
        @test maxtorque(dm) / γLL <= 1e-3
    end
    @testset "steps!, runwhile!, counters, saved fields" begin
        mesh = Mesh((8, 8, 1), (4e-9, 4e-9, 4e-9)); dir = mktempdir()
        mk() = (s = Simulation(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.1); verbose = false, outdir = dir); setmag!(s, VortexConfig(mesh)); s)
        # steps taken one by one are those of a run
        a = mk(); run!(a, 3e-13); b = mk(); steps!(b, a.nsteps - 1)
        @test b.nsteps == a.nsteps - 1 && b.t < a.t
        c = mk(); runwhile!(c, s -> s.t < 2e-13)
        @test c.t >= 2e-13 && c.nsteps > 2
        d = mk(); steps!(d, c.nsteps)
        @test d.m == c.m && d.t == c.t
        # the counters are current during a run (a condition on them ends it)
        e = mk(); runwhile!(e, s -> s.nevals < 50)
        @test 50 <= e.nevals < 60
        e = mk(); setsolver!(e, :rk45; fixdt = 1e-13); runwhile!(e, s -> s.peakerr < 1e-12 && s.nsteps < 50)
        @test e.nsteps == 1 && e.peakerr > 1e-12
        seen = Float64[]
        f = mk(); tableadd!(f, Quantity("n", "", 1, (w, m, t) -> [Float64(f.nevals)])); tableautosave!(f, 5e-14); run!(f, 3e-13)
        n = getindex.(f.table.rows, 5)
        @test issorted(n) && length(unique(n)) > 2
        # a field other than m, and a scalar
        B(s) = (x = similar(s.m); JuliaMag._drive!(s.world, s.t); effectivefield!(x, s.m, s.world))
        setdrive!(f, (w, t) -> setexternalfield!(w, (0, 0, t)))
        p = save!(f; name = "B", field = B)
        @test loadovf(p)[1] ≈ B(f)
        q = joinpath(dir, "s.ovf"); saveovf(q, f.m[3:3, :, :, :], mesh)
        @test occursin("valuedim: 1", read(q, String))
        # the drive at the times of the table rows
        g = mk(); setdrive!(g, (w, t) -> setexternalfield!(w, (0, 0, 1e10 * t))); tableadd!(g, q_Bext()); tableautosave!(g, 1e-13); run!(g, 1e-12)
        @test all(r -> r[7] ≈ 1e10 * r[1], g.table.rows) && length(g.table.rows) > 3
    end
end
