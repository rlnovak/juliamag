using LinearAlgebra: norm

# Analytic damped precession of a macrospin in a field B0 along z (LL form, as torque!):
# tan(θ/2) = tan(θ0/2)·exp(−α γ′ B0 t), φ = φ0 + γ′ B0 t, γ′ = γLL/(1+α²).
function _macrospin_exact(t, B0, α; θ0 = π / 2)
    γp = γLL / (1 + α^2)
    θ = 2 * atan(tan(θ0 / 2) * exp(-α * γp * B0 * t))
    φ = γp * B0 * t
    return (sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
end

_macro_world(α; B0 = 0.1) = World(Mesh((1, 1, 1), (5e-9, 5e-9, 5e-9)),
                                  Material(Msat = 8e5, Aex = 1.3e-11, alpha = α);
                                  demag = false, Bext = (0, 0, B0))

@testset "mumax3 time steppers and the driver layer" begin

    @testset "fixed-step order of every solver (damped macrospin)" begin
        α, B0, tend = 0.1, 0.1, 200e-12
        ex = _macrospin_exact(tend, B0, α)
        # expected global order: Euler 1, Heun 2, RK23 3, RK4 4, RK45 5
        for (s, p) in ((:euler, 1), (:heun, 2), (:rk23, 3), (:rk4, 4), (:rk45, 5))
            errs = Float64[]
            for dt in (4e-12, 2e-12)      # large enough to stay above round-off for RK45
                w = _macro_world(α; B0 = B0)
                m = uniform(w.mesh, (1, 0, 0))
                ts = TimeStepper(w, m; solver = s, fixdt = dt)
                advanceto!(ts, tend)
                @test ts.t ≈ tend
                push!(errs, norm(m[:, 1, 1, 1] .- collect(ex)))
            end
            order = log2(errs[1] / errs[2])
            # Each stage renormalizes m (as in mumax3), which keeps the scheme's order.
            @test order > p - 0.35
        end
    end

    @testset "adaptive RK45 follows the analytic solution within MaxErr" begin
        α, B0 = 0.05, 0.2
        for maxerr in (1e-5, 1e-7)
            w = _macro_world(α; B0 = B0)
            m = uniform(w.mesh, (1, 0, 0))
            ts = TimeStepper(w, m; solver = :rk45, maxerr = maxerr)
            advanceto!(ts, 200e-12)
            err = norm(m[:, 1, 1, 1] .- collect(_macrospin_exact(200e-12, B0, α)))
            @test ts.t == 200e-12                      # alarm: the last step lands on tend
            @test err < 50 * maxerr
            @test ts.nundone < ts.nsteps
            # FSAL: 6 evaluations per step attempt (accepted or rejected) after the first
            @test ts.nevals == 1 + 6 * ts.nsteps + 6 * ts.nundone
        end
    end

    @testset "adaptive step control reproduces mumax3's adaptDt" begin
        w = _macro_world(0.0)
        ts = TimeStepper(w, uniform(w.mesh, (1, 0, 0)); solver = :rk45, maxerr = 1e-5)
        ts.dt = 1e-13
        JuliaMag._adaptdt!(ts, 10.0)         # corr·headroom capped at 2
        @test ts.dt ≈ 2e-13
        JuliaMag._adaptdt!(ts, 0.1)          # floored at 1/2
        @test ts.dt ≈ 1e-13
        JuliaMag._adaptdt!(ts, 1.0)          # 0.8 headroom
        @test ts.dt ≈ 0.8e-13
        ts.maxdt = 5e-14; JuliaMag._adaptdt!(ts, 1.0)
        @test ts.dt == 5e-14
        @test_throws ArgumentError TimeStepper(w, uniform(w.mesh, (1, 0, 0)); solver = :rk9)
        @test_throws ArgumentError TimeStepper(w, uniform(w.mesh, (1, 0, 0)); temperature = 300)
    end

    @testset "run! with each solver agrees on standard-problem-4-like dynamics" begin
        mesh = Mesh((40, 10, 1), (3.125e-9, 3.125e-9, 3e-9))
        mat = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02)
        ref = nothing
        for s in (:tsit5, :rk45, :rk23, :rk4)
            sim = Simulation(mesh, mat; demag = true, Bext = (-24.6e-3, 4.3e-3, 0), verbose = false)
            setmag!(sim, UniformConfig(1, 0.2, 0.1))
            if s === :rk4
                run!(sim, 50e-12; solver = s, fixdt = 5e-14)
            else
                run!(sim, 50e-12; solver = s, maxerr = 1e-7, reltol = 1e-8, abstol = 1e-9)
            end
            @test sim.t ≈ 50e-12
            a = collect(average(sim))
            ref === nothing ? (ref = a) : (@test maximum(abs, a .- ref) < 2e-4)
        end
    end

    @testset "repeated run! calls advance the time (also with Tsit5)" begin
        mesh = Mesh((8, 8, 1), (5e-9, 5e-9, 5e-9))
        for s in (:tsit5, :rk45)
            sim = Simulation(mesh, material("Permalloy"); demag = false, Bext = (0.05, 0, 0),
                             verbose = false)
            setmag!(sim, UniformConfig(1, 1, 0))
            tableautosave!(sim, 5e-12)
            run!(sim, 20e-12; solver = s); run!(sim, 20e-12; solver = s)
            @test sim.t ≈ 40e-12
            ts = getindex.(sim.table.rows, 1)
            @test issorted(ts)
            @test ts[end] ≈ 40e-12 rtol = 1e-6
        end
    end

    @testset "table autosave follows mumax3's rule; exact times with a fixed step" begin
        mesh = Mesh((8, 8, 1), (5e-9, 5e-9, 5e-9))
        sim = Simulation(mesh, material("Permalloy"); demag = false, Bext = (0, 0, 0.1),
                         verbose = false)
        setmag!(sim, UniformConfig(1, 0, 0))
        tableadd!(sim, q_maxtorque())
        tableautosave!(sim, 1e-12)
        setsolver!(sim, :rk4; fixdt = 1e-13)
        run!(sim, 10e-12)
        t = getindex.(sim.table.rows, 1)
        @test length(t) == 11                       # t = 0, 1, …, 10 ps
        @test t ≈ collect(0:10) .* 1e-12
        # A second run continues the schedule without repeating the last row.
        run!(sim, 2e-12)
        @test length(sim.table.rows) == 13
        # maxTorque column in T: max |dm/dt| / γLL of the full RHS
        dm = similar(sim.m); llgrhs!(dm, sim.m, sim.world)
        tablesave!(sim)
        @test sim.table.rows[end][5] ≈ maxtorque(dm) / γLL
        @test tableheader(sim.table)[5] == "maxTorque (T)"
    end

    @testset "spin current through run! = hand-written fixed-step drivers" begin
        # Zhang-Li: run! with setcurrent! and RK4 vs runcurrent! (independent RK4 code).
        mesh = Mesh((40, 20, 1), (2e-9, 2e-9, 1e-9))
        mat = Material(Msat = 5.8e5, Aex = 1.5e-11, alpha = 0.3, Ku = 8e5, anisU = (0, 0, 1),
                       Dind = 3.0e-3, pol = 1.0, xi = 0.2)
        J = (5e12, 0.0, 0.0)
        a = Simulation(mesh, mat; demag = true, verbose = false)
        setmag!(a, NeelSkyrmionConfig(mesh; charge = 1, pol = -1)); relax!(a; stopdm = 1e-5)
        b = Simulation(mesh, mat; demag = true, verbose = false); setmag!(b, a.m)
        runcurrent!(a, J, 20e-12; every = 20e-12, dt = 5e-14)
        setcurrent!(b, J; zhangli = true)
        run!(b, 20e-12; solver = :rk4, fixdt = 5e-14)
        @test maximum(abs, a.m .- b.m) < 1e-10
        @test b.t ≈ 20e-12

        # Slonczewski on a disc with empty cells: run! vs an explicit RK4 loop.
        dmesh = Mesh((16, 16, 2), (4e-9, 4e-9, 4e-9))
        pm = Material(Msat = 8e5, Aex = 1.08e-11, alpha = 0.01, pol = 0.37, lambda = 1.0,
                      epsilonPrime = 0.01)
        rp = RegionParams(dmesh, pm); setregion!(rp, 0; Msat = 0.0)
        setgeometry!(rp, Cylinder(64e-9, 16e-9); id = 1, edgesmooth = 4)
        s = Simulation(dmesh, rp; demag = true, verbose = false)
        setmag!(s, VortexConfig(dmesh; circ = 1, pol = 1)); relax!(s; stopdm = 1e-5)
        m = copy(s.m); w = World(dmesh, rp; demag = true)
        Jz = 3e12; p = (cos(0.35), sin(0.35), 0.0); th = 8e-9
        rhs!(dm, mm) = (llgrhs!(dm, mm, w);
                        slonczewskitorque!(dm, mm, dmesh, pm, Jz, p, th; add = true); dm)
        k1, k2, k3, k4, m0 = similar(m), similar(m), similar(m), similar(m), similar(m)
        dt = 5e-14
        for _ in 1:200
            copyto!(m0, m)
            rhs!(k1, m)
            @. m = m + (1/2) * dt * k1;  normalize!(m); rhs!(k2, m)
            @. m = m0 + (1/2) * dt * k2; normalize!(m); rhs!(k3, m)
            @. m = m0 + dt * k3;         normalize!(m); rhs!(k4, m)
            @. m = m0 + (1/6) * dt * k1 + (1/3) * dt * k2 + (1/3) * dt * k3 + (1/6) * dt * k4
            normalize!(m)
        end
        setcurrent!(s, (0, 0, Jz); zhangli = false, slonczewski = true, fixedlayer = p,
                    thickness = th)
        @test s.world.current.params.Msat == 8e5          # STT parameters of region 1
        run!(s, 200 * dt; solver = :rk4, fixdt = dt)
        @test maximum(abs, s.m .- m) < 1e-12
        setcurrent!(s, nothing)
        @test s.world.current === nothing
    end

    @testset "finite temperature needs a fixed step; runs with Heun" begin
        mesh = Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9))
        sim = Simulation(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5); demag = false,
                         verbose = false)
        setmag!(sim, UniformConfig(0, 0, 1))
        settemperature!(sim, 300)
        @test_throws ArgumentError run!(sim, 1e-12)              # adaptive: refused
        @test_throws ArgumentError run!(sim, 1e-12; solver = :tsit5)
        run!(sim, 1e-12; solver = :heun, fixdt = 1e-14)
        @test sim.t ≈ 1e-12
        @test all(isfinite, sim.m)
        @test average(sim)[3] < 1                                  # thermal kicks
    end

    @testset "OVF save / autosave numbering and round trip" begin
        dir = mktempdir()
        mesh = Mesh((8, 4, 1), (5e-9, 5e-9, 5e-9))
        sim = Simulation(mesh, material("Permalloy"); demag = false, Bext = (0, 0, 0.1),
                         outdir = dir, verbose = false)
        setmag!(sim, UniformConfig(1, 0, 0))
        @test_throws ArgumentError autosave!(Simulation(mesh, material("Permalloy");
                                                        verbose = false), 1e-12)
        autosave!(sim, 2e-12)
        tableautosave!(sim, 1e-12)
        run!(sim, 6e-12; solver = :rk4, fixdt = 1e-13)
        files = sort(filter(f -> endswith(f, ".ovf"), readdir(dir)))
        @test files == ["m000000.ovf", "m000001.ovf", "m000002.ovf", "m000003.ovf"]
        m3, _ = loadovf(joinpath(dir, "m000003.ovf"))
        @test maximum(abs, m3 .- sim.m) < 1e-12
        p = save!(sim; name = "m")
        @test basename(p) == "m000004.ovf"
        # the table was streamed to dir/table.txt as rows were saved
        lines = readlines(joinpath(dir, "table.txt"))
        @test startswith(lines[1], "# t (s)")
        @test length(lines) == 1 + length(sim.table.rows) == 1 + 7
    end

    @testset "run log: wall-clock times, device, solver" begin
        io = IOBuffer()
        mesh = Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9))
        sim = Simulation(mesh, material("Permalloy"); demag = false, Bext = (0, 0, 0.1), log = io)
        setmag!(sim, UniformConfig(1, 0, 0))
        run!(sim, 1e-12)
        s = String(take!(io))
        @test occursin(r"\[JuliaMag\] \d{4}-\d\d-\d\d \d\d:\d\d:\d\d  run 1e-12 s", s)
        @test occursin("device: CPU (", s)
        @test occursin("solver rk45 (maxerr 1.0e-5)", s)
        @test occursin(r"done at t = 1e-12 s: \d+ steps \(\d+ undone\), \d+ evaluations, wall", s)
    end

    @testset "Float32 run keeps a Float64 clock" begin
        mesh = Mesh((8, 8, 1), (5e-9, 5e-9, 5e-9))
        sim = Simulation(mesh, Material{Float32}(material("Permalloy")); demag = true,
                         Bext = (0, 0, 0.1), verbose = false)
        setmag!(sim, UniformConfig(1, 0, 0))
        run!(sim, 30e-12; solver = :rk4, fixdt = 1e-13)
        @test sim.t isa Float64
        @test sim.t ≈ 30e-12 rtol = 1e-12                        # 300 steps, no Float32 drift
        @test eltype(sim.m) == Float32
    end

    @testset "OVF autosave without a table autosave (mumax3 and Tsit5 paths)" begin
        mesh = Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9))
        for s in (:rk45, :tsit5)
            dir = mktempdir()
            sim = Simulation(mesh, material("Permalloy"); demag = false, Bext = (0, 0, 0.1),
                             outdir = dir, verbose = false)
            setmag!(sim, UniformConfig(1, 0, 0))
            autosave!(sim, 2e-12)
            run!(sim, 6e-12; solver = s)
            files = sort(filter(f -> endswith(f, ".ovf"), readdir(dir)))
            @test files == ["m000000.ovf", "m000001.ovf", "m000002.ovf", "m000003.ovf"]
            @test length(sim.table.rows) == 2                # legacy start and end rows
        end
    end

    @testset "a rerun into the same outdir starts a fresh table; no columns after rows" begin
        dir = mktempdir()
        mesh = Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9))
        for rep in 1:2
            sim = Simulation(mesh, material("Permalloy"); demag = false, Bext = (0, 0, 0.1),
                             outdir = dir, verbose = false)
            setmag!(sim, UniformConfig(1, 0, 0))
            tableautosave!(sim, 1e-12)
            run!(sim, 3e-12; solver = :rk4, fixdt = 1e-13)
            lines = readlines(joinpath(dir, "table.txt"))
            @test count(startswith("#"), lines) == 1
            @test length(lines) == 1 + 4
            @test_throws ArgumentError tableadd!(sim, q_energy())
        end
    end

    @testset "maxtorque propagates NaN; a diverging step is never accepted" begin
        a = zeros(3, 4, 4, 1); a[1, 2, 2, 1] = NaN; a[2, 3, 3, 1] = 5.0
        @test isnan(maxtorque(a))
        w = _macro_world(0.0)
        m = uniform(w.mesh, (1, 0, 0))
        ts = TimeStepper(w, m; solver = :rk45)
        m[1, 1, 1, 1] = NaN
        @test_throws ErrorException advanceto!(ts, 1e-12)   # dt shrinks to the floor
        @test ts.nsteps == 0
    end

    @testset "error accumulator: running maxima, NaN sticky" begin
        a = randn(3, 5, 4, 2); b = randn(3, 5, 4, 2)
        n2(x) = maximum(sum(abs2, x; dims = 1))
        acc = JuliaMag._newacc(a)
        JuliaMag._acc_max!(acc, a, nothing, 2.0)
        @test acc[1] ≈ n2(2 .* a) && acc[2] == acc[1]
        JuliaMag._acc_reset!(acc)
        JuliaMag._acc_max!(acc, a, b, 0.5)
        @test acc[1] ≈ n2(0.5 .* (a .- b))
        @test acc[2] ≈ max(n2(2 .* a), n2(0.5 .* (a .- b)))             # peak kept
        a[2, 3, 2, 1] = NaN
        JuliaMag._acc_reset!(acc); JuliaMag._acc_max!(acc, a, nothing, 1.0)
        @test isnan(acc[1]) && isnan(acc[2])
        JuliaMag._acc_reset!(acc); JuliaMag._acc_max!(acc, b, nothing, 1.0)
        @test !isnan(acc[1]) && isnan(acc[2])                            # peak stays NaN
    end

    @testset "fixed step: errors kept on the device, read on demand" begin
        mesh = Mesh((6, 6, 1), (5e-9, 5e-9, 5e-9))
        w = World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02); demag = false,
                  Bext = (0.0, 0.05, 0.0))
        m = uniform(mesh, (1, 0, 0)); m[3, 2, 3, 1] = 0.3; normalize!(m)
        for (solver, i, j) in ((:rk4, 1, 4), (:heun, 1, 2))
            ts = TimeStepper(w, copy(m); solver, fixdt = 1e-13)
            peak = 0.0
            for _ in 1:20
                step!(ts)
                e = 1e-13 * maximum(sqrt.(sum(abs2, ts.k[i] .- ts.k[j]; dims = 1)))
                peak = max(peak, e)
                @test JuliaMag.lasterr(ts) ≈ e rtol = 1e-12
            end
            @test JuliaMag.peakerr(ts) ≈ peak rtol = 1e-12
            @test ts.nundone == 0
        end
        # LastTorque on demand: for RK45 (FSAL) it is the torque at the new state
        ts = TimeStepper(w, copy(m); solver = :rk45)
        @test isnan(JuliaMag.lasttorque(ts))
        advanceto!(ts, 5e-12)
        dm = similar(ts.m); llgrhs!(dm, ts.m, w)
        @test JuliaMag.lasttorque(ts) ≈ maxtorque(dm) rtol = 1e-12
    end

    @testset "fixed step: a diverging state stops the run (no endless retries)" begin
        w = _macro_world(0.0)
        for solver in (:heun, :rk4, :rk45, :euler)
            m = uniform(w.mesh, (1, 0, 0))
            ts = TimeStepper(w, m; solver, fixdt = 1e-13)
            m[1, 1, 1, 1] = NaN
            err = try advanceto!(ts, 1e-12); nothing catch e; e end
            @test err isa ErrorException && occursin("diverged", err.msg)
            @test ts.nsteps == 10                       # accepted, as in mumax3, then stopped
            # the stepper is not poisoned: a fresh state runs on it normally
            ts.m .= uniform(w.mesh, (1, 0, 0)); ts.fsal = false
            advanceto!(ts, 2e-12)
            @test isfinite(peakerr(ts)) && lasterr(ts) >= 0
        end
        # mode switch on one stepper: the fixed-step peak is not polluted by the
        # unscaled adaptive values
        ts = TimeStepper(w, uniform(w.mesh, (0.6, 0.8, 0)); solver = :rk4)
        advanceto!(ts, 1e-12)
        ts.fixdt = ts.dt = 1e-14
        advanceto!(ts, 1.2e-12)
        @test peakerr(ts) < 1e-3

        # in a Simulation the NaN is caught at the first output, and sim.t is kept
        sim = Simulation(Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9)), material("Permalloy");
                         demag = false, Bext = (0, 0, 0.1), verbose = false)
        setmag!(sim, UniformConfig(1, 0, 0))
        tableautosave!(sim, 1e-12)
        sim.m[2, 2, 2, 1] = NaN
        @test_throws ErrorException run!(sim, 10e-12; solver = :heun, fixdt = 1e-13)
        @test sim.t ≈ 1e-12 && sim.nsteps == 10
    end

    @testset "run!(sim, t; abstol, reltol) keeps the Tsit5 meaning" begin
        io = IOBuffer()
        sim = Simulation(Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9)), material("Permalloy");
                         demag = false, Bext = (0, 0, 0.1), log = io)
        setmag!(sim, UniformConfig(1, 0, 0))
        run!(sim, 1e-12; abstol = 1e-7, reltol = 1e-6)
        @test occursin("solver tsit5 (abstol 1.0e-7, reltol 1.0e-6)", String(take!(io)))
        @test sim.solver === :rk45                             # the override was for one run
    end
end
