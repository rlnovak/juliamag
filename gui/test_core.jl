# Headless checks of the GUI's worker (gui/core.jl): no window, no Makie.
#
#   julia -t 2 --project=gui gui/test_core.jl
#
# 1. A run through the GUI ends exactly at the end time with the state of run! for the same
#    duration (publishing reads the state, it must not change it).
# 2. Pause holds the state, Stop ends the run, a field set during the run is applied.
# 3. A script is loaded, with a function of time it defines; :tsit5 and a fixed step run to the
#    exact end time; a field set during a relaxation is applied at its end.
# 4. The cost of publishing: the same run with and without the GUI's publications.
using Test
include(joinpath(@__DIR__, "core.jl"))

waitjob() = (j = job[]; j === nothing || wait(j); @test !istaskfailed(job[]))
adopt(sim) = (startjob(() -> _adopt!(sim, "test"), "adopt"); waitjob())

function disc(n)
    sim = make_sim(n, n, 1, 4.0, "Permalloy", 0.02, "Vortex", "Cylinder", 4.0n, false)
    relax!(sim)
    JuliaMag.setexternalfield!(sim, (5e-3, 0, 0))
    return sim
end

@testset "GUI worker" begin
    @test Threads.nthreads(:default) + Threads.nthreads(:interactive) >= 2

    @testset "against run!" begin
        ref = disc(32); gui = disc(32)
        @test gui.m == ref.m
        run!(ref, 0.3e-9)
        adopt(gui); want_energy[] = true
        @test job_run(0.3e-9)
        waitjob()
        @test gui.t == 0.3e-9
        println("GUI run against run!: max |Δm| = $(maximum(abs, gui.m .- ref.m)), steps $(gui.nsteps) vs $(ref.nsteps)")
        @test gui.m == ref.m                    # publishing only reads the state
        @test gui.nsteps == ref.nsteps
        s, mp, ep, _ = takeupdates!()
        @test s.phase == "idle" && s.t == gui.t
        @test all(s.m .≈ Tuple(average(gui)))
        @test length(mp) > 1 && issorted(first.(mp))
        @test length(ep) >= 1
        @test s.slice !== nothing && size(s.slice) == (3, 32, 32)
        @test s.torque ≈ q_maxtorque()(gui.world, gui.m, gui.t)[1] rtol = 1e-12
    end

    @testset "pause, field, stop" begin
        gui = sim_ref[]
        @test job_run(Inf)
        sleep(2.0)
        pauseflag[] = true
        sleep(1.0)
        t1 = gui.t; sleep(0.5)
        @test gui.t == t1
        @test takeupdates!()[1].phase == "paused"
        job_field((0.0, 20e-3, 0.0))            # applied between steps while paused
        sleep(0.3)
        @test gui.world.Bext[2] ≈ 20e-3
        pauseflag[] = false
        sleep(1.0)
        @test gui.t > t1
        stopflag[] = true
        waitjob()
        @test startswith(takeupdates!()[1].message, "Stopped")
    end

    @testset "script, solvers, field during relax" begin
        path = joinpath(mktempdir(), "my script.jl")
        write(path, """
            mesh = Mesh((16, 16, 1), (4e-9, 4e-9, 4e-9))
            sim = Simulation(mesh, material("Permalloy"); verbose = false)
            setmag!(sim, UniformConfig(1, 0, 0))
            pulse(t) = 0.01 * exp(-(t / 20e-12)^2)          # defined by the script
            addfield!(sim, fieldmap(mesh) do x, y, z; (0, 1, 0); end, pulse)
            """)
        u = "file:///" * replace(replace(path, "\\" => "/"), " " => "%20")
        @test replace(urlpath(u), "\\" => "/") == replace(path, "\\" => "/")
        @test job_load(path)
        waitjob()
        s = takeupdates!()[1]
        @test startswith(s.message, "Loaded") && sim_ref[] isa Simulation
        sim = sim_ref[]
        @test job_run(50e-12); waitjob()
        @test sim.t == 50e-12
        setsolver!(sim, :rk4; fixdt = 7e-14)
        @test job_run(1e-12); waitjob()               # not a multiple of the step
        @test sim.t ≈ 51e-12 rtol = 1e-14
        clearfields!(sim); setsolver!(sim, :tsit5)        # tsit5 refuses functions of time
        @test job_run(20e-12); waitjob()
        @test sim.t ≈ 71e-12 rtol = 1e-14
        @test startswith(takeupdates!()[1].message, "Done")
        setsolver!(sim, :rk45)
        # a field set during a relaxation is applied when it ends
        @test job_relax()
        job_field((0.0, 0.0, 30e-3))
        waitjob()
        @test sim.world.Bext[3] ≈ 30e-3
        @test takeupdates!()[1].Bext[3] ≈ 30e-3
    end

    @testset "cost of publishing" begin
        a = disc(64); b = disc(64)
        ta = @elapsed run!(a, 0.2e-9)
        adopt(b); rate[] = 20; erate[] = 4; want_energy[] = true
        tb = @elapsed (job_run(0.2e-9); waitjob())
        println("64x64 disc, 0.2 ns: run! $(round(ta, digits = 2)) s, GUI worker $(round(tb, digits = 2)) s, ",
                "ratio $(round(tb / ta, digits = 3))")
        @test a.m == b.m
    end
end
