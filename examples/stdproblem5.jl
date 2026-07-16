# µMAG Standard Problem 5 — spin-transfer torque (Zhang-Li)
# M. Najafi et al., J. Appl. Phys. 105, 113914 (2009).
#
# 100 × 100 × 10 nm Permalloy, discretized 32 × 32 × 4. Relax a vortex, then
# apply an in-plane current J = (1e12, 0, 0) A/m² with polarization P = 1 and
# non-adiabaticity ξ = 0.05. The vortex core moves and settles at a displaced
# equilibrium. Reference (mumax3): ⟨m⟩(1 ns) = (-0.23480, -0.09454, 0.02296).
#
# Run:  julia --project=examples examples/stdproblem5.jl

using JuliaMag
using Printf

# The Zhang-Li torque is applied through a small integrator wrapper: the RHS is
# the LLG torque plus the STT term. We drive the built-in Integrator but override
# its RHS by adding the STT into the effective-field-derived torque via a custom
# world callback — simplest is to step our own RK from the package's pieces.
function llg_stt_rhs!(dm, m, world, mesh, mat, J)
    B = world._Bbuf
    effectivefield!(B, m, world)
    torque!(dm, m, B, mat.alpha)                 # LLG torque
    zhanglitorque!(dm, m, mesh, mat, J; add = true)   # + Zhang-Li STT
    return dm
end

# Fixed-step RK4 (the STT run is short and stiff-free once relaxed); keeps the
# example self-contained without wiring STT into the ODE integrator.
function rk4_stt!(m, world, mesh, mat, J, dt, nsteps)
    k1 = similar(m); k2 = similar(m); k3 = similar(m); k4 = similar(m); tmp = similar(m)
    for _ in 1:nsteps
        llg_stt_rhs!(k1, m, world, mesh, mat, J)
        @. tmp = m + 0.5dt*k1; normalize!(tmp); llg_stt_rhs!(k2, tmp, world, mesh, mat, J)
        @. tmp = m + 0.5dt*k2; normalize!(tmp); llg_stt_rhs!(k3, tmp, world, mesh, mat, J)
        @. tmp = m + dt*k3;    normalize!(tmp); llg_stt_rhs!(k4, tmp, world, mesh, mat, J)
        @. m = m + (dt/6)*(k1 + 2k2 + 2k3 + k4)
        normalize!(m)
    end
    return m
end

function main()
    mesh = Mesh((32, 32, 4), (100e-9/32, 100e-9/32, 10e-9/4))
    mat  = Material(Msat = 800e3, Aex = 13e-12, alpha = 0.1, pol = 1.0, xi = 0.05)

    println("Standard Problem 5 (Zhang-Li STT)")
    show(stdout, MIME"text/plain"(), mesh); println(); flush(stdout)

    # --- Relax the vortex ---------------------------------------------------
    world = World(mesh, mat; demag = true)
    m = setconfig(mesh, translate(VortexConfig(mesh; circ = 1, pol = 1), 0, 0, 0))
    println("\nRelaxing the vortex (energy minimizer)…"); flush(stdout)
    mn = Minimizer(world, m; stopdm = 1e-6)
    minimize!(mn; maxsteps = 20_000)
    m = mn.m
    @printf("  relaxed after %d steps, ⟨m⟩ = (%.4f, %.4f, %.4f)\n", mn.step, average(m)...)
    flush(stdout)

    # --- Apply the current and integrate 1 ns -------------------------------
    J = (1e12, 0.0, 0.0)
    dt = 5e-14
    nsteps = round(Int, 1e-9 / dt)
    println("\nApplying J = $J A/m², integrating 1 ns…"); flush(stdout)
    nchunk = 20
    per = nsteps ÷ nchunk
    for c in 1:nchunk
        rk4_stt!(m, world, mesh, mat, J, dt, per)
        @printf("  t = %.2f ns  ⟨m⟩ = (%.5f, %.5f, %.5f)\n", c/nchunk, average(m)...)
        flush(stdout)
    end

    mx, my, mz = average(m)
    ref = (-0.23479773, -0.09453578, 0.02296375)
    @printf("\nJuliaMag  ⟨m⟩(1 ns) = (%.5f, %.5f, %.5f)\n", mx, my, mz)
    @printf("mumax3    ⟨m⟩(1 ns) = (%.5f, %.5f, %.5f)\n", ref...)
    @printf("Δ = (%.5f, %.5f, %.5f)\n", mx-ref[1], my-ref[2], mz-ref[3])

    # Save a table of the final state's core position could go here; the scalar
    # comparison above is the standard-problem-5 validation quantity.
end

main()
