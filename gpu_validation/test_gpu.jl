# GPU vs CPU agreement of every GPU method of the CUDA extension, in Float64 and
# Float32. The CPU methods (src/) are the reference; each GPU kernel mirrors the
# arithmetic of its CPU loop, so the two must agree to rounding: ~1e-15 relative in
# Float64 (the FFT libraries differ, FFTW vs cuFFT), ~1e-6 in Float32.
#
# Cases: scalar Material (open and periodic boundaries, Nz = 1 and Nz > 1, with
# anisotropy, interfacial or bulk DMI, Zeeman), and RegionParams (a bilayer with an
# exchange/Msat/Ku/DMI interface, and an edge-smoothed disc in an empty
# background). Every term is checked with add = false and add = true, plus the
# effective field, the LLG / Zhang-Li / Slonczewski torques, normalize!, the
# minimizer helpers, a short relax and a short adaptive run.
#
# Run (labmac04):  julia --project=$HOME/juliamag gpu_validation/test_gpu.jl

using JuliaMag, CUDA, Random, Printf
using JuliaMag: _demag!, _damping_torque!, _cayley_step!, _bb_sums

CUDA.functional() || error("no CUDA GPU")
println("device: ", CUDA.name(CUDA.device()), "   CUDA.jl ", pkgversion(CUDA))
# the batched demag transforms must run as one cuFFT call over the 3 components
let mesh = Mesh((64, 64, 8), (2e-9, 2e-9, 2e-9)), p = togpu(World(mesh, Material(Msat = 8e5, Aex = 1e-11, alpha = 0.01)).demagplan)
    println("cuFFT batch: plan_max_dims((1,2,3), size(mhat)) = ",
            CUDA.CUFFT.plan_max_dims((1, 2, 3), size(p.mhat)), " (4 = one batched call)")
end

const RESULTS = Tuple{String,Float64,Float64,Bool}[]   # (name, err, tol, pass)

relerr(a, b) = (s = maximum(abs, b); s == 0 ? maximum(abs, a) : maximum(abs.(a .- b)) / s)

function check(name, gpu, cpu, tol)
    e = relerr(Array(gpu), cpu)
    ok = e <= tol
    push!(RESULTS, (name, e, tol, ok))
    @printf("  %-58s %9.2e  (tol %.0e)  %s\n", name, e, tol, ok ? "ok" : "FAIL")
    return ok
end

tolfor(T) = T == Float64 ? 1e-12 : 2e-5

function mats(T)
    base = (Msat = T(8e5), Aex = T(1.3e-11), alpha = T(0.02), pol = T(0.5), xi = T(0.05),
            lambda = T(1.3), epsilonPrime = T(0.02))
    mI = Material(; base..., Ku = T(3e5), anisU = (0.3, 0.2, 1.0), Dind = T(2.5e-3))
    mB = Material(; base..., Ku = T(1e5), anisU = (1.0, 0.0, 0.2), Dbulk = T(1.5e-3))
    return mI, mB
end

function bilayer(T)
    mesh = Mesh((20, 16, 4), (3e-9, 3e-9, 2e-9))
    mI, _ = mats(T)
    rp = RegionParams(mesh, mI)
    defregion!(rp, 1, Layers(mesh, 3, 5))
    setregion!(rp, 1; Msat = 1.2e6, Aex = 2.2e-11, Ku = 5e5, anisU = (0, 1, 1), Dind = -1e-3)
    return mesh, rp
end

# Region bulk DMI with periodic x (region exchange and DMI across the wrap).
function bilayer_bulk(T)
    mesh = Mesh((18, 12, 4), (2e-9, 2e-9, 2e-9); pbc = (1, 0, 0))
    _, mB = mats(T)
    rp = RegionParams(mesh, mB)
    defregion!(rp, 1, Layers(mesh, 1, 3))
    setregion!(rp, 1; Msat = 6e5, Aex = 0.9e-11, Dbulk = 2.5e-3, Ku = 3e4)
    return mesh, rp
end

function disc(T)
    mesh = Mesh((24, 24, 3), (2e-9, 2e-9, 2e-9))
    mI, _ = mats(T)
    rp = RegionParams(mesh, mI)
    setregion!(rp, 0; Msat = 0.0)
    setgeometry!(rp, Cylinder(40e-9, 6e-9); id = 1, edgesmooth = 4)
    setregion!(rp, 1; Ku = 2e5, Dind = 1.5e-3)
    return mesh, rp
end

function field_terms(label, mesh, params, T; Bext = (0.01, -0.02, 0.03))
    tol = tolfor(T)
    Random.seed!(1)
    m = randommag!(zeromag(T, mesh)); clearempty!(m, params)
    w = World(mesh, params; demag = true, Bext = Bext)
    wg = togpu(w); mg = togpu(m); pg = wg.material
    B0 = T(0.3) .* randommag!(zeromag(T, mesh))           # prefilled buffer for add=true
    for add in (false, true)
        tag = add ? "add" : "set"
        Bc = copy(B0); Bg = togpu(B0)
        exchange!(Bc, m, mesh, params; add = add); exchange!(Bg, mg, mesh, pg; add = add)
        check("$label exchange ($tag)", Bg, Bc, tol)
        Bc = copy(B0); Bg = togpu(B0)
        anisotropy!(Bc, m, mesh, params; add = add); anisotropy!(Bg, mg, mesh, pg; add = add)
        check("$label anisotropy ($tag)", Bg, Bc, tol)
        Bc = copy(B0); Bg = togpu(B0)
        dmi!(Bc, m, mesh, params; add = add); dmi!(Bg, mg, mesh, pg; add = add)
        check("$label DMI ($tag)", Bg, Bc, tol)
        Bc = copy(B0); Bg = togpu(B0)
        zeeman!(Bc, Bext; add = add); zeeman!(Bg, Bext; add = add)
        check("$label Zeeman ($tag)", Bg, Bc, tol)
        Bc = copy(B0); Bg = togpu(B0)
        _demag!(Bc, m, w; add = add); _demag!(Bg, mg, wg; add = add)
        check("$label demag ($tag)", Bg, Bc, tol)
    end
    # the demag padding is never written: still exactly zero after the calls
    P = Array(wg.demagplan.padm); Nx, Ny, Nz = mesh.size
    pad = copy(P); pad[1:Nx, 1:Ny, 1:Nz, :] .= 0
    ok = all(iszero, pad)
    push!(RESULTS, ("$label demag padding stays zero", ok ? 0.0 : 1.0, 0.0, ok))
    @printf("  %-58s %s\n", "$label demag padding stays zero", ok ? "ok" : "FAIL")

    Bc = similar(m); Bg = similar(mg)
    effectivefield!(Bc, m, w); effectivefield!(Bg, mg, wg)
    check("$label effective field", Bg, Bc, tol)
    α = JuliaMag.damping(params)
    dc = similar(m); dg = similar(mg)
    torque!(dc, m, Bc, α); torque!(dg, mg, Bg, α)
    check("$label LLG torque", dg, dc, tol)
    return m, w
end

function stt_terms(label, mesh, mat::Material, T)
    tol = tolfor(T)
    Random.seed!(2)
    m = randommag!(zeromag(T, mesh)); mg = togpu(m)
    τ0 = T(1e9) .* randommag!(zeromag(T, mesh))
    for J in ((2e11, 0, 0), (1e11, -3e11, 0), (5e10, 2e10, 4e11))
        for add in (false, true)
            τc = copy(τ0); τg = togpu(τ0)
            zhanglitorque!(τc, m, mesh, mat, J; add = add)
            zhanglitorque!(τg, mg, mesh, mat, J; add = add)
            check("$label Zhang-Li J=$(J) ($(add ? "add" : "set"))", τg, τc, tol)
        end
    end
    for add in (false, true)
        τc = copy(τ0); τg = togpu(τ0)
        slonczewskitorque!(τc, m, mesh, mat, 3e11, (cos(0.3), sin(0.3), 0.2), 3e-9; add = add)
        slonczewskitorque!(τg, mg, mesh, mat, 3e11, (cos(0.3), sin(0.3), 0.2), 3e-9; add = add)
        check("$label Slonczewski ($(add ? "add" : "set"))", τg, τc, tol)
    end
end

function minimizer_terms(label, mesh, params, T)
    tol = tolfor(T)
    Random.seed!(3)
    m = randommag!(zeromag(T, mesh)); clearempty!(m, params)
    w = World(mesh, params; demag = true); wg = togpu(w); mg = togpu(m)
    B = similar(m); effectivefield!(B, m, w); Bg = togpu(B)
    τc = similar(m); τg = similar(mg)
    _damping_torque!(τc, m, B, T(γLL)); _damping_torque!(τg, mg, Bg, T(γLL))
    check("$label damping torque", τg, τc, tol)
    mc = copy(m); mgg = copy(mg)
    dc = _cayley_step!(mc, τc, T(1e-13)); dg = _cayley_step!(mgg, τg, T(1e-13))
    check("$label Cayley step m", mgg, mc, tol)
    check("$label Cayley step max|Δm|²", [dg], [dc], tol)
    # each inner product on its own (they differ by many orders of magnitude)
    sc = _bb_sums(mc, m, τc, T(0.9) .* τc); sg = _bb_sums(mgg, mg, τg, T(0.9) .* τg)
    for (q, nm) in enumerate(("ss", "sy", "yy"))
        check("$label BB sum $nm", [sg[q]], [sc[q]], T == Float64 ? 1e-11 : 1e-4)
    end
    mn = copy(m); normalize!(mn); mgn = T(1.7) .* mg; normalize!(mgn)
    check("$label normalize!", mgn, mn, tol)
end

# Vortex start: 20 BB minimizer steps, a fixed-step RK4 run (the production path of
# the disc driver, deterministic), and an adaptive Tsit5 run compared at its tolerance
# level (an accept/reject decision may differ).
#
# The BB sequence far from equilibrium is chaotic: the step length ss/sy feeds back
# into the next state, so rounding differences grow ~1e4× in 20 steps. In the "mat
# open 3D Dind" case (vortex in a strong-Ku/DMI film, Δm ~ 0.3 per step) the CPU run
# itself moves by 5.7e-4 (Float32) or 2.5e-12 (Float64) when mx is perturbed by one
# ulp (the edge-smoothed disc near equilibrium: 8.5e-6 / 3e-14).
# A fixed tolerance is therefore ill-posed; the GPU result is required to lie within
# 10× that intrinsic spread, measured here on the CPU in the same precision (and
# never tighter than the floor 1e-9 / 1e-4).
function _ulp_perturbed(m)
    mp = copy(m)
    @views mp[1, :, :, :] .= ifelse.(mp[1, :, :, :] .!= 0, nextfloat.(mp[1, :, :, :]), mp[1, :, :, :])
    return mp
end

function solver_terms(label, mesh, params, T)
    m = setconfig(T, mesh, VortexConfig(mesh)); clearempty!(m, params)
    w = World(mesh, params; demag = true, Bext = (0.02, 0, 0))
    wg = togpu(w)
    mnc = Minimizer(w, copy(m)); for _ in 1:20; minimizestep!(mnc); end
    mnp = Minimizer(w, _ulp_perturbed(m)); for _ in 1:20; minimizestep!(mnp); end
    mng = Minimizer(wg, togpu(m)); for _ in 1:20; minimizestep!(mng); end
    spread = relerr(mnp.m, mnc.m)
    @printf("  %-58s %9.2e\n", "$label relax, intrinsic 1-ulp spread", spread)
    check("$label relax, 20 BB steps, m (tol 10× spread)", mng.m, mnc.m,
          max(T == Float64 ? 1e-9 : 1e-4, 10 * spread))

    α = JuliaMag.damping(params); dt = T(2e-14)
    function rk4run!(mm, ww, n)
        B = similar(mm); k1 = similar(mm); k2 = similar(mm); k3 = similar(mm); k4 = similar(mm)
        tmp = similar(mm); hdt = T(0.5) * dt; dt6 = dt / T(6); two = T(2)
        f!(d, x) = (effectivefield!(B, x, ww); torque!(d, x, B, α))
        for _ in 1:n
            f!(k1, mm); @. tmp = mm + hdt*k1; normalize!(tmp); f!(k2, tmp)
            @. tmp = mm + hdt*k2; normalize!(tmp); f!(k3, tmp)
            @. tmp = mm + dt*k3;  normalize!(tmp); f!(k4, tmp)
            @. mm = mm + dt6*(k1 + two*k2 + two*k3 + k4); normalize!(mm)
        end
        mm
    end
    mc = rk4run!(copy(m), w, 200); mgr = rk4run!(togpu(m), wg, 200)
    check("$label fixed-step RK4 run 4 ps, m", mgr, mc, T == Float64 ? 1e-10 : 1e-4)

    itc = Integrator(w, copy(m); tend = 1e-10); advance!(itc, 1e-11)
    itg = Integrator(wg, togpu(m); tend = 1e-10); advance!(itg, 1e-11)
    check("$label adaptive Tsit5 run 10 ps, m", state(itg), state(itc), T == Float64 ? 1e-5 : 1e-3)
end

# mumax3 time steppers (src/timestepper.jl) on the GPU vs the CPU, with a spin
# current in the World (Zhang-Li + Slonczewski), and the Simulation driver layer.
function stepper_terms(label, mesh, params, T)
    tol = T == Float64 ? 1e-10 : 1e-4
    m = setconfig(T, mesh, VortexConfig(mesh)); clearempty!(m, params)
    Random.seed!(7); τ = randommag!(zeromag(T, mesh)) .* T(1e9)
    check("$label maxtorque", [maxtorque(togpu(τ))], [maxtorque(τ)], T == Float64 ? 1e-14 : 1e-6)
    # Error accumulator: atomic-max reduction on the bit patterns vs the CPU loop,
    # difference form with a scale, the peak slot, NaN stickiness.
    τ2 = randommag!(zeromag(T, mesh)) .* T(3e8)
    ac = JuliaMag._newacc(τ); ag = JuliaMag._newacc(togpu(τ))
    for (x, y, s) in ((τ, nothing, 1e-13), (τ, τ2, 2e-13), (τ2, nothing, 1e-13))
        JuliaMag._acc_reset!(ac); JuliaMag._acc_max!(ac, x, y, s)
        JuliaMag._acc_reset!(ag); JuliaMag._acc_max!(ag, togpu(x), y === nothing ? nothing : togpu(y), s)
        check("$label error accumulator (this step, peak)", collect(JuliaMag._acc_read(ag)),
              collect(JuliaMag._acc_read(ac)), T == Float64 ? 1e-14 : 1e-6)
    end
    τn = copy(τ); τn[2, end, 1, 1] = T(NaN); ag = JuliaMag._newacc(togpu(τ))
    JuliaMag._acc_max!(ag, togpu(τn), nothing, 1.0)
    JuliaMag._acc_reset!(ag); JuliaMag._acc_max!(ag, togpu(τ), nothing, 1.0)
    r = JuliaMag._acc_read(ag)
    check("$label accumulator: NaN sticky in the peak only", [Float64(!isnan(r[1]) && isnan(r[2]))], [1.0], 0)
    check("$label maxtorque with a NaN cell is NaN", [Float64(isnan(maxtorque(togpu(τn))))], [1.0], 0)
    w = World(mesh, params; demag = true, Bext = (0.02, 0, 0))
    setcurrent!(w, (2e11, -1e11, 4e11); zhangli = true, slonczewski = true,
                fixedlayer = (cos(0.3), sin(0.3), 0.2), thickness = 3e-9)
    wg = togpu(w)
    dc = similar(m); dg = togpu(similar(m))
    llgrhs!(dc, m, w); llgrhs!(dg, togpu(m), wg)
    check("$label llgrhs! (LLG + Zhang-Li + Slonczewski)", dg, dc, tolfor(T))
    # energy! (field + one Float64 reduction) and the linear self-field operator
    Ec = energy!(similar(m), m, w); Eg = energy!(togpu(similar(m)), togpu(m), wg)
    check("$label energy! (total energy)", [Eg], [Ec], T == Float64 ? 1e-12 : 1e-4)
    Random.seed!(9); δ = randn(T, size(m)...)
    Lc = similar(m); linearfield!(Lc, δ, w)
    Lg = togpu(similar(m)); linearfield!(Lg, togpu(δ), wg)
    check("$label linearfield! (no Zeeman, arbitrary δm)", Lg, Lc, tolfor(T))
    for s in (:euler, :heun, :rk23, :rk4, :rk45)
        mc = copy(m); mg = togpu(m)
        tc = TimeStepper(w, mc; solver = s, fixdt = 2e-14); advanceto!(tc, 2e-12)
        tg = TimeStepper(wg, mg; solver = s, fixdt = 2e-14); advanceto!(tg, 2e-12)
        check("$label $s fixed step, 100 steps, m", mg, mc, tol)
        # The embedded error estimate is a cancellation Σ cᵢkᵢ (Σ cᵢ = 0): its rounding
        # floor is ~eps(T)·max|k|·dt, so with a small fixed step (RK45 here: err/(|k|dt)
        # ~ 5e-9) CPU and GPU agree only to eps(T)·|k|dt/err relative.
        floor = 20 * eps(T) * maxtorque(tc.k[1]) * 2e-14 / JuliaMag.peakerr(tc)
        check("$label $s fixed step, lasterr/peakerr (device-side)",
              [JuliaMag.lasterr(tg), JuliaMag.peakerr(tg)],
              [JuliaMag.lasterr(tc), JuliaMag.peakerr(tc)], max(T == Float64 ? 1e-8 : 2e-3, floor))
    end
    mn = copy(m); mn[1, cld(size(m, 2), 2), cld(size(m, 3), 2), 1] = T(NaN)   # a cell inside the magnet
    tg = TimeStepper(wg, togpu(mn); solver = :heun, fixdt = 2e-14)
    diverged = try advanceto!(tg, 2e-13); false catch e; occursin("diverged", sprint(showerror, e)) end
    check("$label fixed step on a NaN state stops with an error", [Float64(diverged)], [1.0], 0)
    mc = copy(m); mg = togpu(m)
    tc = TimeStepper(w, mc; solver = :rk45); advanceto!(tc, 5e-12)
    tg = TimeStepper(wg, mg; solver = :rk45); advanceto!(tg, 5e-12)
    @printf("  %-58s CPU %d/%d  GPU %d/%d\n", "$label rk45 adaptive steps/undone",
            tc.nsteps, tc.nundone, tg.nsteps, tg.nundone)
    check("$label rk45 adaptive 5 ps, m", mg, mc, T == Float64 ? 1e-8 : 1e-3)
    check("$label rk45 adaptive 5 ps, last dt", [tg.dt], [tc.dt], T == Float64 ? 1e-6 : 1e-2)

    # Simulation on the GPU (togpu): table rows (device and host columns) and OVF.
    dir = mktempdir()
    mk() = (sim = Simulation(mesh, params; demag = true, Bext = (0.02, 0, 0), verbose = false);
            setmag!(sim, m); tableadd!(sim, q_maxtorque()); tableadd!(sim, q_energy());
            tableautosave!(sim, 1e-12); setcurrent!(sim, (2e11, 0, 4e11); zhangli = true,
            slonczewski = true, fixedlayer = (1, 0, 0), thickness = 3e-9); sim)
    sc = mk(); sg = togpu(mk()); setoutdir!(sg, dir); autosave!(sg, 2e-12)
    run!(sc, 4e-12; solver = :rk4, fixdt = 2e-14); run!(sg, 4e-12; solver = :rk4, fixdt = 2e-14)
    check("$label togpu(sim) run!, m", sg.m, sc.m, tol)
    check("$label togpu(sim) table (t, m, maxTorque, E_total)",
          reduce(vcat, sg.table.rows), reduce(vcat, sc.table.rows), T == Float64 ? 1e-9 : 1e-4)
    m2, _ = loadovf(joinpath(dir, "m000002.ovf"))
    check("$label togpu(sim) OVF autosave m000002 = state", m2, Array(sg.m), T == Float64 ? 1e-15 : 1e-7)
    println("  ", deviceinfo(sg.m))
end

function average_terms(label, mesh, params, T)
    Random.seed!(5)
    m = randommag!(zeromag(T, mesh)); clearempty!(m, params)
    pg = togpu(World(mesh, params; demag = false)).material
    check("$label average", collect(average(togpu(m))), collect(average(m)),
          T == Float64 ? 1e-12 : 1e-6)
    if params isa RegionParams
        check("$label average_magnet", collect(JuliaMag.average_magnet(togpu(m), pg)),
              collect(average_magnet(m, params)), T == Float64 ? 1e-12 : 1e-6)
    end
end

# GNEB on the GPU: the band built on CuArrays (same initial path) must have the same
# energies and forces as on the CPU, and relax to the same barrier (the barrier is a
# property of the saddle, so it agrees to the force tolerance even though the BB
# iterates diverge at the rounding level). Wire: domain-wall nucleation with demag;
# disc: RegionParams with an empty background, no edge smoothing (conservative).
function gneb_terms(T)
    wire = Mesh((32, 1, 1), (2e-9, 2e-9, 2e-9))
    mw = Material(Msat = T(8e5), Aex = T(1.3e-11), alpha = T(0.5), Ku = T(5e5), anisU = (1, 0, 0))
    dm = Mesh((16, 16, 1), (4e-9, 4e-9, 4e-9))
    rp = RegionParams(dm, Material(Msat = T(8e5), Aex = T(1.3e-11), alpha = T(0.5)))
    setregion!(rp, 0; Msat = 0.0)
    setregion!(rp, 1; Msat = 8e5, Aex = 1.3e-11, Ku = 2e5, anisU = (1, 0, 0))
    setgeometry!(rp, Cylinder(60e-9, 1.0); id = 1)
    for (lab, mesh, par, Bx) in (("wire", wire, mw, -0.1), ("disc regions", dm, rp, -0.05))
        w = World(mesh, par; demag = true, Bext = (Bx, 0, 0)); wg = togpu(w)
        mA = uniform(T, mesh, (1, 0, 0)); mB = uniform(T, mesh, (-1, 0, 0))
        m0 = zeromag(T, mesh)                       # a wall in the middle as waypoint
        for I in CartesianIndices(axes(m0)[2:4])
            θ = π / (1 + exp(-(I[1] - mesh.size[1] / 2) / 2.5))
            m0[1, I] = cos(θ); m0[2, I] = sin(θ); m0[3, I] = 0.02
        end
        normalize!(m0)
        foreach(m -> clearempty!(m, par), (mA, m0, mB))
        # a cell antiparallel between neighbours (and empty cells in the disc): midpoints
        q = copy(mA); q[:, mesh.size[1] ÷ 2, mesh.size[2] ÷ 2 + 1, 1] .= (-1, 0, 0)
        check("GNEB $lab: midpoints, antiparallel cell",
              midpointenergies(GNEB(wg, [togpu(mA), togpu(q), togpu(m0), togpu(mB)]; nimages = 4)),
              midpointenergies(GNEB(w, [mA, q, m0, mB]; nimages = 4)), T == Float64 ? 1e-12 : 1e-5)
        nc = GNEB(w, [mA, m0, mB]; nimages = 9)
        ng = GNEB(wg, [togpu(mA), togpu(m0), togpu(mB)]; nimages = 9)
        check("GNEB $lab: initial energies", ng.E, nc.E, T == Float64 ? 1e-12 : 1e-5)
        check("GNEB $lab: midpoint energies", midpointenergies(ng), midpointenergies(nc), T == Float64 ? 1e-12 : 1e-5)
        check("GNEB $lab: initial forces (all images)", reduce(vcat, vec.(Array.(ng.F))),
              reduce(vcat, vec.(nc.F)), T == Float64 ? 1e-10 : 1e-4)
        tol = T == Float64 ? 1e-6 : 1e-3
        relax!(nc; stopforce = tol); relax!(ng; stopforce = tol)
        check("GNEB $lab: relaxed barrier", [barrier(ng)], [barrier(nc)], T == Float64 ? 1e-8 : 1e-4)
        check("GNEB $lab: converged on the GPU", [ng.maxforce < tol ? 0.0 : 1.0], [0.0], 0.0)
    end
end

for T in (Float64, Float32)
    println("\n================ ", T, " ================")
    gneb_terms(T)
    mI, mB = mats(T)
    for (lab, mesh, mat) in (
            ("mat open 3D Dind", Mesh((20, 16, 4), (3e-9, 3e-9, 2e-9)), mI),
            ("mat periodic-xy 2D Dind", Mesh((24, 20, 1), (3e-9, 3e-9, 2e-9); pbc = (1, 1, 0)), mI),
            ("mat periodic-xz 3D Dbulk", Mesh((16, 12, 6), (2e-9, 2e-9, 2e-9); pbc = (1, 0, 1)), mB),
            ("mat periodic-x odd Nx (odd P1)", Mesh((15, 8, 2), (3e-9, 3e-9, 3e-9); pbc = (1, 0, 0)), mI),
            ("mat wire 1×16×4", Mesh((1, 16, 4), (3e-9, 3e-9, 3e-9)), mB))
        field_terms(lab, mesh, mat, T)
        stt_terms(lab, mesh, mat, T)
        minimizer_terms(lab, mesh, mat, T)
        average_terms(lab, mesh, mat, T)
    end
    for (lab, (mesh, rp)) in (("regions bilayer", bilayer(T)), ("regions bulk periodic-x", bilayer_bulk(T)),
                              ("regions disc es4", disc(T)))
        field_terms(lab, mesh, rp, T)
        minimizer_terms(lab, mesh, rp, T)
        solver_terms(lab, mesh, rp, T)
        stepper_terms(lab, mesh, rp, T)
        average_terms(lab, mesh, rp, T)
    end
    stepper_terms("mat open 3D Dind", Mesh((20, 16, 4), (3e-9, 3e-9, 2e-9)), mI, T)
    solver_terms("mat open 3D Dind", Mesh((20, 16, 4), (3e-9, 3e-9, 2e-9)), mI, T)
end

nfail = count(r -> !r[4], RESULTS)
println("\n", length(RESULTS) - nfail, "/", length(RESULTS), " checks passed")
nfail == 0 || (println("FAILED:"); foreach(r -> r[4] || println("  ", r[1], "  ", r[2]), RESULTS))
exit(nfail == 0 ? 0 : 1)
