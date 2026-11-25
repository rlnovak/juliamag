# Linear eigenmodes: JuliaMag against MicroMagnetic.jl (its src/eigen, Fan et al., JMMM 622,
# 172989 (2025)), which linearizes the LLG equation by automatic differentiation.
#
#   julia --project=ENV gpu_validation/eigenmodes/compare_micromagnetic.jl CASE...
#
# ENV has JuliaMag and MicroMagnetic. CASE: fmr, vortex, regions, skyrmion (default: all).
# Like for like: the same state (relaxed in JuliaMag), gamma, open DMI boundaries and, with
# demag, Newell's tensor in both codes (newell.jl here, MicroMagnetic's own there).
#   1. fields of the state and of a random state;
#   2. the linear operators, as maps of the same random tangent fields;
#   3. all the frequencies; 4. the damped spectrum, against JuliaMag's matrix and against
#      its first-order linewidths; 5. overlaps of the lowest modes.
# fmr: also the frequencies about the state that MicroMagnetic relaxes itself, against the paper.

using JuliaMag, Printf, Random
using LinearAlgebra: norm, dot, eigen, eigvals
import MicroMagnetic as MM
include(joinpath(@__DIR__, "newell.jl"))

const MU0 = 4π * 1e-7
const GAMMA = JuliaMag.γLL * MU0                     # m/(A s), MicroMagnetic's unit
const PAPERF = [8.270, 9.402, 10.839, 11.233, 11.992, 13.045, 13.816, 14.276, 15.316, 15.907, 16.718, 17.234, 17.457, 18.409, 19.806]

# --- the cases: JuliaMag world and starting state, and MicroMagnetic terms -------------------
# `ms`, `ku`, ...: functions of (i, j, k), region constants. `dmi`: (type, D(i,j,k)).
struct Case
    mesh::Mesh
    world
    start
    alpha::Float64
    terms            # sim -> nothing, adds the terms (without Zeeman and demag)
    Ms               # (i, j, k) -> A/m
end

function usedemag!(w)
    usenewell!(w)
    return w
end
fill3(f, mesh) = [f(i, j, k) for i in 1:mesh.size[1], j in 1:mesh.size[2], k in 1:mesh.size[3]]
asfun(f) = (i, j, k, dx, dy, dz) -> f(i, j, k)

function case_fmr()
    mesh = Mesh((24, 24, 2), (5e-9, 5e-9, 5e-9)); B = MU0 * 8e4
    w = usedemag!(World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.008);
                        Bext = (B * 0.81923192051904048, B * 0.57346234436332832, 0)))
    Case(mesh, w, uniform(mesh, (0.82, 0.57, 0.0) ./ hypot(0.82, 0.57)), 0.008,
         sim -> MM.add_exch(sim, 1.3e-11), (i, j, k) -> 8e5)
end

function case_vortex()
    mesh = Mesh((28, 28, 1), (128e-9 / 28, 128e-9 / 28, 16e-9))
    rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.08e-11, alpha = 0.01))
    setregion!(rp, 0; Msat = 0.0); setregion!(rp, 1; Msat = 8e5, Aex = 1.08e-11)
    setgeometry!(rp, Cylinder(128e-9, 1.0); id = 1)
    w = usedemag!(World(mesh, rp; demag = true))
    m = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1)); clearempty!(m, rp)
    msc = JuliaMag._msatcells(rp, mesh)
    Case(mesh, w, m, 0.01, sim -> MM.add_exch(sim, 1.08e-11), (i, j, k) -> msc[1, i, j, k])
end

# two materials differing in every constant, interfacial DMI, demag, oblique field
function case_regions()
    mesh = Mesh((14, 10, 2), (3e-9, 2.5e-9, 2e-9))
    reg(i, j, k) = (i <= 6 && j <= 7) ? 2 : 1
    Ms = (8e5, 5e5); A = (1.3e-11, 0.8e-11); K = (6e5, 2e5); D = (1.5e-3, 0.6e-3)
    rp = RegionParams(mesh, Material(Msat = Ms[1], Aex = A[1], alpha = 0.02, Ku = K[1], anisU = (0, 0, 1), Dind = D[1]))
    for r in 1:2
        setregion!(rp, r; Msat = Ms[r], Aex = A[r], alpha = 0.02, Ku = K[r], anisU = (0.2, 0, 1), Dind = D[r])
    end
    for k in 1:2, j in 1:10, i in 1:14
        defregioncell!(rp.regions, reg(i, j, k), i, j, k)
    end
    w = usedemag!(World(mesh, rp; Bext = (0.15, 0.06, 0.25), openbc = true))
    function terms(sim)
        MM.add_exch(sim, asfun((i, j, k) -> A[reg(i, j, k)]))
        MM.add_anis(sim, asfun((i, j, k) -> K[reg(i, j, k)]); axis = (0.2, 0, 1))
        MM.add_dmi(sim, asfun((i, j, k) -> INTERFACIALSIGN * D[reg(i, j, k)]); type = "interfacial")
    end
    Case(mesh, w, uniform(mesh, (0.3, 0.1, 1.0) ./ norm((0.3, 0.1, 1.0))), 0.02, terms, (i, j, k) -> Ms[reg(i, j, k)])
end

# one Bloch skyrmion confined in a square: bulk DMI, demag
function case_skyrmion()
    mesh = Mesh((24, 24, 1), (4e-9, 4e-9, 4e-9))
    Ms, A, D = 3.84e5, 3.25e-12, 5.83e-4
    w = usedemag!(World(mesh, Material(Msat = Ms, Aex = A, alpha = 0.1, Dbulk = D); Bext = (0, 0, 0.3), openbc = true))
    m = zeromag(Float64, mesh)
    for j in 1:24, i in 1:24
        x, y = i - 12.5, j - 12.5; r = hypot(x, y); θ = π / (1 + exp((r - 5) / 1.2))
        m[:, i, j, 1] .= (-sin(θ) * y / r, sin(θ) * x / r, cos(θ))
    end
    Case(mesh, w, normalize!(m), 0.1, sim -> (MM.add_exch(sim, A); MM.add_dmi(sim, D)), (i, j, k) -> Ms)
end

# Sign of MicroMagnetic's interfacial constant in JuliaMag's (mumax3's) convention: opposite. Its
# kernels.jl (interfacial_dmi_kernel!) has B_i = (D/Ms dx) y×m of the neighbour at -x; mumax3 and
# JuliaMag have the opposite sign (and the fields agree only so: see the printed differences)
const INTERFACIALSIGN = parse(Int, get(ENV, "DSIGN", "-1"))

# --- MicroMagnetic side --------------------------------------------------------------
function mmsim(c::Case, precision)
    MM.set_backend("cpu"); MM.set_precision(precision)
    n, d = c.mesh.size, c.mesh.cellsize
    mm = MM.FDMesh(; nx = n[1], ny = n[2], nz = n[3], dx = d[1], dy = d[2], dz = d[3])
    sim = MM.Sim(mm; name = "cmp", driver = precision === Float64 ? "SD" : "None", save_data = false)
    MM.set_Ms(sim, asfun(c.Ms))
    c.terms(sim)
    MM.add_zeeman(sim, c.world.Bext ./ MU0)
    MM.add_demag(sim)
    return sim
end
setspin!(sim, m, on) = (copyto!(sim.spin, vec(m .* on)); sim)

# A MicroMagnetic field [T] of the state m: Float64 simulation (FFT demag)
function mmfield(sim, m, on)
    setspin!(sim, m, on)
    MM.effective_field(sim, sim.spin, 0.0)
    return MU0 .* reshape(copy(sim.field), size(m)) .* on
end

# Matrix of MicroMagnetic's operator on the magnetic cells, with its frames. The symbolic
# precision makes the demag tensor (direct sum) a matrix of boxed numbers: it is replaced
# by the same numbers as Float64, which only makes the products fast.
function mmoperator(c, m, on, alpha)
    sim = mmsim(c, MM.SFlat{16})
    setspin!(sim, m, on)
    for (n, it) in enumerate(sim.interactions)
        if it isa MM.DirectDemag
            sim.interactions[n] = MM.DirectDemag{Float64}(MM.cpart.(Array(it.tensor)), zeros(length(it.field)), zeros(length(it.energy)), it.name)
        end
    end
    # spin-dependent products of the direct demag read mu0_Ms only for its energy
    return MM.build_matrix(sim; gamma = GAMMA, alpha = alpha, matrixfree = true)
end

# tangent field (3,...) -> MicroMagnetic coordinates and back
function tomm(x, op)
    X = reshape(x, 3, :); u = zeros(2 * size(X, 2))
    for i in axes(X, 2)
        R = op.R_invs[i]
        u[2i-1] = R[1, 1] * X[1, i] + R[1, 2] * X[2, i] + R[1, 3] * X[3, i]
        u[2i] = R[2, 1] * X[1, i] + R[2, 2] * X[2, i] + R[2, 3] * X[3, i]
    end
    return u
end
function frommm(u, op, sz)
    X = zeros(eltype(u), 3, prod(sz[2:4]))
    for i in axes(X, 2), q in 1:3
        X[q, i] = op.Rs[i][q, 1] * u[2i-1] + op.Rs[i][q, 2] * u[2i]
    end
    return reshape(X, sz)
end

# --- the comparison ------------------------------------------------------------------
# largest relative difference of two spectra sorted by frequency
function cmpspec(a, b)
    n = length(a) == length(b) ? maximum(abs.(b ./ a .- 1)) : NaN
    return n
end
pos(ev) = sort(ev[imag.(ev).>0]; by = imag)

function compare(name)
    c = getfield(@__MODULE__, Symbol("case_", name))()
    w = c.world
    mn = Minimizer(w, copy(c.start); stopdm = 1e-11); minimize!(mn; maxsteps = 200_000)
    m = mn.m; H = TangentHessian(w, m)
    msc = JuliaMag._msatcells(w.material, c.mesh); on = msc .!= 0
    act = findall(vec(on)); idx = reduce(vcat, [[2i - 1, 2i] for i in act])
    @printf("\n=== %s: %d cells, %d magnetic; max|m x B| %.1e T, <m> (%.4f, %.4f, %.4f)\n", name, length(on), length(act), H.torque, average(m)...)
    # 1. fields
    s64 = mmsim(c, Float64)
    rng = MersenneTwister(5)
    mr = randommag!(rng, similar(m)) .* on
    for (lab, x) in (("state", m), ("random state", mr))
        B = effectivefield!(similar(x), x, w) .* on
        Bm = mmfield(s64, x, on)
        @printf("field, %-12s  max|dB|/max|B| = %.1e\n", lab, maximum(abs.(Bm .- B)) / maximum(abs, B))
    end
    # 2. operators
    op = mmoperator(c, m, on, 0.0); opa = mmoperator(c, m, on, c.alpha)
    γ = JuliaMag.γLL
    function jmdot(δ, α)                          # d(δm)/dt of JuliaMag for a tangent field
        y = hessian!(similar(m), H.sw .* δ, H)    # A u
        Ju = similar(y); JuliaMag._cross!(Ju, H.m, y)
        return γ / (1 + α^2) .* H.isw .* (Ju .- α .* y)
    end
    for (lab, o, α) in (("undamped", op, 0.0), ("damped", opa, c.alpha))
        worst = 0.0
        for _ in 1:5
            δ = JuliaMag._tangentproject!(randn(rng, size(m)...), H)
            y = zeros(length(idx) == 0 ? 0 : 2length(on))
            MM.LinearAlgebra.mul!(y, o, tomm(δ, o))
            a = jmdot(δ, α); b = frommm(y, o, size(m)) .* on
            worst = max(worst, norm(a .- b) / norm(a))
        end
        @printf("operator, %-9s  |y_MM - y_JM|/|y_JM| <= %.1e  (5 random tangent fields)\n", lab, worst)
    end
    # 3. frequencies, all
    t = @elapsed Bm = Matrix(op)[idx, idx]
    F = eigen(Bm); ev = F.values
    em = eigenmodes(w, m; nmodes = length(act), method = :dense)
    fm = imag.(pos(ev)) ./ 2π
    @printf("frequencies: %d (MM matrix %.0f s); largest relative difference %.1e; MM max|Re|/max|Im| %.1e\n",
            length(fm), t, cmpspec(em.frequencies, fm), maximum(abs.(real.(ev))) / maximum(abs.(imag.(ev))))
    n10 = min(10, length(fm))
    worstcells = [abs(fm[k] / em.frequencies[k] - 1) for k in 1:n10]
    @printf("   lowest %d: largest relative difference %.1e\n", n10, maximum(worstcells))
    for k in 1:min(5, n10)
        @printf("   %d  MM %.9f GHz  JM %.9f GHz  rel %+.1e\n", k, fm[k] / 1e9, em.frequencies[k] / 1e9, fm[k] / em.frequencies[k] - 1)
    end
    # 4. damped spectrum
    ea = pos(eigvals(Matrix(opa)[idx, idx]))
    A, = densehessian(H); JA = JuliaMag._Jrows!(similar(A), A)
    ej = pos(eigvals(γ / (1 + c.alpha^2) .* (JA .- c.alpha .* A)))
    @printf("damped (alpha %g): frequencies %.1e, decay rates %.1e (largest relative differences to JuliaMag's damped matrix)\n",
            c.alpha, cmpspec(imag.(ej), imag.(ea)), length(ej) == length(ea) ? maximum(abs.(real.(ea) ./ real.(ej) .- 1)) : NaN)
    for k in 1:min(3, n10)
        @printf("   %d  MM decay %.8f 1/ns  JM first order %.8f 1/ns  rel %+.1e (alpha^2 = %.0e)\n", k, -real(ea[k]) / 1e9, em.linewidths[k] / 1e9,
                -real(ea[k]) / em.linewidths[k] - 1, c.alpha^2)
    end
    # 5. modes
    ord = sortperm(imag.(ev)); ord = ord[imag.(ev[ord]).>0]
    worst = 1.0; skipped = 0
    for k in 1:n10
        fu = zeros(ComplexF64, 2length(on)); fu[idx] .= F.vectors[:, ord[k]]
        a = frommm(fu, op, size(m)); b = em.modes[k]
        near = minimum(abs(em.frequencies[q] / em.frequencies[k] - 1) for q in 1:length(act) if q != k)
        near < 1e-6 && (skipped += 1; continue)    # degenerate: the modes of the pair are arbitrary
        ov = abs(dot(vec(a .* msc), vec(b))) / sqrt(real(dot(vec(a .* msc), vec(a))) * real(dot(vec(b .* msc), vec(b))))
        worst = min(worst, ov)
    end
    @printf("modes 1-%d (%d degenerate ones skipped): 1 - |overlap| <= %.1e\n", n10, skipped, 1 - worst)
    if name == "fmr"
        ms = mmsim(c, Float64); setspin!(ms, c.start, on)
        MM.relax(ms; max_steps = 200_000, stopping_dmdt = 1e-5)
        mo = reshape(copy(ms.spin), size(m))
        oo = mmoperator(c, mo, on, 0.0)
        fo = imag.(pos(eigvals(Matrix(oo)))) ./ 2π
        g = 2.210173e5 / GAMMA
        @printf("state relaxed by MicroMagnetic: max|dm| %.1e; 15 lowest frequencies differ by %.1e\n", maximum(abs.(mo .- m)), maximum(abs.(fo[1:15] ./ em.frequencies[1:15] .- 1)))
        for k in 1:15
            @printf("   %2d  MM (own state) %.6f GHz  paper %.3f  rel %+.2e\n", k, fo[k] * g / 1e9, PAPERF[k], fo[k] * g / 1e9 / PAPERF[k] - 1)
        end
    end
    flush(stdout)
end

foreach(compare, isempty(ARGS) ? ["regions", "skyrmion", "vortex", "fmr"] : ARGS)
println("done")
