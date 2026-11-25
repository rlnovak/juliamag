# Energy Hessian and linear eigenmodes. References:
# - the Hessian against second differences of the energy along great circles, at random
#   states, for every term, regions and empty cells;
# - the linearized dynamics against finite differences of llgrhs! (sign and factor of J A);
# - Kittel's macrospin, spin waves of a periodic chain (exact dispersion of the discretized
#   exchange; the DMI shift ±(2D/Ms) sin(kc)/c), damping of the macrospin;
# - Stoner–Wohlfarth curvatures at the minimum and at the saddle;
# - the dense and the matrix-free solvers against each other.
using JuliaMag, Test
import Random
using LinearAlgebra: norm, dot, cross, eigvals

const _eγ = JuliaMag.γLL

_tangent(rng, H) = (x = randn(rng, eltype(H.m), size(H.m)); JuliaMag._tangentproject!(x, H))

# m moved along the great circles of δ (tangent)
function _geodesic(m, δ)
    out = copy(m)
    for I in CartesianIndices(axes(m)[2:4])
        d = norm(δ[:, I])
        d == 0 || (out[:, I] .= cos(d) .* m[:, I] .+ sin(d) / d .* δ[:, I])
    end
    return out
end

function _hesscases()
    base = (Msat = 8e5, Aex = 1.3e-11, alpha = 0.02)
    m1 = Mesh((6, 5, 3), (3e-9, 3e-9, 2e-9))
    m2 = Mesh((6, 6, 2), (3e-9, 3e-9, 3e-9); pbc = (2, 2, 0))
    c = Pair{String,Any}[]
    push!(c, "Ku + demag + Bext" => World(m1, Material(; base..., Ku = 3e5, anisU = (0.3, 0.2, 1.0)); Bext = (0.02, -0.01, 0.03)))
    push!(c, "Dind, chiral" => World(m1, Material(; base..., Dind = 2.5e-3)))
    push!(c, "Dbulk, openbc" => World(m1, Material(; base..., Dbulk = 1.5e-3); demag = false, openbc = true))
    push!(c, "periodic film, Dind" => World(m2, Material(; base..., Dind = 2.5e-3); Bext = (0, 0, 0.1)))
    m4 = Mesh((8, 8, 2), (4e-9, 4e-9, 4e-9))
    rp = RegionParams(m4, Material(; base...))
    setregion!(rp, 0; Msat = 0.0)
    setregion!(rp, 1; Msat = 8e5, Aex = 1.3e-11, Dind = 2e-3)
    setregion!(rp, 2; Msat = 1.4e6, Aex = 2.0e-11, Ku = 5e5, anisU = (0, 0, 1), Dind = -1e-3)
    setgeometry!(rp, Cylinder(30e-9, 1.0); id = 1)
    defregion!(rp, 2, (x, y, z) -> x > 0 && x^2 + y^2 < 14e-9^2)
    push!(c, "two regions, empty background, demag" => World(m4, rp; Bext = (0.01, 0, 0)))
    return c
end

@testset "eigenmodes" begin
    @testset "local frame" begin
        rng = Random.MersenneTwister(2)
        vs = [randn(rng, 3) for _ in 1:50]
        append!(vs, [[0.0, 0, 1], [0.0, 0, -1], [1.0, 0, 0], [1.0, 0, -0.0], [0.0, -1, 1e-17]])
        for v in vs
            n = v ./ norm(v)
            e1, e2 = collect.(JuliaMag._frame(n...))
            @test norm(cross(e1, e2) .- n) < 1e-14
            @test abs(dot(e1, e2)) < 1e-15 && abs(norm(e1) - 1) < 1e-14 && abs(dot(e1, n)) < 1e-15
        end
    end

    @testset "Hessian: second variation, symmetry ($lab)" for (lab, w) in _hesscases()
        rng = Random.MersenneTwister(5)
        m = clearempty!(randommag!(rng, zeromag(Float64, w.mesh)), w.material)
        H = TangentHessian(w, m)
        x, y = _tangent(rng, H), _tangent(rng, H)
        Ax = hessian!(similar(m), x, H); Ay = hessian!(similar(m), y, H)
        @test abs(dot(y, Ax) - dot(x, Ay)) < 1e-12 * norm(x) * norm(Ay)
        @test maximum(abs, sum(Ax .* H.m; dims = 1)) < 1e-12 * maximum(abs, Ax)
        @test all(iszero, Ax .* (H.sw .== 0))
        # E along the geodesic of δ = u/√w: E'' = V Msref u·A u, 6th-order difference
        δ = H.isw .* x
        δ ./= maximum(abs, δ)
        u = H.sw .* δ
        Au = hessian!(similar(m), u, H)
        want = cellvolume(w.mesh) * H.Msref * dot(u, Au)
        E(s) = totalenergy(_geodesic(m, s .* δ), w)
        h = 0.02
        c = (-49 / 18, 3 / 2, -3 / 20, 1 / 90)
        fd = (c[1] * E(0.0) + sum(c[k+1] * (E(k * h) + E(-k * h)) for k in 1:3)) / h^2
        @test fd ≈ want rtol = 1e-6
        # dense form: symmetric, same quadratic form
        A, cells, F = densehessian(H)
        @test maximum(abs, A .- A') < 1e-12 * maximum(abs, A)
        a = zeros(size(A, 1)); JuliaMag._fromfield!(a, u, cells, F)
        @test dot(a, A * a) ≈ dot(u, Au) rtol = 1e-11
        @test JuliaMag._tofield!(similar(u), a, cells, F) ≈ u rtol = 1e-12
    end

    @testset "linearized LLG" begin
        # at the equilibrium m0 = u of a uniform film with Ku and field: dδ/dt = γ m0 × Hδ
        mesh = Mesh((5, 4, 2), (3e-9, 3e-9, 2e-9); pbc = (1, 1, 0))
        for α in (0.0, 0.1)
            w = World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = α, Ku = 8e5, anisU = (0, 0, 1), Dind = 1e-3);
                      Bext = (0, 0, 0.1))
            m0 = uniform(mesh, (0, 0, 1))
            H = TangentHessian(w, m0)
            @test H.torque < 1e-9
            x = _tangent(Random.MersenneTwister(3), H)
            Ax = hessian!(similar(x), x, H)
            JA = JuliaMag._cross!(similar(x), m0, Ax)
            want = _eγ / (1 + α^2) .* (JA .- α .* Ax)
            h = 1e-5
            f(s) = llgrhs!(similar(x), normalize!(m0 .+ s .* x), w)
            fd = (f(h) .- f(-h)) ./ 2h
            @test maximum(abs, fd .- want) < 1e-7 * maximum(abs, want)
        end
    end

    @testset "Kittel macrospin" begin
        mesh = Mesh((1, 1, 1), (4e-9, 6e-9, 2e-9))
        Ms, K, B = 8e5, 3e5, 0.2
        w = World(mesh, Material(Msat = Ms, Aex = 1.3e-11, alpha = 0.01, Ku = K, anisU = (1, 0, 0)); Bext = (B, 0, 0))
        N = [-linearfield!(zeromag(mesh), uniform(mesh, e), World(mesh, Material(Msat = Ms, Aex = 1.3e-11, alpha = 0.01)))[c, 1, 1, 1]
             for (c, e) in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1)))]           # μ0 Ms N_c [T]
        a, b = B + 2K / Ms + N[2] - N[1], B + 2K / Ms + N[3] - N[1]
        em = eigenmodes(w, uniform(mesh, (1, 0, 0)); nmodes = 1)
        @test em.frequencies[1] ≈ _eγ / 2π * sqrt(a * b) rtol = 1e-12
        # exact damped eigenvalue: γ/(1+α²) [−α(a+b)/2 ± …], first order in α for Γ (exact here)
        @test em.linewidths[1] ≈ _eγ * 0.01 / (1 + 0.01^2) * (a + b) / 2 rtol = 1e-12
        # elliptical precession: |δy|/|δz| = √(b/a), δz = −i … sign: counterclockwise about +x
        δ = em.modes[1][:, 1, 1, 1]
        @test abs(δ[2]) / abs(δ[3]) ≈ sqrt(b / a) rtol = 1e-12
        @test imag(δ[3] / δ[2]) < 0                       # y → z: m(t) turns from y to z
        c = curvatures(w, uniform(mesh, (1, 0, 0)))
        @test c.values ≈ sort([a, b]) rtol = 1e-12
        @test em.residuals[1] < 1e-12
    end

    # One-cell-thick chain, periodic along x, no demag, m0 along the field (y):
    # Ω(k) = B + (2A/Ms)(2 − 2cos kc)/c² ± (2D/Ms) sin(kc)/c, with interfacial DMI (k ⟂ m).
    @testset "spin waves of a chain, $method, D = $D" for D in (0.0, 1e-3), method in (:dense, :lanczos)
        n = 12; c = 4e-9; Ms, A, B = 8e5, 1.3e-11, 0.5
        mesh = Mesh((n, 1, 1), (c, c, c); pbc = (1, 1, 0))
        w = World(mesh, Material(; Msat = Ms, Aex = A, alpha = 0.02, (D == 0 ? (;) : (; Dind = D))...); demag = false, Bext = (0, B, 0))
        k = 2π .* (0:n-1) ./ (n * c)
        Ω = @. B + 2A / Ms * (2 - 2cos(k * c)) / c^2 + 2D / Ms * sin(k * c) / c
        em = eigenmodes(w, uniform(mesh, (0, 1, 0)); nmodes = 7, method = method)
        @test em.frequencies ≈ sort(_eγ / 2π .* Ω)[1:7] rtol = 1e-9     # D = 0: ±k degenerate
        @test all(em.residuals .< 1e-6)
        @test length(em.modes) == 7
        if D != 0 && method == :dense                        # the direction of propagation
            q = argmin(Ω) - 1                                # phase advance per cell of the lowest mode
            δ = em.modes[1][1, :, 1, 1]
            ph = angle(δ[2] / δ[1])
            @test cos(ph) ≈ cos(2π * q / n) atol = 1e-6
        end
        if method == :lanczos
            @test em.method == :lanczos
            for seed in 1:12, nm in (5, 7, 12)               # any starting vector, all modes
                e = eigenmodes(w, uniform(mesh, (0, 1, 0)); nmodes = nm, method = :lanczos, rng = Random.MersenneTwister(seed))
                @test e.frequencies ≈ sort(_eγ / 2π .* Ω)[1:nm] rtol = 1e-9
            end
        end
    end

    @testset "Lanczos: highly degenerate and nearly complete spectra" begin
        mat = Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02)
        sq = Mesh((6, 6, 1), (4e-9, 4e-9, 4e-9); pbc = (1, 1, 0))          # up to 8-fold
        ω(n, c) = [2 * 1.3e-11 / 8e5 * (2 - 2cos(2π * q / n)) / c^2 for q in 0:n-1]
        Ω = sort(vec([0.3 + a + b for a in ω(6, 4e-9), b in ω(6, 4e-9)]))
        w = World(sq, mat; demag = false, Bext = (0, 0.3, 0))
        for seed in 1:5, nm in (9, 20, 36)
            e = eigenmodes(w, uniform(sq, (0, 1, 0)); nmodes = nm, method = :lanczos, rng = Random.MersenneTwister(seed))
            @test e.frequencies ≈ _eγ / 2π .* Ω[1:nm] rtol = 1e-9
        end
        bar = Mesh((6, 4, 1), (5e-9, 5e-9, 5e-9))                          # no degeneracy, demag
        wb = World(bar, mat; Bext = (0.3, 0, 0))
        mn = Minimizer(wb, uniform(bar, (1, 0.01, 0.01)); stopdm = 1e-11); minimize!(mn; maxsteps = 20000)
        d = eigenmodes(wb, mn.m; nmodes = 24)
        for seed in 1:3, nm in (6, 18, 24)
            e = eigenmodes(wb, mn.m; nmodes = nm, method = :lanczos, rng = Random.MersenneTwister(seed))
            @test e.frequencies ≈ d.frequencies[1:nm] rtol = 1e-8
            @test e.linewidths ≈ d.linewidths[1:nm] rtol = 1e-5
        end
    end

    # Two regions of different Msat, K and A next to empty cells, demag: the returned complex
    # field must satisfy the linearized LLG itself, iω δ = d/ds rhs(m0 + s δ), where the rhs is
    # llgrhs! (no Hessian, no scaling: checks the conversion to δm, conj, sign, frequency).
    @testset "modes satisfy the linearized LLG, $method" for method in (:dense, :lanczos)
        mesh = Mesh((8, 6, 1), (5e-9, 5e-9, 5e-9))
        rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.0))
        setregion!(rp, 1; Msat = 1.6e6, Aex = 2.2e-11, Ku = 1e5, anisU = (1, 0, 0))
        setregion!(rp, 2; Msat = 0.0)
        defregion!(rp, 1, (x, y, z) -> x > 0); defregion!(rp, 2, (x, y, z) -> x < -10e-9 && y > 5e-9)
        w = World(mesh, rp; Bext = (0.25, 0.05, 0.1))
        mn = Minimizer(w, clearempty!(uniform(mesh, (0.9, 0.1, 0.1) ./ sqrt(0.83)), rp); stopdm = 1e-12)
        minimize!(mn; maxsteps = 50000)
        m0 = copy(mn.m)
        em = eigenmodes(w, m0; nmodes = 5, method = method)
        rhs(m) = llgrhs!(similar(m), m, w)
        for k in 1:5
            δ = em.modes[k]; h = 1e-4
            D(v) = (rhs(m0 .+ h .* v) .- rhs(m0 .- h .* v)) ./ 2h
            lhs = im * 2π * em.frequencies[k] .* δ
            dF = complex.(D(real.(δ)), D(imag.(δ)))
            @test maximum(abs, lhs .- dF) < 1e-5 * maximum(abs, lhs)
            a = dropdims(sum(abs2, δ; dims = 1); dims = 1); I = argmax(a)
            @test a[I] ≈ 1 rtol = 1e-12
            q = argmax(abs.(δ[:, I]))
            @test abs(imag(δ[q, I])) < 1e-12 && real(δ[q, I]) > 0
            @test all(iszero, δ[:, JuliaMag._msatcells(rp, mesh)[1, :, :, :] .== 0])
        end
        dir = mktempdir(); savemode(dir, em, 2, mesh)
        @test maximum(abs, loadovf(joinpath(dir, "mode002_re.ovf"))[1] .- real.(em.modes[2])) < 1e-6
        @test maximum(abs, loadovf(joinpath(dir, "mode002_im.ovf"))[1] .- imag.(em.modes[2])) < 1e-6
        # strong damping enters as 1/(1+α²); both solvers
        e1 = eigenmodes(w, m0; nmodes = 5, method = method, alpha = 0.5); e0 = eigenmodes(w, m0; nmodes = 5, method = method, alpha = 1e-3)
        @test e1.linewidths ≈ e0.linewidths .* (0.5 / 1.25) / (1e-3 / (1 + 1e-6)) rtol = 1e-9
        @test all(0 .< em.residuals .< 1e-4)
    end

    # Two identical islands (Msat differs: equal frequencies need not share the linewidth).
    # Degenerate level: first-order linewidths are the eigenvalues of A² in the level.
    @testset "linewidths of degenerate modes, $method" for method in (:dense, :lanczos)
        mesh = Mesh((5, 1, 1), (4e-9, 4e-9, 4e-9))
        rp = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.01))
        setregion!(rp, 0; Msat = 0.0)
        # curvatures a, b per cell from the field (b) and a hard axis: Ω = √(ab) equal in both
        setregion!(rp, 1; Msat = 8e5, Aex = 1.3e-11, Ku = 8e4, anisU = (1, 0, 0))     # a = b = B + 2K/Ms
        setregion!(rp, 2; Msat = 8e5, Aex = 1.3e-11, Ku = -1.0e5, anisU = (0, 0, 1))  # hard z: a = B, b = B + 0.25
        defregioncell!(rp, 1, 1, 1, 1); defregioncell!(rp, 2, 4, 1, 1)
        B = 0.2; K2 = -1.0e5
        a1 = B + 2 * 8e4 / 8e5                                            # region 1: 0.4
        # choose Ku of region 2 so that √(B (B + κ)) = a1
        κ = a1^2 / B - B
        setregion!(rp, 2; Ku = -κ * 8e5 / 2)
        w = World(mesh, rp; demag = false, Bext = (B, 0, 0))
        m0 = clearempty!(uniform(mesh, (1, 0, 0)), rp)
        em = eigenmodes(w, m0; nmodes = 2, method = method)
        @test em.frequencies ≈ fill(_eγ / 2π * a1, 2) rtol = 1e-9
        @test em.linewidths ≈ sort([_eγ * 0.01 / (1 + 1e-4) * s for s in (a1, (2B + κ) / 2)]) rtol = 1e-7
        # the modes returned are the rotated ones: each lives on one cell (the degenerate pair is
        # separated by its linewidth, region 1 has the smaller one)
        for (k, cell) in ((1, 1), (2, 4))
            other = cell == 1 ? 4 : 1
            @test norm(em.modes[k][:, other, 1, 1]) < 1e-6 && norm(em.modes[k][:, cell, 1, 1]) ≈ 1
        end
    end

    @testset "damping to first order against the general eigenproblem" begin
        mesh = Mesh((6, 4, 1), (5e-9, 5e-9, 5e-9))
        α = 1e-4
        w = World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = α); Bext = (0.3, 0, 0))
        mn = Minimizer(w, uniform(mesh, (1, 0.01, 0.01)); stopdm = 1e-11); minimize!(mn; maxsteps = 20000)
        em = eigenmodes(w, mn.m; nmodes = 8)
        A, = densehessian(TangentHessian(w, mn.m))
        JA = JuliaMag._Jrows!(similar(A), A)
        ev = eigvals(_eγ / (1 + α^2) .* (JA .- α .* A))
        ev = sort(ev[imag.(ev).>0]; by = imag)[1:8]
        @test imag.(ev) ./ 2π ≈ em.frequencies rtol = 1e-6
        @test -real.(ev) ≈ em.linewidths rtol = 1e-5
        el = eigenmodes(w, mn.m; nmodes = 8, method = :lanczos)
        @test el.frequencies ≈ em.frequencies rtol = 1e-9
        @test el.linewidths ≈ em.linewidths rtol = 1e-6
        for k in 1:8                                         # same modes up to a phase
            ov = abs(dot(vec(em.modes[k]), vec(el.modes[k]))) / norm(em.modes[k]) / norm(el.modes[k])
            @test ov ≈ 1 atol = 1e-8
        end
        # the general solver (used for non-symmetric fields) on the same problem
        JAg = eigvals(JA)
        @test sort(imag.(JAg[imag.(JAg).>0]))[1:8] .* _eγ ./ 2π ≈ em.frequencies rtol = 1e-9
    end

    @testset "edge-smoothed disc: general solver" begin
        mesh = Mesh((8, 8, 1), (4e-9, 4e-9, 4e-9))
        rs = RegionParams(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02))
        setregion!(rs, 0; Msat = 0.0); setregion!(rs, 1; Msat = 8e5, Aex = 1.3e-11)
        setgeometry!(rs, Cylinder(30e-9, 1.0); id = 1, edgesmooth = 4)
        w = World(mesh, rs; Bext = (0.5, 0, 0))
        mn = Minimizer(w, clearempty!(uniform(mesh, (1, 0.01, 0.01)), rs); stopdm = 1e-11); minimize!(mn; maxsteps = 20000)
        em = eigenmodes(w, mn.m; nmodes = 3)
        @test em.method == :densegeneral
        @test all(em.residuals .< 1e-9) && issorted(em.frequencies) && em.frequencies[1] > 1e9
        @test all(em.linewidths .> 0)
        A, = densehessian(TangentHessian(w, mn.m))             # not symmetric: A is the damped operator's
        JA = JuliaMag._Jrows!(similar(A), A)
        α = 1e-4
        ev = eigvals(_eγ / (1 + α^2) .* (JA .- α .* A)); ev = sort(ev[imag.(ev).>0]; by = imag)[1:3]
        e2 = eigenmodes(w, mn.m; nmodes = 3, alpha = α)
        @test imag.(ev) ./ 2π ≈ e2.frequencies rtol = 1e-6
        @test -real.(ev) ≈ e2.linewidths rtol = 1e-4
        bad = World(mesh, rs; Bext = (0, 0, -6))                   # m along +z against the field
        up = clearempty!(uniform(mesh, (0, 0, 1)), rs)
        for nm in (1, 2, 6)                                        # not only among the modes returned
            @test_throws ErrorException eigenmodes(bad, up; nmodes = nm)
        end
        @test_throws ErrorException curvatures(w, mn.m)
        @test_throws ErrorException eigenmodes(w, mn.m; method = :lanczos)
    end

    @testset "errors and warnings" begin
        mesh = Mesh((1, 1, 1), (4e-9, 4e-9, 4e-9))
        w = World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02, Ku = 5e5, anisU = (0, 0, 1)); demag = false)
        hard = uniform(mesh, (1, 0, 0))                      # a maximum: not positive definite
        @test_throws ErrorException eigenmodes(w, hard; method = :dense)
        @test_throws ErrorException eigenmodes(w, hard; method = :lanczos)
        tilted = uniform(mesh, (0.1, 0, 1))
        @test_logs (:warn, r"not an equilibrium") eigenmodes(w, normalize!(tilted); nmodes = 1)
        @test_throws ErrorException TangentHessian(w, 2 .* hard)
        @test_throws ErrorException curvatures(w, hard; method = :foo)
        @test_throws ErrorException curvatures(w, hard; foo = 1)
        @test_throws ErrorException eigenmodes(w, uniform(mesh, (0, 0, 1)); foo = 1)
        # no energy at all: H = 0 is symmetric, and singular
        free = World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02); demag = false)
        @test curvatures(free, hard).values == [0, 0]
        @test_throws ErrorException eigenmodes(free, hard)
        # a zero mode (easy plane: rotation about z) stops both solvers with the reason
        ch = Mesh((6, 1, 1), (4e-9, 4e-9, 4e-9); pbc = (1, 0, 0))
        wz = World(ch, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02, Ku = -4e5, anisU = (0, 0, 1)); demag = false)
        for a in range(0, π; length = 40), meth in (:dense, :lanczos)     # rounding decides the branch
            err = try eigenmodes(wz, uniform(ch, (cos(a), sin(a), 0)); nmodes = 3, method = meth, cgmaxiter = 200); "" catch e; sprint(showerror, e) end
            @test occursin("singular", err)
        end
        # torque over magnetic cells only, whatever m is in the empty ones
        rq = RegionParams(Mesh((3, 1, 1), (4e-9, 4e-9, 4e-9)), w.material); setregion!(rq, 1; Msat = 0.0); defregioncell!(rq, 1, 3, 1, 1)
        wq = World(rq.regions.mesh, rq; demag = false, Bext = (0.3, 0, 0)); mq = uniform(rq.regions.mesh, (0, 0, 1))
        Bq = effectivefield!(similar(mq), clearempty!(copy(mq), rq), wq); mq[:, 3, 1, 1] .= (0, 5, 7)
        Hq = TangentHessian(wq, mq)
        @test Hq.torque ≈ maximum(norm(cross(mq[:, i, 1, 1], Bq[:, i, 1, 1])) for i in 1:2) rtol = 1e-12
        wst = World(mesh, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02, Ku = 5e5, anisU = (0, 0, 1), pol = 0.4); demag = false); setcurrent!(wst, (0, 0, 1e12); zhangli = false, slonczewski = true, thickness = 4e-9)
        @test_logs (:warn, r"spin-transfer") eigenmodes(wst, uniform(mesh, (0, 0, 1)); nmodes = 1)
        # |m| off by 1e-7 (single-precision file): not a false "not a minimum"
        bar = Mesh((6, 4, 2), (5e-9, 5e-9, 5e-9))
        wb = World(bar, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02); Bext = (0.1, 0, 0))
        mn = Minimizer(wb, uniform(bar, (1, 0.01, 0.01)); stopdm = 1e-11); minimize!(mn; maxsteps = 20000)
        d = eigenmodes(wb, mn.m; nmodes = 4)
        @test_throws ErrorException eigenmodes(wb, mn.m; nmodes = 2, restol = 1e-30)
        m1 = Float64.(Float32.(mn.m)) .* (1 + 1e-7)
        e1 = eigenmodes(wb, m1; nmodes = 4, method = :lanczos)
        @test e1.frequencies ≈ d.frequencies rtol = 1e-5
        c1 = curvatures(wb, m1; nev = 3, method = :lobpcg, tol = 1e-9)
        @test c1.values ≈ curvatures(wb, m1; nev = 3).values atol = 1e-9
        @test all(c1.residuals .< 1e-9)
        # single precision: all solvers, default tolerances
        w32 = World(bar, Material{Float32}(wb.material); Bext = (0.1, 0, 0))
        m32 = Float32.(mn.m)
        @test eigenmodes(w32, m32; nmodes = 4).frequencies ≈ d.frequencies rtol = 1e-4
        @test eigenmodes(w32, m32; nmodes = 4, method = :lanczos).frequencies ≈ d.frequencies rtol = 1e-4
        @test curvatures(w32, m32; nev = 3).values ≈ curvatures(wb, mn.m; nev = 3).values rtol = 1e-4
        @test curvatures(w32, m32; nev = 3, method = :lobpcg).values ≈ curvatures(wb, mn.m; nev = 3).values rtol = 1e-3
        # eigenvectors and residuals of both curvature solvers
        H = TangentHessian(wb, mn.m)
        for meth in (:dense, :lobpcg)
            c = curvatures(wb, mn.m; nev = 3, method = meth)
            for k in 1:3
                v = c.vectors[k]
                r = norm(hessian!(similar(v), v, H) .- c.values[k] .* v)
                @test norm(v) ≈ 1 rtol = 1e-10
                @test r < 2e-6 && abs(r - c.residuals[k]) < 1e-7 + 1e-2 * r
            end
        end
        sim = Simulation(bar, wb.material; Bext = (0.1, 0, 0), verbose = false)
        copyto!(sim.m, mn.m)
        @test eigenmodes(sim; nmodes = 2).frequencies ≈ d.frequencies[1:2] rtol = 1e-12
        # torque of a state that is not stationary
        tl = normalize!(uniform(bar, (1, 0.2, 0.1))); Bt = effectivefield!(similar(tl), tl, wb)
        @test TangentHessian(wb, tl).torque ≈ maximum(norm(cross(tl[:, I], Bt[:, I])) for I in CartesianIndices(axes(tl)[2:4])) rtol = 1e-12
        em = eigenmodes(w, uniform(mesh, (0, 0, 1)); nmodes = 5)
        @test length(em.frequencies) == 1                    # one cell, one mode
        @test em.frequencies[1] ≈ _eγ / 2π * 2 * 5e5 / 8e5 rtol = 1e-12
        dir = mktempdir(); savemode(dir, em, 1, mesh)
        @test isfile(joinpath(dir, "mode001_re.ovf")) && isfile(joinpath(dir, "mode001_im.ovf"))
    end

    # Stoner–Wohlfarth: E/V = −K cos²θ + 2Kh cosθ. Minimum at θ = 0: curvature 2K(1 − h)/Ms
    # (twice). Saddle at cosθ = h: −2K(1 − h²)/Ms along the path, 0 around the axis.
    @testset "saddle certification: Stoner–Wohlfarth, h = $h" for h in (0.0, 0.3, 0.6)
        Ms, K = 8e5, 5e5
        mesh = Mesh((1, 1, 1), (4e-9, 4e-9, 4e-9))
        w = World(mesh, Material(Msat = Ms, Aex = 1.3e-11, alpha = 0.5, Ku = K, anisU = (0, 0, 1)); demag = false,
                  Bext = (0, 0, -2K * h / Ms))
        neb = GNEB(w, uniform(mesh, (0, 0, 1)), uniform(mesh, (0, 0, -1)); nimages = 12)
        relax!(neb; stopforce = 1e-9)
        s = certifysaddle(neb)
        @test s.index == 1 && s.nzero == 1
        @test s.values[1] ≈ -2K * (1 - h^2) / Ms rtol = 1e-7
        @test abs(s.values[2]) < 1e-8
        @test s.along ≈ s.values[1] rtol = 1e-7
        @test s.torque < 1e-8 && s.image == neb.climb
        c = curvatures(w, uniform(mesh, (0, 0, 1)))
        @test c.values ≈ fill(2K * (1 - h) / Ms, 2) rtol = 1e-12
        e = certifysaddle(neb; image = 1)
        @test e.index == 0 && e.nzero == 0
    end

    @testset "saddle certification: wall nucleation in a wire" begin
        Ms, K, A = 8e5, 5e5, 1.3e-11
        mesh = Mesh((32, 1, 1), (2e-9, 2e-9, 2e-9))
        h = 0.3
        w = World(mesh, Material(Msat = Ms, Aex = A, alpha = 0.5, Ku = K, anisU = (1, 0, 0)); demag = false,
                  Bext = (-2K * h / Ms, 0, 0))
        mw = zeromag(Float64, mesh)
        for i in 1:32
            θ = π / (1 + exp(-(i - 16) / 2.5))
            mw[:, i, 1, 1] .= (cos(θ), sin(θ), 0.02)
        end
        neb = GNEB(w, [uniform(mesh, (1, 0, 0)), normalize!(mw), uniform(mesh, (-1, 0, 0))]; nimages = 13)
        relax!(neb; stopforce = 1e-8)
        s = certifysaddle(neb)                               # nucleation at the end of the wire
        @test s.index == 1 && s.nzero == 1                   # rotation about the axis
        @test s.values[1] < -0.05 && s.values[3] > 0.01
        l = certifysaddle(neb; method = :lobpcg, nev = 4, tol = 1e-9)
        @test l.values ≈ s.values[1:4] atol = 1e-8
        @test l.method == :lobpcg && l.index == 1
        # More negative values than nev: the index is not truncated
        tr = GNEB(World(mesh, w.material; demag = false), [uniform(mesh, (1, 0, 0)), uniform(mesh, (0, 1, 0)), uniform(mesh, (-1, 0, 0))]; nimages = 3)
        st = certifysaddle(tr; image = 2, nev = 2)
        @test st.torque ≈ TangentHessian(tr.world, tr.m[2]).torque
        @test certifysaddle(tr; image = 2, nev = 2, method = :lobpcg, tol = 1e-7).index == st.index
        c = 2e-9; k = π .* (0:31) ./ (32c)
        @test st.index == count(<(0), [-2K / Ms + 2A / Ms * (2 - 2cos(q * c)) / c^2 for q in k]) > 2
        # regions of different Msat: the curvature along the path against second differences of E
        r6 = RegionParams(mesh, w.material)
        setregion!(r6, 1; Msat = 4e5, Ku = K)
        defregion!(r6, 1, (x, y, z) -> x > 6e-9)
        wr = World(mesh, r6; demag = false, Bext = (-0.2, 0, 0))
        nr = GNEB(wr, [uniform(mesh, (1, 0, 0)), mw, uniform(mesh, (-1, 0, 0))]; nimages = 9)
        relax!(nr; stopforce = 1e-6)
        sr = certifysaddle(nr)
        i = sr.image; t6 = copy(nr.t[i]); m6 = nr.m[i]; H6 = TangentHessian(wr, m6)
        E(s) = totalenergy(_geodesic(m6, s .* t6), wr)
        hh = 0.02; cc = (-49 / 18, 3 / 2, -3 / 20, 1 / 90)
        fd = (cc[1] * E(0.0) + sum(cc[q+1] * (E(q * hh) + E(-q * hh)) for q in 1:3)) / hh^2
        u6 = H6.sw .* t6
        @test fd ≈ cellvolume(mesh) * H6.Msref * dot(u6, u6) * sr.along rtol = 1e-5
        @test sr.index == 1 && sr.nzero == 1 && sr.values[1] <= sr.along
        # The straight path over the uniform transverse state is stationary but not a first-
        # order saddle: 64 nm wire, wall width 5 nm, many unstable modes −2K/Ms + exchange.
        t = curvatures(World(mesh, w.material; demag = false), uniform(mesh, (0, 1, 0)); nev = 12)
        # one tangent direction along the axis (unstable), one around it (neutral): free-end modes
        an = sort(vcat([-2K / Ms + 2A / Ms * (2 - 2cos(q * c)) / c^2 for q in k], [2A / Ms * (2 - 2cos(q * c)) / c^2 for q in k]))
        @test t.values ≈ an[1:12] atol = 1e-9
        @test count(<(-1e-6), t.values) == count(<(-1e-6), an)
    end
end
