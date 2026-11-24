# Geodesic nudged elastic band. Exact references: the Stoner–Wohlfarth macrospin
# barrier K V (1 − h)² with the saddle at m_z = h; a flat demagnetizing cell whose
# minimum-energy path lies in the plane (exact barrier by 1D maximization of the
# energy along the in-plane great circle), reached from an out-of-plane initial path;
# and, for a non-uniform reversal (domain-wall nucleation in a wire), independence
# of the barrier from the spring constant and the number of images, and the wall
# energy 4√(AK)·S at zero field.
using JuliaMag, Test
import Random

const _nMs, _nK = 8e5, 5e5

_nmat(T = Float64; K = _nK, u = (0, 0, 1)) =
    Material{T}(Material(Msat = _nMs, Aex = 1.3e-11, alpha = 0.5, Ku = K, anisU = u))

_normerr(neb) = maximum(maximum(abs.(sqrt.(sum(abs2, m; dims = 1)) .- 1)) for m in images(neb))

function _wire(T; h = 0.3, spring = 1.0, nimages = 13, stopforce = 1e-6)
    mesh = Mesh((32, 1, 1), (2e-9, 2e-9, 2e-9))
    w = World(mesh, _nmat(T; u = (1, 0, 0)); demag = false, Bext = (-2_nK * h / _nMs, 0, 0))
    mw = zeromag(T, mesh)                          # a wall in the middle as waypoint
    for i in 1:32
        θ = π / (1 + exp(-(i - 16) / 2.5))
        mw[:, i, 1, 1] .= (cos(θ), sin(θ), 0.02)
    end
    normalize!(mw)
    mA, mB = uniform(T, mesh, (1, 0, 0)), uniform(T, mesh, (-1, 0, 0))
    neb = GNEB(w, [mA, mw, mB]; nimages = nimages, spring = spring)
    relax!(neb; stopforce = stopforce)
    return neb, mesh, mA, mB
end

@testset "GNEB" begin
    @testset "Stoner–Wohlfarth macrospin, h = $h" for h in (0.0, 0.2, 0.5, 0.8)
        mesh = Mesh((1, 1, 1), (4e-9, 4e-9, 4e-9))
        w = World(mesh, _nmat(); demag = false, Bext = (0, 0, -2_nK * h / _nMs))
        neb = GNEB(w, uniform(mesh, (0, 0, 1)), uniform(mesh, (0, 0, -1)); nimages = 12)
        relax!(neb; stopforce = 1e-7)
        @test neb.maxforce < 1e-7
        @test barrier(neb) ≈ _nK * volume(mesh) * (1 - h)^2 rtol = 1e-9
        @test images(neb)[neb.climb][3, 1, 1, 1] ≈ h atol = 1e-6
        @test _normerr(neb) < 1e-12
    end

    # Flat cell, easy axis x, demag: hard axis z, so the path is in-plane. The initial
    # path goes over an out-of-plane waypoint and must relax into the plane.
    mesh = Mesh((1, 1, 1), (4e-9, 4e-9, 1e-9))
    h = 0.3
    K2 = 2e5
    w = World(mesh, _nmat(; K = K2, u = (1, 0, 0)); demag = true, Bext = (-2K2 * h / _nMs, 0, 0))
    Eθ(θ) = totalenergy(uniform(mesh, (cos(θ), sin(θ), 0)), w)
    θs = range(0, π, length = 2001); i = argmax(Eθ.(θs))
    a, b = θs[i-1], θs[i+1]
    for _ in 1:100                                   # golden-section refinement
        c = b - (b - a) / MathConstants.golden; d = a + (b - a) / MathConstants.golden
        Eθ(c) > Eθ(d) ? (b = d) : (a = c)
    end
    exact = Eθ((a + b) / 2) - Eθ(0.0)
    wp = (0.3, 0.5, 0.8) ./ sqrt(0.98)
    states = [uniform(mesh, (1, 0, 0)), uniform(mesh, wp), uniform(mesh, (-1, 0, 0))]
    @testset "demag cell, out-of-plane initial path, spring $k" for k in (0.1, 1.0, 10.0)
        neb = GNEB(w, states; nimages = 15, spring = k)
        relax!(neb; stopforce = 1e-7)
        @test neb.maxforce < 1e-7
        @test barrier(neb) ≈ exact rtol = 1e-9
        mc = images(neb)[neb.climb]
        @test abs(mc[3, 1, 1, 1]) < 1e-6
        @test mc[1, 1, 1, 1] ≈ cos((a + b) / 2) atol = 1e-6
        @test saddle(neb)[2] - energies(neb)[1] ≈ exact rtol = 1e-9
    end
    @testset "Hermite saddle estimate without climbing" begin
        neb = GNEB(w, states[1], states[3]; nimages = 9)
        relax!(neb; stopforce = 1e-7, climbing = false)
        @test neb.climb == 0
        eimg = abs(barrier(neb) / exact - 1)
        eher = abs((saddle(neb)[2] - energies(neb)[1]) / exact - 1)
        @test eher < 1e-3
        @test eher < eimg / 10
        s = pathcoordinate(neb)
        @test s[1] == 0 && issorted(s)
        @test s[end] ≈ π rtol = 1e-6                      # half a great circle
    end

    @testset "non-uniform reversal: domain-wall nucleation in a wire" begin
        neb, mesh, mA, mB = _wire(Float64)
        ΔE = barrier(neb)
        @test neb.maxforce < 1e-6
        @test 0 < ΔE < _nK * volume(mesh) * 0.7^2 / 2    # far below coherent rotation
        @test images(neb)[1] == mA && images(neb)[end] == mB   # end images fixed
        @test _normerr(neb) < 1e-12
        mc = images(neb)[neb.climb]                      # nucleated at an end
        @test mc[1, 1, 1, 1] > 0.99 && mc[1, end, 1, 1] < 0.5
        for (k, n) in ((0.3, 13), (3.0, 13), (1.0, 21))
            nk = _wire(Float64; spring = k, nimages = n)[1]
            @test nk.maxforce < 1e-6
            @test barrier(nk) ≈ ΔE rtol = 1e-8
        end
        n32 = _wire(Float32; stopforce = 1e-4)[1]
        @test barrier(n32) ≈ ΔE rtol = 1e-5
        @test _normerr(n32) < 1e-4
        # relaxing a converged band again changes nothing; climbing off un-sets the image
        steps = neb.steps
        relax!(neb; stopforce = 1e-6)
        @test neb.steps == steps && barrier(neb) == ΔE
        relax!(neb; stopforce = 1e-4, climbing = false)
        @test neb.climb == 0
        # a user-chosen climbing image is kept; an automatic one follows the top
        setclimbing!(neb, 11)
        @test neb.climb == 11 && !neb.climbauto
        neb.climbauto = true
        relax!(neb; stopforce = 1e-6)
        @test neb.maxforce < 1e-6
        @test neb.climb == argmax(energies(neb))
        @test barrier(neb) ≈ ΔE rtol = 1e-8
        # a step that raises the force is undone with a finite `reject`
        nr = relax!(GNEB(neb.world, [mA, images(neb)[7], mB]; nimages = 13);
                    stopforce = 1e-6, reject = 1.25)
        @test nr.maxforce < 1e-6 && isfinite(nr.dtmax) && nr.rejected > 0
        @test barrier(nr) ≈ ΔE rtol = 1e-8
        # zero field: the saddle is a wall in the middle, energy 4√(AK)·S
        # a steep descent whose barrier falls between the first images: no image is above
        # the ends, and the midpoints show it. With more images the barrier is there.
        ns = @test_logs((:warn, r"no free image"), (:warn, r"halfway between images"),
                        _wire(Float64; h = 0.8, nimages = 7)[1])
        @test ns.climb == 0 && barrier(ns) == 0
        ns = @test_logs _wire(Float64; h = 0.8, nimages = 21)[1]
        @test barrier(ns) > 0
        # (its midpoints are 0.6 % above the saddle, which is not a hidden maximum)
        n0 = @test_logs _wire(Float64; h = 0.0)[1]
        @test n0.maxforce < 1e-6
        @test barrier(n0) ≈ 4sqrt(1.3e-11 * _nK) * (2e-9)^2 rtol = 0.02
    end

    # Float32 nearly antiparallel states: the initial path must stay on the sphere
    @testset "initial path is normalized, perturbation $ε" for ε in (1f-6, 1f-5, 1f-4, 1f-3)
        m1 = Mesh((8, 8, 1), (2e-9, 2e-9, 2e-9))
        rng = Random.MersenneTwister(5)
        pm = [normalize!(uniform(Float32, m1, (0, 0, s)) .+ ε .* randn(rng, Float32, 3, 8, 8, 1)) for s in (1, -1)]
        w32 = World(m1, _nmat(Float32); demag = false)
        @test _normerr(GNEB(w32, pm[1], pm[2]; nimages = 9)) < 1e-6
    end

    # Two regions of different Msat with the same K/Msat, next to empty cells, coupled by
    # exchange: a uniform state rotates coherently, so ΔE = (1 − h)² Σ K V exactly. The
    # forces are weighted by Msat(cell)/Msref, which no other case exercises.
    @testset "regions with different Msat and empty cells" begin
        m6 = Mesh((6, 2, 1), (2e-9, 2e-9, 2e-9))
        r6 = RegionParams(m6, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.5))
        setregion!(r6, 0; Msat = 0.0)
        setregion!(r6, 1; Msat = 8e5, Aex = 1.3e-11, Ku = 4e5, anisU = (0, 0, 1))
        setregion!(r6, 2; Msat = 4e5, Aex = 1.3e-11, Ku = 2e5, anisU = (0, 0, 1))
        for j in 1:2, i in 1:4
            defregioncell!(r6.regions, i <= 2 ? 1 : 2, i, j, 1)
        end
        hh = 0.3
        w6 = World(m6, r6; demag = false, Bext = (0, 0, -2 * 4e5 * hh / 8e5))
        up = clearempty!(uniform(m6, (0, 0, 1)), r6)
        tilt = clearempty!(uniform(m6, (0.8, 0.36, 0.48)), r6)
        dn = clearempty!(uniform(m6, (0, 0, -1)), r6)
        nr = GNEB(w6, [up, tilt, dn]; nimages = 13)
        relax!(nr; stopforce = 1e-8)
        @test nr.maxforce < 1e-8
        @test barrier(nr) ≈ cellvolume(m6) * 4 * (4e5 + 2e5) * (1 - hh)^2 rtol = 1e-9
        # 8 magnetic cells: length ≥ π√8. Not equal to it: the forces are ∝ Msat, so away
        # from the saddle the descent path of the two regions is not coherent.
        @test π * sqrt(8) < pathcoordinate(nr)[end] < 1.001 * π * sqrt(8)
        @test all(iszero, images(nr)[7][:, 5:6, :, :])             # empty cells untouched
        @test all(isfinite, midpointenergies(nr))
        @test maximum(midpointenergies(nr)) < maximum(energies(nr))
    end

    @testset "no interior maximum; changed field; unnormalized states" begin
        m1 = Mesh((1, 1, 1), (4e-9, 4e-9, 4e-9))
        w1 = World(m1, _nmat(); demag = false, Bext = (0, 0, -2_nK * 1.5 / _nMs))   # h > 1
        n1 = GNEB(w1, uniform(m1, (0, 0, 1)), uniform(m1, (0, 0, -1)); nimages = 9)
        @test_logs (:warn, r"no free image") match_mode = :any relax!(n1; stopforce = 1e-7)
        @test n1.climb == 0 && barrier(n1) == 0
        setexternalfield!(w1, (0, 0, 0))
        relax!(n1; stopforce = 1e-7)
        @test barrier(n1) ≈ _nK * volume(m1) rtol = 1e-9
        @test_throws ErrorException GNEB(w1, 2 .* uniform(m1, (0, 0, 1)), uniform(m1, (0, 0, -1)))
    end

    # A maximum that falls between two images: a band whose only free image sits next to
    # the first one, θ = 0.2, so the midpoint of the next segment is at (0.2 + π)/2.
    @testset "hidden maximum between images, $T" for T in (Float64, Float32)
        m1 = Mesh((2, 1, 1), (4e-9, 4e-9, 4e-9))
        w1 = World(m1, _nmat(T); demag = false)
        up, dn = uniform(T, m1, (0, 0, 1)), uniform(T, m1, (0, 0, -1))
        KV = _nK * 2 * cellvolume(m1)
        nh = GNEB(w1, [up, uniform(T, m1, (sin(0.2), 0, cos(0.2))), dn]; nimages = 3)
        e = midpointenergies(nh)
        @test e .- nh.E[1] ≈ KV .* sin.([0.1, (0.2 + π) / 2]) .^ 2 rtol = (T == Float64 ? 1e-12 : 1e-5)
        @test_logs (:warn, r"did not converge") (:warn, r"halfway between images 2 and 3") relax!(nh; stopforce = 1e-7, maxsteps = 0)
        @test_logs (:warn, r"did not converge") relax!(nh; stopforce = 1e-7, maxsteps = 0, climbing = false)
        @test_throws ErrorException relax!(nh; reject = 1)
        # a converged climbing band: no warning, and the call leaves the band as it was
        ng = GNEB(w1, up, dn; nimages = 12)
        tol = T == Float64 ? 1e-7 : 1e-4
        @test_logs relax!(ng; stopforce = tol)
        @test maximum(midpointenergies(ng)) < maximum(energies(ng))
        before = (images(ng), copy(ng.E), ng.maxforce)
        midpointenergies(ng)
        JuliaMag._update!(ng)
        @test (images(ng), ng.E, ng.maxforce) == before
        # a cell antiparallel between neighbours has no midpoint: its value in image i is kept
        q = copy(up); q[:, 2, 1, 1] .= (1, 0, 0); q[:, 1, 1, 1] .= (0, 0, -1)
        na = GNEB(w1, [up, q, dn]; nimages = 3)
        x1 = copy(up); x1[:, 2, 1, 1] .= (1, 0, 1) ./ sqrt(T(2))
        @test midpointenergies(na)[1] ≈ totalenergy(x1, w1) rtol = 1e-5
    end

    @testset "warnings, errors, output" begin
        m4 = Mesh((12, 12, 1), (4e-9, 4e-9, 4e-9))
        rs = RegionParams(m4, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02))
        setregion!(rs, 0; Msat = 0.0); setregion!(rs, 1; Msat = 8e5, Aex = 1.3e-11)
        setgeometry!(rs, Cylinder(44e-9, 1.0); id = 1, edgesmooth = 4)
        ws = World(m4, rs; demag = false)
        @test !isconservative(ws)
        @test_logs (:warn, r"not the exact") GNEB(ws, uniform(m4, (1, 0, 0)), uniform(m4, (0, 1, 0)); nimages = 3)
        @test_throws ErrorException GNEB(w, states[1], states[3]; nimages = 2)
        neb = GNEB(w, states[1], states[3]; nimages = 5)
        dir = mktempdir()
        saveimages(neb, dir)
        @test all(isfile(joinpath(dir, "image$(lpad(i, 3, '0')).ovf")) for i in 0:4)
        @test length(readlines(joinpath(dir, "neb.txt"))) == 6
        m3 = loadovf(joinpath(dir, "image002.ovf"))
        @test maximum(abs, m3[1] .- images(neb)[3]) < 1e-6      # OVF may store Float32
    end
end
