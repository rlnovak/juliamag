# Geodesic nudged elastic band. Exact references: the Stoner–Wohlfarth macrospin
# barrier K V (1 − h)² with the saddle at m_z = h; a flat demagnetizing cell whose
# minimum-energy path lies in the plane (exact barrier by 1D maximization of the
# energy along the in-plane great circle), reached from an out-of-plane initial path;
# and, for a non-uniform reversal (domain-wall nucleation in a wire), independence
# of the barrier from the spring constant and the number of images, and the wall
# energy 4√(AK)·S at zero field.
using JuliaMag, Test

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
            @test barrier(_wire(Float64; spring = k, nimages = n)[1]) ≈ ΔE rtol = 1e-8
        end
        n32 = _wire(Float32; stopforce = 1e-4)[1]
        @test barrier(n32) ≈ ΔE rtol = 1e-5
        # zero field: the saddle is a wall in the middle, energy 4√(AK)·S
        n0 = _wire(Float64; h = 0.0)[1]
        @test barrier(n0) ≈ 4sqrt(1.3e-11 * _nK) * (2e-9)^2 rtol = 0.02
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
