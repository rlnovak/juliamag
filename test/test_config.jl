using LinearAlgebra: norm
using Random: MersenneTwister

@testset "Initial configurations" begin
    mesh = Mesh((41, 41, 1), (4e-9, 4e-9, 4e-9))   # odd → a cell at the centre
    ic, jc = 21, 21                                # centre cell indices

    isnorm(m; atol = 1e-12) = all(CartesianIndices(axes(m)[2:4])) do I
        abs(norm(@view m[:, I]) - 1) <= atol
    end

    @testset "uniform" begin
        m = setconfig(mesh, UniformConfig(3, 4, 0))
        @test isnorm(m)
        @test all(average(m) .≈ (0.6, 0.8, 0.0))
    end

    @testset "vortex: circulation, core polarity, and placement" begin
        m = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1))
        @test isnorm(m)
        @test m[3, ic, jc, 1] > 0.9                 # core points +z
        @test average(m)[3] > 0                     # net core is +z
        # In-plane part circulates: at a cell +x of centre, m points +y (circ=1).
        @test m[2, ic + 5, jc, 1] > 0.5
        @test abs(m[1, ic + 5, jc, 1]) < 0.2

        # pol = -1 flips the core.
        @test setconfig(mesh, VortexConfig(mesh; pol = -1))[3, ic, jc, 1] < -0.9
        # circ = -1 reverses the in-plane sense.
        mrev = setconfig(mesh, VortexConfig(mesh; circ = -1))
        @test mrev[2, ic + 5, jc, 1] < -0.5

        # Placement: move the core to a chosen cell with translate.
        off = 8 * 4e-9
        mp = setconfig(mesh, translate(VortexConfig(mesh), off, 0, 0))
        @test mp[3, ic + 8, jc, 1] > 0.9            # core now at +8 cells in x
        @test mp[3, ic, jc, 1] < mp[3, ic + 8, jc, 1]
    end

    @testset "antivortex differs from vortex in the in-plane winding" begin
        mv = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1))
        ma = setconfig(mesh, AntiVortexConfig(mesh; circ = 1, pol = 1))
        @test isnorm(ma)
        @test ma[3, ic, jc, 1] > 0.9                # same core
        # Antivortex mx = -x·circ/r (vortex has mx = -y·circ/r): at +x of centre
        # the antivortex has mx < 0 while the vortex has mx ≈ 0.
        @test ma[1, ic + 5, jc, 1] < -0.5
    end

    @testset "Néel skyrmion: radial in-plane, opposite core/rim" begin
        m = setconfig(mesh, NeelSkyrmionConfig(mesh; charge = 1, pol = 1))
        @test isnorm(m)
        @test m[3, ic, jc, 1] > 0.9                 # core +z (pol=1)
        @test m[3, 1, 1, 1] < 0                      # rim points -z
        # Néel: in-plane component is radial (along +x at a cell +x of centre).
        @test m[1, ic + 6, jc, 1] > 0.3
        @test abs(m[2, ic + 6, jc, 1]) < 0.2

        @test setconfig(mesh, NeelSkyrmionConfig(mesh; pol = -1))[3, ic, jc, 1] < -0.9
    end

    @testset "Bloch skyrmion: tangential in-plane" begin
        m = setconfig(mesh, BlochSkyrmionConfig(mesh; charge = 1, pol = 1))
        @test isnorm(m)
        @test m[3, ic, jc, 1] > 0.9
        @test m[3, 1, 1, 1] < 0
        # Bloch: in-plane is tangential (along +y at a cell +x of centre).
        @test m[2, ic + 6, jc, 1] > 0.3
        @test abs(m[1, ic + 6, jc, 1]) < 0.2
    end

    @testset "two-domain: correct domains and a smoothed wall" begin
        m = setconfig(mesh, TwoDomainConfig(mesh, (1, 0, 0), (0, 1, 0), (-1, 0, 0)))
        @test isnorm(m)
        @test m[1, 1, jc, 1] ≈ 1.0 atol = 0.05       # left domain +x
        @test m[1, 41, jc, 1] ≈ -1.0 atol = 0.05     # right domain -x
        @test m[2, ic, jc, 1] > 0.9                  # wall centre points +y
    end

    @testset "vortex wall: uniform domains, vortex in the middle" begin
        # VortexWall places the uniform domains beyond ±Ly/2 in x, so the strip
        # must be longer in x than in y for the domains to appear at the ends.
        lmesh = Mesh((81, 21, 1), (4e-9, 4e-9, 4e-9))
        lic, ljc = 41, 11
        m = setconfig(lmesh, VortexWallConfig(lmesh, 1, -1; circ = 1, pol = 1))
        @test all(CartesianIndices(axes(m)[2:4])) do I
            abs(norm(@view m[:, I]) - 1) <= 1e-12
        end
        @test m[1, 1, ljc, 1] ≈ 1.0 atol = 1e-6      # left domain +x
        @test m[1, 81, ljc, 1] ≈ -1.0 atol = 1e-6    # right domain -x
        @test m[3, lic, ljc, 1] > 0.9                # vortex core at the centre
    end

    @testset "random config is normalized and varied" begin
        m = setconfig(mesh, RandomConfig(MersenneTwister(7)))
        @test isnorm(m)
        @test m[:, 1, 1, 1] != m[:, 2, 1, 1]
    end
end
