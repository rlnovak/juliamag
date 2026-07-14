using LinearAlgebra: norm
using Random: MersenneTwister

# |m| == 1 in every cell.
function isnormalized(m::AbstractArray{T,4}; atol = 1e-12) where {T}
    all(CartesianIndices(axes(m)[2:4])) do I
        abs(norm(@view m[:, I]) - 1) <= atol
    end
end

@testset "Magnetization" begin
    mesh = Mesh((8, 6, 4), (5e-9, 5e-9, 5e-9))

    @testset "layout" begin
        m = zeromag(mesh)
        @test size(m) == (3, 8, 6, 4)
        @test eltype(m) === Float64
        @test all(iszero, m)
        @test eltype(zeromag(Float32, mesh)) === Float32
    end

    @testset "uniform" begin
        m = uniform(mesh, (0, 0, 1))
        @test isnormalized(m)
        @test all(average(m) .≈ (0.0, 0.0, 1.0))

        # The direction is normalized, so an unnormalized argument works.
        m = uniform(mesh, (3, 4, 0))
        @test isnormalized(m)
        @test all(average(m) .≈ (0.6, 0.8, 0.0))

        @test_throws ArgumentError uniform(mesh, (0, 0, 0))
    end

    @testset "random" begin
        m = randommag!(MersenneTwister(42), zeromag(mesh))
        @test isnormalized(m)
        # Distinct cells get distinct vectors — guards against filling the whole
        # array from a single draw.
        @test m[:, 1, 1, 1] != m[:, 2, 1, 1]
    end

    @testset "vortex" begin
        m = vortex(mesh)
        @test isnormalized(m)
        # In-plane circulation cancels, so ⟨mx⟩ = ⟨my⟩ = 0 and only the core
        # contributes to ⟨mz⟩.
        mx, my, mz = average(m)
        @test mx ≈ 0 atol = 1e-12
        @test my ≈ 0 atol = 1e-12
        @test mz > 0

        @test average(vortex(mesh; core = -1))[3] < 0

        # Reversing the circulation reverses the in-plane component everywhere.
        mp = vortex(mesh; circulation = 1)
        mn = vortex(mesh; circulation = -1)
        @test mp[1:2, :, :, :] ≈ -mn[1:2, :, :, :]
        @test mp[3, :, :, :] ≈ mn[3, :, :, :]
    end

    @testset "normalize!" begin
        m = zeromag(mesh)
        m[:, :, :, :] .= 3.0
        normalize!(m)
        @test isnormalized(m)
        @test all(m .≈ 1 / sqrt(3))
    end

    @testset "average" begin
        m = zeromag(mesh)
        m[:, :, :, :] .= 0.0
        m[1, :, :, :] .= 1.0            # half the cells along x point +x…
        m[1, 1:4, :, :] .= -1.0         # …and half point -x
        @test average(m)[1] ≈ 0 atol = 1e-12
    end
end
