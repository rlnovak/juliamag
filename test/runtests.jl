using JuliaMag
using Test

@testset "JuliaMag" begin
    include("test_mesh.jl")
    include("test_material.jl")
    include("test_magnetization.jl")
end
