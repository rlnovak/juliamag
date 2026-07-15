using JuliaMag
using Test

@testset "JuliaMag" begin
    include("test_mesh.jl")
    include("test_material.jl")
    include("test_magnetization.jl")
    include("test_fields.jl")
    include("test_demag_kernel.jl")
end
