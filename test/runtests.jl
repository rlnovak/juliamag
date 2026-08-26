using JuliaMag
using Test

@testset "JuliaMag" begin
    include("test_mesh.jl")
    include("test_material.jl")
    include("test_magnetization.jl")
    include("test_fields.jl")
    include("test_dmi.jl")
    include("test_demag_kernel.jl")
    include("test_demag_field.jl")
    include("test_dynamics.jl")
    include("test_energy.jl")
    include("test_stt.jl")
    include("test_shape.jl")
    include("test_regions.jl")
    include("test_region_params.jl")
    include("test_voronoi.jl")
    include("test_config.jl")
    include("test_ovf.jl")
    include("test_output.jl")
    include("test_simulation.jl")
    include("test_thermal.jl")
end
