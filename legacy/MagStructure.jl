"""
    Defines the magnetic structure that will be simulated, based on the finite difference mesh from MeshGeometry.
    Defines the structure geometry and initializes the 3D magnetization vector field (Mx, My and Mz).

    Rafael L. Novak, rlnovak@gmail.com, jul2020, UFSC/Blumenau (Brazil).
"""

module MagStructure

using MeshGeometry
import Base:zeros

export initialize
export MagStructure

struct MagStructure{T}
    M::Array{Array{T,3},1}
end


end
