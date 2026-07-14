""" Utility and miscelaneous functions for micromagnetics with Julia.

   Rafael L. Novak, UFSC Blumenau. rlnovak@gmail.com
   Created: 3oct19. Modified: 25fev20
"""

module Utils
import Base: zeros, ones

using MeshGeometry

export convert_to_MagVectorArray, neigh, neighbours
export μ0, μB, kB, qe, γLL

############## Physical constants ###############
const μ0 = 4*π*1e-7 # Vacuum permeability in Tm/A
const μB = 9.2740091523e-24 # Bohr magneton in J/T
const kB  = 1.380650424e-23 # Boltzmann's constant in J/K
const qe  = 1.60217646e-19  # Electron charge in C
const γLL = 1.7595e11 # Gyromagnetic ratio in rad/Ts -> 28.02495164 GHz/T
#################################################

## Dot product -> Defined in LinearAlgebra
#⋅(a::Vector{T},b::Vector{T}) where T <: Real = sum(a.*b)

function convert_to_MagVectorArray(mx,my,mz,nx,ny,nz) # Old convert_m
	""" Takes three Array{Float64,3} as input -> mx, my and mz, each containing the x, y and z components of the reduced magnetization vector.
        Outputs an Array{Array{Float64,1},3}, that is, a 3D array of reduced magnetization vectors.
    """
    m = Array{Vector{Float64}}(undef,nx,ny,nz)
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                m[i,j,k] = [mx[i,j,k], my[i,j,k], mz[i,j,k]]
            end
        end
    end
    return m                
end

function neigh(i, n)
    if n == 1
        return (1,1)
    elseif n > 1
        if i == 1
            return (i,i+1)
        elseif i == n
            return (n-1,n)
        else
            return (i-1, i+1)
        end
    elseif n <= 0
        throw(DomainError(x, "The number of cells cannot be < 1!"))
    end
end


function neighbours(m::Mesh; linear = true)
    N = length(m)
    Nx, Ny, Nz = m.size[1], m.size[2], m.size[3]
    pbc = m.pbc
    ######################
    S = ones(Ny, Nx, Nz) # Not the best way to do it...
    ######################
    nbr = Array{Array{Tuple{Int64,Int64,Int64},1},3}(undef, Ny, Nx, Nz)
    @inbounds for k in 1:Nz
        @inbounds for j in 1:Nx
            @inbounds for i in 1:Ny
                if pbc[1] != 0 && pbc[2] != 0 && pbc[3] != 0 # PBC along all directions!
                    nbr[i,j,k] = [(wrap(i+1,Ny),j,k),
                    (i,wrap(j+1,Nx),k),
                    (wrap(i-1,Ny),j,k),
                    (i,wrap(j-1,Nx),k),
                    (i,j,wrap(k+1,Nz)),
                    (i,j,wrap(k-1,Nz))]
                elseif pbc[1] != 0 && pbc[2] != 0 && pbc[3] == 0 # In-plane pbc
                    nbr[i,j,k] = [(wrap(i+1,Ny),j,k),
                    (i,wrap(j+1,Nx),k),
                    (wrap(i-1,Ny),j,k),
                    (i,wrap(j-1,Nx),k),
                    (i,j,k+1 > Nz ? Nz : k+1),
                    (i,j,k-1 < 1 ? 1 : k-1)]
                elseif pbc[1] != 0 && pbc[2] == 0 && pbc[3] == 0 # pbc along x direction
                    nbr[i,j,k] = [(i+1 > Ny ? Ny : i+1,j,k),
                    (i,wrap(j+1,Nx),k),
                    (i-1 < 1 ? 1 : i-1,j,k),
                    (i,wrap(j-1,Nx),k),
                    (i,j,k+1 > Nz ? Nz : k+1),
                    (i,j,k-1 < 1 ? 1 : k-1)]
                elseif pbc[1] == 0 && pbc[2] != 0 && pbc[3] == 0 # pbc along y direction
                    nbr[i,j,k] = [(wrap(i+1,Ny),j,k),
                    (i,j+1 > Nx ? Nx : j+1,k),
                    (wrap(i-1,Ny),j,k),
                    (i,j-1 < 1 ? 1 : j-1,k),
                    (i,j,k+1 > Nz ? Nz : k+1),
                    (i,j,k-1 < 1 ? 1 : k-1)]
                elseif pbc[1] == 0 && pbc[2] == 0 && pbc[3] != 0 # pbc along z direction
                    nbr[i,j,k] = [(i+1 > Ny ? Ny : i+1,j,k),
                    (i,j+1 > Nx ? Nx : j+1,k),
                    (i-1 < 1 ? 1 : i-1,j,k),
                    (i,j-1 < 1 ? 1 : j-1,k),
                    (i,j,wrap(k+1,Nz)),
                    (i,j,wrap(k-1,Nz))]
                elseif pbc == [0,0,0] # No pbc at all!
                    nbr[i,j,k] = [(i+1 > Ny ? Ny : i+1,j,k),
                    (i,j+1 > Nx ? Nx : j+1,k),
                    (i-1 < 1 ? 1 : i-1,j,k),
                    (i,j-1 < 1 ? 1 : j-1,k),
                    (i,j,k+1 > Nz ? Nz : k+1),
                    (i,j,k-1 < 1 ? 1 : k-1)]
                ## More combinations! (x,z), (y,z) -> Not very useful
                end
            end
        end
    end
    if linear
        nbr_linear = Array{Array{Int64,1},3}(undef, Ny, Nx, Nz)
        @inbounds for k in 1:Nz
            @inbounds for j in 1:Nx
                @inbounds for i in 1:Ny
                    neighs = nbr[i,j,k]
                    nbr_linear[i,j,k] = [LinearIndices(S)[l[1], l[2], l[3]] for l in neighs]
                end
            end
        end
        return nbr_linear
    else
        nbrC = [CartesianIndex.(n) for n in nbr]
        return nbrC
    end
end

############## Helper functions ####################
wrap(i,N) = mod((i-1),N) + 1
ispbc(i::Int64) = i == 0 ? 0 : one(i)
zeros(a::Array{Int64,1}) = zeros(a[1],a[2],a[3])
ones(a::Array{Int64,1}) = ones(a[1],a[2],a[3])
####################################################


##############################################################
############### Spaghetti version of neighbours ##############
#
# function neighbours_old(m::Mesh; linear = true)
#     N = length(m)
#     Nx, Ny, Nz = m.size[1], m.size[2], m.size[3]
#     PBC = m.pbc
#     ######################
#     S = ones(m.size) # Not the best way to do it...
#     ######################
#     nbr = Array{Array{Tuple{Int64,Int64,Int64},1},3}(undef, Ny, Nx, Nz)

#     if PBC != [0,0,0]
#         @inbounds for k in 1:Nz
#             @inbounds for j in 1:Nx
#                 @inbounds for i in 1:Ny
#                     nbr[i,j,k] = [(((i+1) == (Ny+1) ? 1 : i+1),j,k),
#                     (i,((j+1)==(Nx+1) ? 1 : j+1),k),
#                     (((i-1 == 0 ? Ny : i-1)%(Ny+1)),j,k),
#                     (i,(j-1 == 0 ? Nx : j-1)%(Nx+1),k),
#                     (i,j,((k+1)==(Nz+1) ? 1 : k+1),
#                     (i,j,(k-1 == 0 ? Nz : k-1)%(Nz+1))]
#                 end
#             end
#         end
#     # Introduzir eixo z
#     # 6 tuplas em cada array 1D em cada posicao 3d da mesh. Repetir sitios nos bordos
#     elseif PBC == [0,0,0]
#         @inbounds for k in 1:Nz
#             @inbounds for j in 1:Nx
#                 @inbounds for i in 1:Ny
#                     if i == 1
#                         if j == 1
#                             nbr[i,j,k] = [(i+1, j), (i, j+1)]
#                         elseif j == Nx
#                             nbr[i,j,k] = [(i+1, j), (i, j-1)]
#                         else
#                             nbr[i,j,k] = [(i+1,j), (i,j+1), (i,j-1)]
#                         end
#                     elseif i == Ny
#                         if j == 1
#                             nbr[i,j,k] = [(i, j+1), (i-1, j)]
#                         elseif j == Nx
#                             nbr[i,j,k] = [(i-1, j), (i, j-1)]
#                         else
#                             nbr[i,j,k] = [(i,j+1), (i-1,j), (i,j-1)]
#                         end
#                     elseif j == 1 && (i != 1 && i != Ny)
#                         nbr[i,j,k] = [(i, j+1), (i-1, j), (i+1,j)]
#                     elseif j == Nx && (i != 1 && i != L)
#                         nbr[i,j,k] = [(i-1, j), (i, j-1), (i+1,j)]
#                     else
#                         nbr[i,j,k] = [(i+1, j), (i, j+1), (i-1,j), (i, j-1)]
#                     end
#                 end
#             end
#         end
#     end
#     if linear
#         nbr_linear = Array{Array{Int64,1},3}(undef, Ny, Nx, Nz)
#         @inbounds for k in 1:Nz
#             @inbounds for j in 1:Nx
#                 @inbounds for i in 1:Ny
#                     neighs = nbr[i,j,k]
#                     nbr_linear[i,j,k] = [LinearIndices(S)[l[1], l[2], l[3]] for l in neighs]
#                 end
#             end
#         end
#         return nbr_linear
#     else
#         return nbr
#     end
# end

end