""" Function definitions of the micromagnetic energies.

   Rafael L. Novak, UFSC Blumenau. rlnovak@gmail.com
   Created: 3oct19. Modified: 23oct19
"""

module Energies

using Utils: neigh, μ0
using LinearAlgebra: ⋅

export uniaxialAnisotropyEnergy, exchangeEnergy6Nbr, uniaxialAnisotropyEnergy, zeemanEnergy

### TO DO!
## Now we can use the linearindices from Utils.neighbours() to accelerate this!
###

function exchangeEnergy6Nbr(mx::Array{T,3}, my::Array{T,3}, mz::Array{T,3},
                                 nx, ny, nz, cx::T, cy::T, cz::T, A::T, Msat::T; pbc::Bool = false) where T <: AbstractFloat
    # Remove args nx, ny and nz -> determine them from size(mx), for example.
    # Receive pointers to mx, my and mz instead of the arrays. How to work with pointers in Julia?
    ## Then, these functions should be reimplemented to run on GPU -> Cuda and/or ArrayFire.
    E = 0.0
    for k in 1:nz # Add @inbounds ?
        for j in 1:ny
            for i in 1:nx
                m_central = [mx[i,j,k], my[i,j,k], mz[i,j,k]]
                for n in neigh(i,nx)
                    m_neigh = [mx[n,j,k], my[n,j,k], mz[n,j,k]]
                    aux = 1.0 - m_central⋅m_neigh
                    E += aux/(cx*cx)
                end
                m_central = [mx[i,j,k], my[i,j,k], mz[i,j,k]]
                for n in neigh(j,ny)
                    m_neigh = [mx[i,n,k], my[i,n,k], mz[i,n,k]]
                    aux = 1.0 - m_central⋅m_neigh
                    E += aux/(cy*cy)
                end
                if nz > 1
                    m_central = [mx[i,j,k], my[i,j,k], mz[i,j,k]]
                    for n in neigh(k,nz)
                        m_neigh = [mx[i,j,n], my[i,j,n], mz[i,j,n]]
                        aux = 1.0 - m_central⋅m_neigh
                        E += aux/(cz*cz)
                    end
                end
            end
        end
    end
    E = A*cx*cy*cz*E
end

function uniaxialAnisotropyEnergy(mx::Array{T,3}, my::Array{T,3}, mz::Array{T,3},
                                 nx, ny, nz, cx::T, cy::T, cz::T, K::T, u::Array{T,1}, Msat::T; pbc::Bool = false) where T <: AbstractFloat
    E = 0.0
    #u = reshape(u,3,1)
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                m_central = [mx[i,j,k], my[i,j,k], mz[i,j,k]]
                E += 1.0 - (m_central⋅u)^2
            end
        end
    end
    E = K*cx*cy*cz*E
end

function DMexchangeEnergy6nbr(mx::Array{T,3}, my::Array{T,3}, mz::Array{T,3},
                                 nx, ny, nz, cx::T, cy::T, cz::T, Dmi::T, Msat::T; pbc::Bool = false) where T <: AbstractFloat
                                 ## FAZER!
end

function dipolarEnergyFFT(mx::Array{T,3}, my::Array{T,3}, mz::Array{T,3},
                                 nx, ny, nz, cx::T, cy::T, cz::T, demagTensor::Array{Array{T,3},1}, Msat::T; pbc::Bool = false) where T <: AbstractFloat
                                 ## FAZER!
end

function zeemanEnergy(mx::Array{T,3}, my::Array{T,3}, mz::Array{T,3},
                                 nx, ny, nz, cx::T, cy::T, cz::T, Hext::Array{T,1}, Msat::T; pbc::Bool = false) where T <: AbstractFloat

    E = 0.0
    H = reshape(Hext,3,1)
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                m_central = [mx[i,j,k], my[i,j,k], mz[i,j,k]]
                E += m_central⋅H
            end
        end
    end
    E = -mu0*Msat*cx*cy*cz*E
    return E
end

end
