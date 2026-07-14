""" Function definitions of the fields in a micromagnetic simulation.
	These fields will be added to the external (Zeeman) field, giving the effective field in the LLG equation.

   Rafael L. Novak, UFSC Blumenau. rlnovak@gmail.com
   Created: 4oct19. Modified: 23oct19
"""

module Fields

using Utils: neigh, μ0
using LinearAlgebra: ⋅, ×

export exchangeField6nbr, uniaxialAnisotropyField

### TO DO!
## Now we can use the linearindices from Utils.neighbours() to accelerate this!
###

function exchangeField6nbr(mx::Array{T,3}, my::Array{T,3}, mz::Array{T,3},
                           nx, ny, nz, cx::T, cy::T, cz::T, A::T, Msat::T; pbc::Bool = false) where T <: AbstractFloat
    # Hex_x::Array{T,3}, Hex_y::Array{T,3}, Hex_z::Array{T,3},
    # Remove args nx, ny and nz -> determine them from size(mx), for example.
    ## These should be pointers to Hex_x, y and z. The function will receive the pointers as arguments.
    ## Then, these functions should be reimplemented to run on GPU -> Cuda and/or ArrayFire.
    Hex_x = zeros(size(mx))
    Hex_y = zeros(size(my))
    Hex_z = zeros(size(mz))
    for k in 1:nz # Add @inbounds ?
        for j in 1:ny
            for i in 1:nx
                aux_x = zeros(1,3)
                aux_y = zeros(1,3)
                aux_z = zeros(1,3)
                m_neigh = zeros(1,3)
                m_central = [mx[i,j,k] my[i,j,k] mz[i,j,k]]
                for n in neigh(i,nx)
                    m_neigh += [mx[n,j,k] my[n,j,k] mz[n,j,k]]
                end
                aux_x = (m_neigh - 2.0.*m_central)/(cx*cx)
                m_neigh = zeros(1,3)
                for n in neigh(j,ny)
                    m_neigh += [mx[i,n,k] my[i,n,k] mz[i,n,k]]
                end
                aux_y = (m_neigh - 2.0.*m_central)/(cy*cy)
                if nz > 1
                    m_neigh = zeros(1,3)
                    for n in neigh(k,nz)
                        m_neigh += [mx[i,j,n] my[i,j,n] mz[i,j,n]]
                    end
                    aux_z = (m_neigh - 2.0.*m_central)/(cz*cz)
                end
                Hex_x[i,j,k], Hex_y[i,j,k], Hex_z[i,j,k] = (aux_x+aux_y+aux_z)
            end
        end
    end
    Hex_x = 2.0*A.*Hex_x./(μ0*Msat)
    Hex_y = 2.0*A.*Hex_y./(μ0*Msat)
    Hex_z = 2.0*A.*Hex_z./(μ0*Msat)
    return (Hex_x, Hex_y, Hex_z)
end

function uniaxialAnisotropyField(mx::Array{T,3}, my::Array{T,3}, mz::Array{T,3},
                                 nx, ny, nz, cx::T, cy::T, cz::T, K::T, u::Array{T,1}, Msat::T; pbc::Bool = false) where T <: AbstractFloat
## Pointers!!
    hx = zeros(size(mx))
    hy = zeros(size(my))
    hz = zeros(size(mz))
    #u = reshape(u,3,1)
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                m_central = [mx[i,j,k], my[i,j,k], mz[i,j,k]]
                aux = (m_central⋅u).*u
                hx[i,j,k] = aux[1]
                hy[i,j,k] = aux[2]
                hz[i,j,k] = aux[3]
            end
        end
    end
    hx = 2.0*((K*cx*cy*cz)/(mu0*Msat)).*hx
    hy = 2.0*((K*cx*cy*cz)/(mu0*Msat)).*hy
    hz = 2.0*((K*cx*cy*cz)/(mu0*Msat)).*hz
    return (hx, hy, hz)
end

function DMexchangeField6nbr(mx::Array{T,3}, my::Array{T,3}, mz::Array{T,3},
                                 nx, ny, nz, cx::T, cy::T, cz::T, Dmi::T, Msat::T; pbc::Bool = false) where T <: AbstractFloat
end

function dipolarFieldFFT(mx::Array{T,3}, my::Array{T,3}, mz::Array{T,3},
                                 nx, ny, nz, cx::T, cy::T, cz::T, demagTensor::Array{Array{T,3},1}, Msat::T; pbc::Bool = false) where T <: AbstractFloat
end

end
