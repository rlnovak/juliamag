"""
    Calculates the demagnetization field for a certain magnetization state and pre-calculated demagnetization kernel (DemagKernel.jl).
    The goal is to apply the convolution theorem to the magnetization arrays (3x3) and kernel arrays in order to determine the
    demagnetization field over the entire mesh for the magnetization state given by the input M. The goal is to calculate 3 3x3 matrices,
    each one corresponding to the x, y or z component of the demagnetization field, Bd.
    The calculation proceeds, roughly, as
    
    Rafael L. Novak, rlnovak@gmail.com, feb2020, UFSC/Blumenau (Brazil).
"""

module DemagField

using MeshGeometry
import Base: copyto!, unsafe_copyto!

export zeroAll!, zeroPaddedRegion!, initPadded, padIndices

# Talvez seja melhor deixar alocado o padded e copiar data com copyto!(), como no mumax3.

function zeroAll!(dst)
    for i in eachindex(dst)
        @inbounds dst[i] = 0f0
    end
end

function zeroPaddedRegion!(dst, indices)
    for i in indices
        @inbounds dst[i] = 0f0
    end
end

function initPadded(padDims)
    paddedArray = zeros(padDims)
    return paddedArray
end

# Incluir nas criadoras de Mesh e de PaddedArray!
# E em alguma outra de conveniência, como por exemplo de Magnetization (que ainda tenho que criar!)
# offsets = Int64[]
# for k in 0:Nz-1
#     for i in 0:Nx-1
#         off = k*padNy*padNx + 0.25*padNy*(padNx+1)+i*padNy
#         push!(offsets, off+1)
#         push!(offsets, off+Ny)
#     end
# end


"""
Preciso da matriz 3D dst, com size padSize já alocada em padIndices, pois o LinearIndices precisa
dela para criar o array indices, objetivo da function!

Tenho que pensar a melhor estrategia: padIndices recebe dst (criada em outra funcao) como arg.
e a usa em LinearIndices, ou uso padIndices em uma funcao que faz tudo: cria/aloca dst com size padSize
e a retorna juntamente com os indices da regiao onde Mi sera copiada com copyto!() ?

copyto!(dest, Rdest::CartesianIndices, src, Rsrc::CartesianIndices) -> dest
https://docs.julialang.org/en/v1/base/c/#Base.copyto!

O arg para copyto!() é de tipo CartesianIndices!

Em vista disso, posso simplesmente retornar um array de CartesianIndices e eliminar o problema de ter
que passar dst como arg ou cria-lo ao mesmo tempo que rodo padIndices. Implementar amanha!

Depois: padIndices está me dando uma z-slice a mais do que eu preciso!!
Data original é (8,8,2) e o A.size (padSize) está com ()


Tem que usar copy!(pA.data, A) para copiar (hard copy) o array A para o padded array sem simplesmente
passar os ponteiros.
"""

function zeroPad!(dst, src, m::Mesh)
    # dst initialized with initPadded() before first call to zeroPad!()
    # Subsequent calls must be made following zeroPaddedRegion!(dst, indices)
    # Really necessary? Because copying the new Mag data to the padded array will overwrite
    # old data, so there should be no need to zero the values.
    # try
    #     slices = size(src)[3]
    # catch
    #     BoundsError("src must have 3 dimensions!")
    # end
    # TODO:
    # Overload unsafe_copyto!() to accept UnitRanges as arguments.
    @assert length(size(src)) == 3
    slices = size(src)[3]
    Ny = m.size[2]
    #for k in 1:slices
        for (i,r) in enumerate(m.dataranges)
            _srcrange = ((i-1)*Ny+1):(i*Ny)
            unsafe_copyto!(dst, r, src, _srcrange)
        end
    #end
    #copyto!(dst, src, indices)
end

## TODO: Check this function, because it is returning indices along dataless z slices!
function padIndices(m::Mesh)
    meshsize = m.size
    padSize = m.padsize
    pbc = m.pbc
    @assert length(meshsize) == 3
    Nx,Ny,Nz = meshsize
    _length = prod(meshsize) # Nx*Ny*Nz
    #@assert _length == length(M) # M is no longer an argument.

    if pbc == [0,0,0] # No PBC along x, y or z directions!
        #padSize = [meshsize[i]*2 for i in 1:length(meshsize)] # Implement -> round it to next power of 2! # Moved to Mesh() creator in MeshGeometry.jl
        ##########################################
        ## Using Mumax3 zero-padding along Z! ##
        ##########################################
        minRow = padSize[1] ÷ 4 + 1
        maxRow = padSize[1] ÷ 4 + Ny
        minCol = padSize[2] ÷ 4 + 1
        maxCol = padSize[2] ÷ 4 + Nx
        ## Using Mumax3 zero-padding along Z!
        # The correct padded sizes along Z for both 2D and 3D cases are covered by padSize() in the Mesh() constructor.
        minSlice = 1
        maxSlice = padSize[3]
        idxC = CartesianIndex.(Iterators.product(minRow:maxRow, minCol:maxCol, minSlice:maxSlice) |> collect)
        #indices = [LinearIndices(A)[idxC[i]] for i in 1:length(idxC)]
        return idxC
    else # PBC along one or more directions.
        #pbc_axes = findall(pbc.!=0) # Moved to Mesh() creator in MeshGeometry.jl
        #padSize = [i ∉ pbc_axes ? meshsize[i]*2 : meshsize[i] for i in 1:length(meshsize)] # Moved to Mesh() creator in MeshGeometry.jl
        if 1 ∈ pbc_axes && 2 ∈ pbc_axes # PBC along x and y directions
            minRow = 1
            maxRow = padSize[1] # Ny
            minCol = 1
            maxCol = padSize[2] # Nx
        elseif 1 ∈ pbc_axes && 2 ∉ pbc_axes # PBC along x direction
            minRow = padSize[1] ÷ 4 + 1
            maxRow = padSize[1] ÷ 4 + Ny
            minCol = 1
            maxCol = padSize[2] # Nx
        elseif 2 ∈ pbc_axes && 1 ∉ pbc_axes # PBC along y direction
            minRow = 1
            maxRow = padSize[1] # Ny
            minCol = padSize[2] ÷ 4 + 1
            maxCol = padSize[2] ÷ 4 + Nx
        end
        ## Using Mumax3 zero-padding along Z!
        # The correct padded sizes along Z for both 2D and 3D cases are covered by padSize() in the Mesh() constructor.
        minSlice = 1
        maxSlice = padSize[3]
        idxC = CartesianIndex.(Iterators.product(minRow:maxRow, minCol:maxCol, minSlice:maxSlice) |> collect)
        #indices = [LinearIndices(A)[idxC[i]] for i in 1:length(idxC)]
        return idxC
    end
end

###################################################################################################################
###################################################################################################################
## Testar aqui metdos de https://julialang.org/blog/2016/02/iteration/ para calcular o campo de troca com e sem pbc
###################################################################################################################
function testExchange(A::AbstractArray)
    out = similar(A)
    R = CartesianIndices(A)
    Ifirst, Ilast = first(R), last(R)
    I1 = oneunit(Ifirst)
    for I in R
        n, s = 0, zero(eltype(out))
        for J in max(Ifirst, I-I1):min(Ilast, I+I1)
            s += A[J]
            n += 1
        end
        out[I] = s/n
    end
    return out
end
###################################################################################################################
###################################################################################################################

function demagFieldFFT(sample::MagStructure, kernel::Kernel)
    M = sample.magnetization
    B = similar(M)
    for comp in [1,2,3]
        zero_init!(dst)
        zero_pad!(dst, M[comp])
        forward_fft!(dst)
        convolve!(dst, kernel)
        inverse_fft!(dst)
        unpad!(B[comp], dst)
    end
    return B
    # Descrever o que cada uma dessas funções faz! Verificar como é feito no mumax3 e na tese do Selke.
    # Fazer igual!
end
#function padIndices(meshsize, pbc)
    #     @assert length(meshsize) == 3
    #     Nx,Ny,Nz = meshsize
    #     _length = prod(meshsize) # Nx*Ny*Nz
    #     #@assert _length == length(M) # M is no longer an argument.
    
    #     if pbc == [0,0,0] # No PBC along x, y or z directions!
    #         padSize = [meshsize[i]*2 for i in 1:length(meshsize)] # Implement -> round it to next power of 2!
    #         ##########################################
    #         ## Correct Nz for zero padding along z! ##
    #         ##########################################
    #         minRow = padSize[1] ÷ 4 + 1
    #         maxRow = padSize[1] ÷ 4 + Ny
    #         minCol = padSize[2] ÷ 4 + 1
    #         maxCol = padSize[2] ÷ 4 + Nx
    #         if Nz < 2 # 2D
    #             ## Nothing along z in 2D case. CHECK THIS!
    #             minSlice = 1
    #             maxSlice = 1
    #         elseif Nz > 1 # 3D
    #             minSlice = 1
    #             maxSlice = padSize[3]
    #         end
    #         idxC = CartesianIndex.(Iterators.product(minRow:maxRow, minCol:maxCol, minSlice:maxSlice) |> collect)
    #         indices = [LinearIndices(A)[idxC[i]] for i in 1:length(idxC)]
    #         return indices
    #     else # PBC along one or more directions.
    #         pbc_axes = findall(pbc.!=0)
    #         padSize = [i ∉ pbc_axes ? meshsize[i]*2 : meshsize[i] for i in 1:length(meshsize)]
    #         if 1 ∈ pbc_axes && 2 ∈ pbc_axes # PBC along x and y directions
    #             minRow = 1
    #             maxRow = padSize[1] # Ny
    #             minCol = 1
    #             maxCol = padSize[2] # Nx
    #         elseif 1 ∈ pbc_axes && 2 ∉ pbc_axes # PBC along x direction
    #             minRow = padSize[1] ÷ 4 + 1
    #             maxRow = padSize[1] ÷ 4 + Ny
    #             minCol = 1
    #             maxCol = padSize[2] # Nx
    #         elseif 2 ∈ pbc_axes && 1 ∉ pbc_axes # PBC along y direction
    #             minRow = 1
    #             maxRow = padSize[1] # Ny
    #             minCol = padSize[2] ÷ 4 + 1
    #             maxCol = padSize[2] ÷ 4 + Nx
    #         end
    
    #         if Nz < 2
    #             ## Nothing along z in 2D case. CHECK THIS!
    #             minSlice = 1
    #             maxSlice = 1            
    #         elseif Nz > 1
    #             minSlice = 1
    #             maxSlice = padSize[3]
    #         end
    #         idxC = CartesianIndex.(Iterators.product(minRow:maxRow, minCol:maxCol, minSlice:maxSlice) |> collect)
    #         indices = [LinearIndices(A)[idxC[i]] for i in 1:length(idxC)]
    #         return indices
    #     end
    # end

# ##################### Versão spaghetti ########################
#################################################################
#
#     function padIndicesOld(M, _size, pbc) # Spaghetti version of the function. COMPLETE IT!!
#         @assert length(_size) == 3
#         Nx,Ny,Nz = _size
#         _length = prod(_size) # Nx*Ny*Nz
#         @assert _length == length(M)
    
#         if pbc == [0,0,0] # No PBC along x, y or z directions!
#             indices = zeros(Int64, _length)
#             padDims = [_size[i]*2 for i in 1:length(_size)]
#             if Nz < 2 # 2D
#                 padNx, padNy = padDims[1], padDims[2]
#                 for j in 1:Nx
#                     for i in ((j-1)*Nx+1):j*Ny # Nx*Ny
#                         indices[i] = 0.25*padNy*(padNx + 1) + (j-1)*padNy + (i-1)%Ny+1
#                     end
#                 end
#                 return indices
#             elseif Nz > 1 # 3D
#                 padNx, padNy, padNz = padDims[1], padDims[2], padDims[3]  # Check if padNz is correct.
#                 for k in 1:Nz
#                     for j in 1:Nx
#                         for i in ((j-1)*Nx+1):j*Ny # Nx*Ny*Nz
#                             indices[i] = (k-1)*padNx*padNy + 0.25*padNy*(padNx + 1) + (j-1)*padNy + (i-1)%Ny+1
#                         end
#                     end
#                 end
#                 return indices
#             end
#         else # PBC along one or more directions.
#             pbc_axes = findall(pbc.!=0)
#             padDims = [i ∉ pbc_axes ? _size[i]*2 : _size[i] for i in 1:length(_size)]
#             if Nz < 2 # 2D
#                 padNx, padNy = padDims[1], padDims[2]
#                 if 1 ∈ pbc_axes && 2 ∈ pbc_axes # PBC along x and y directions
#                     indices = collect(1:_length)
#                     return indices
#                 elseif 1 ∈ pbc_axes && 2 ∉ pbc_axes # PBC along x direction
#                     indices = zeros(Int64, _length)
#                     padNx, padNy = padDims[1], padDims[2]
#                 for j in 1:Nx
#                     for i in ((j-1)*Nx+1):j*Ny # Nx*Ny
#                         indices[i] = 0.25*padNy + 1 + (j-1)*padNy + (i-1)%Ny+1
#                     end
#                 end
#                 return indices
#                 for j in 1:Nx
#                     for i in ((j-1)*Nx+1):j*Ny # Nx*Ny
#                         indices[i] = 0.25*padNy*(padNx + 1) + (j-1)*padNy + (i-1)%Ny+1
#                     end
#                 end
#                 return indices
#             elseif Nz > 1 # 3D
#                 # padDims[end] = ?? # Correct padded Nz!
#                 padNx, padNy, padNz = padDims[1], padDims[2], padDims[3]  # Check if padNz is correct.
#                 for k in 1:Nz
#                     for j in 1:Nx
#                         for i in 1:_length # Nx*Ny*Nz
#                             indices[i] = (k-1)*padNx*padNy + (j+1)*Ny*0.5*Nx+0.25*padNy+i
#                         end
#                     end
#                 end
#                 return indices
#             end
#         end
#     end
# end



end
