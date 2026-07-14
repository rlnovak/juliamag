module PaddedArray
import Base: show, copyto!, size, length, getindex

export PaddedArray


## TODO: Finish! The getindex mehods are not correct. Get the index arithmetic right and finish this (probably useless!) module!

###############################################################################
###############################################################################

""" Estou pensando agora: a melhor estratégia para o zero padding não seria definir um tipo PaddedArray e através do overloading
    de setindex, getindex, etc., o comportamento seria: acesso a indices na regiao zero-padded da zero (sem ter alocado isso na memória!),
    enquanto que acesso a outros indices retornaria o elemento desejado.
"""

###############################################################################
############ Definition of new type PaddedArray and its methods ###############
mutable struct PaddedArray{T, N}
    data::Array{T,N}
    size::Vector{Int64}
    datasize::Vector{Int64}
    dataidx::Array{CartesianIndex{N},N}
    lineardataidx::Array{Int64,N}
    linearidx::LinearIndices{N,Tuple{UnitRange{Int64},UnitRange{Int64},UnitRange{Int64}}}
    offsets::Vector{Int64}
end

######################### Outer Constructor ####################################
function PaddedArray(src::Array{T,N}, m::Mesh) where T<:AbstractFloat where N
    #m.size == size(src) || throw(ArgumentError("The source array and the mesh must have same size (got $(size(src)) and $(m.size))"))
    data = src
    size = m.padsize
    datasize = m.size
    dataidx = padIndices(m)
    linearidx = LinearIndices((1:size[1], 1:size[2], 1:size[3]))
    lineardataidx = LinearIndices(linearidx)[dataidx]
    offsets = [lineardataidx[1,1,i] for i in 1:size(lineardataidx)[end]]
    PaddedArray(data,size,datasize,dataidx,lineardataidx,linearidx)
end

###################### Overloaded methods ########################
size(A::PaddedArray{T,N}) where {T,N} = A.size
length(A::PaddedArray{T,N}) where {T,N} = prod(size(A))

"""
Tenho que pensarbem nessa estrategia. Para definir o getindex e o setindex, eu preciso converter os
CartesianIndices em LinearIndices e vice-versa. O problema é que para isso, eu preciso do array
com padSize mas o array que eu tenho alocado é o A.data, que tem o tamanho original da mesh.
Como se resolve isso sem alocar outro array???

Depois: padIndices está me dando uma z-slice a mais do que eu preciso!!
Data original é (8,8,2) e o A.size (padSize) está com ()

Overload de eachindex(), first, last, axes(A,n)

Tem que usar copy!(pA.data, A) para copiar (hard copy) o array A para o padded array sem simplesmente
passar os ponteiros.
"""
function getindex(A::PaddedArray{T,N}, i::Int) where {T,N}
    i > length(A) && throw(BoundsError("Index $i is out of bounds."))
    _lineardataidx = A.lineardataidx
    if i ∉ _lineardataidx
        return eltype(A.data)(0)
    else
        slice = findall(x->x==i,_lineardataidx)[1][end]
        ## O PaddedArray poderia conter uma LUT com ranges de lineardataidx correspondentes a cada slice!
        offset = A.offsets[slice]
        return A.data[k-offset]
    end
end

function getindex(A::PaddedArray{T,N}, I::Vararg{Int,N}) where {T,N}
    if I[1] > A.size[1] || I[2] > A.size[2] || I[3] > A.size[3]
        throw(BoundsError("Index $I is out of bounds."))
    end
    _dataidx = A.dataidx # CartesianIndices
    if CartesianIndex(I) ∉ _dataidx
        return 0.0::eltype(A.data)
    else
        return A.data[i]
    end
end

Base.show(io::IO, ::MIME"text/plain", A::PaddedArray{T,N}) where {T,N} =
    print(io, "Zero-padded Array{$T,$N}\nPadded size: ", A.size, "\nData size: ", A.datasize)

Base.show(io::IO, ::MIME"text/html", A::PaddedArray{T,N}) where {T,N} = 
    print(io, "Zero-padded Array{$T,$N}\nPadded size: ", A.size, "\nData size: ", A.datasize)