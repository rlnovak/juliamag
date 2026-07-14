""" Functions to read ovf formatted files. The ovf file can be in ascii or binary format.
    Outputs:  A dict containing the simulation information contained in the ovf header.
              Keys -> "xbase", "ybase", "zbase", "xstepsize", "ystepsize", "zstepsize", "xnodes", "ynodes", "znodes", "valuemultiplier"
              A dict containing extra header lines, in case they are detected. Keys -> "SimTime", "Iteration", "Stage", "MIFSource"
              An Array{Float64,4} containing the x, y and z components of the magnetization vector field in the ovf file.
              1st index -> x node, 2nd index -> y node, 3rd index -> z node, 4th index -> component (1 -> x, 2 -> y, 3 -> z).
              The nodes correspond to the nodes in the simulated volume where the vectors are located.
    The ovf format is defined in the oommf manual -> https://math.nist.gov/oommf/doc/
    This code is based on OOMMFTools -> https://github.com/deparkes/OOMMFTools

    Rafael L. Novak, UFSC. rlnovak@gmail.com
    Created: 27jul19. Modified: 19feb20
"""

module OVF
    
export readOvf

## TO DO: Implement type safe function!() inside function() pattern in readOvf.

function _textDecode(f::IOStream, outArray::Array{Float64,4}, headers::Dict{String,Float64}, extraCaptures::Dict{String,Any})
    valm = get(headers, "valuemultiplier", 1.0)
    nx = Int(headers["xnodes"])
    ny = Int(headers["ynodes"])
    nz = Int(headers["znodes"])
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                text = split(readline(f))
                #println(text)
                outArray[i,j,k,1] = parse(Float64, text[1])
                outArray[i,j,k,2] = parse(Float64, text[2])
                outArray[i,j,k,3] = parse(Float64, text[3])
            end
        end
    end
    return (permutedims(reverse(outArray,dims=2),(2,1,3,4)), headers, extraCaptures)
end

## Implement a _binaryDecode macro that calls read(Float32 or Float64) according to the argument passed
function _binaryDecode8(f::IOStream, chunksize::Int64, rev::Bool, outArray::Array{Float64,4}, headers::Dict{String,Float64}, extraCaptures::Dict{String,Any})
    valm = get(headers, "valuemultiplier", 1.0)
    nx = Int(headers["xnodes"])
    ny = Int(headers["ynodes"])
    nz = Int(headers["znodes"])
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                for coord in 1:3
                    if rev ## rev = True -> reverse byte ordering
                        outArray[i,j,k,coord] = htol(read(f, Float64))
                    else
                        outArray[i,j,k,coord] = read(f, Float64)
                    end
                end
            end
        end
    end
    return (permutedims(reverse(outArray,dims=2),(2,1,3,4)), headers, extraCaptures)
end
################################################################################
######### Ver no manual do oommf a ordem seguida pelos valores no arquivo ovf! 
######### Uma fita retangular horizontal está ficando vertical, então devemos ter que 
######### inverter os indices i e j nos loops dessas funções!!!
################################################################################
function _binaryDecode4(f::IOStream, chunksize::Int64, rev::Bool, outArray::Array{Float64,4}, headers::Dict{String,Float64}, extraCaptures::Dict{String,Any})
    valm = get(headers, "valuemultiplier", 1.0)
    nx = Int(headers["xnodes"])
    ny = Int(headers["ynodes"])
    nz = Int(headers["znodes"])
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                for coord in 1:3
                    if rev ## rev = True -> reverse byte ordering
                        outArray[i,j,k,coord] = htol(read(f, Float32))
                    else
                        outArray[i,j,k,coord] = read(f, Float32)
                    end
                end
            end
        end
    end
    return (permutedims(reverse(outArray,dims=2),(2,1,3,4)), headers, extraCaptures)
end

function readOvf(filename)
    """ Pass the path or the filename (if in the same dir) as a string (double quotes).
        The function will return:
        
        1) A 4-dimensional array (M) with the 3 magnetization components given by the last index:
        M[:,:,:,1] -> Mx, M[:,:,:,2] -> My, M[:,:,:,3] -> Mz. Each of the Mi component arrays are 3-dimensional and have
        Nx x Ny x Nz elements, where Ni correspond to the finite difference mesh dimensions.
        
        2) A Dict. (headers) with the metadata presented as text in the header of the ovf file. These data describe the
        finite difference mesh and give some other details about the simulation that originated the ovf file.
        
        3) A Dict. (extraCaptures) with other, less important informations about the simulation that originated the ovf. 
    """
    f = open(filename)
    headers = Dict{String, Float64}()
    extraCaptures = Dict("SimTime" => -1, "Iteration" => -1, "Stage" => -1, "MIFSource" => "")
    _keys = ["xbase", "ybase", "zbase", "xstepsize", "ystepsize", "zstepsize", "xnodes", "ynodes", "znodes", "valuemultiplier"]
    line = ""
    while !occursin("Begin: Data", line)
        line = readline(f)
        for _key in _keys
            if occursin(_key, line)
                headers[_key] = parse(Float64, split(line)[end])
            end
        end
        if occursin("Total simulation time", line)
            extraCaptures["SimTime"] = parse(Float64,split(split(line,':')[end])[1])
        end
        if occursin("Stage simulation time", line)
            extraCaptures["StageTime"] = parse(Float64,split(split(line,':')[end])[1])
        end
        if occursin("Iteration", line)
            extraCaptures["Iteration"] = parse(Float64, split(split(line, ':')[3],',')[1])
        end
        if occursin("Stage", line) && !occursin("Stage simulation time",line)
            extraCaptures["Stage"] = parse(Float64, split(split(line, ':')[3],',')[1])
        end
        if occursin("MIF source file", line)
            extraCaptures["MIFSource"] = split(line, ':')[3]
        end
    end

    dec = split(line)
    #println(dec)
    pos = position(f)
    #println("At $pos")
    nx = convert(Int64, headers["xnodes"])
    ny = convert(Int64, headers["ynodes"])
    nz = convert(Int64, headers["znodes"])
    outArray = zeros(nx, ny, nz, 3)

    if dec[4] == "Text"
        #println("It's a text ovf!")
        #println("I'm at position $pos.")
        return _textDecode(f, outArray, headers, extraCaptures)
    elseif dec[4] == "Binary" && dec[5] == "4"
        println("Binary 4 bytes format.")
        endiantest = read(f, Float32)
        println(endiantest)
        if endiantest == 1234567.0
            return _binaryDecode4(f, 4, false, outArray, headers, extraCaptures)
        else
            return _binaryDecode4(f, 4, true, outArray, headers, extraCaptures)
        end
    elseif dec[4] == "Binary" && dec[5] == "8"
        println("Binary 8 bytes format.")
        endiantest = read(f, Float64)
        println(endiantest)
        if endiantest == 123456789012345.0
            return _binaryDecode8(f, 8, false, outArray, headers, extraCaptures)
        else
            return _binaryDecode8(f, 8, true, outArray, headers, extraCaptures)
        end
    end

    #println(position(f))
    close(f)
    #return (headers, extraCaptures)
end

## TO DO (feb20):
## Create importOvf!() -> puts the Mi comps. in the first argument, that must be a 4-dimensional array.
## Implement an automatic instantiation of a Mesh object using the headers[] data, and return it along with the Mi arrays.
## TO DO:
## Create functions to write/save to disk ovf files!

end