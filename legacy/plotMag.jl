""" Plotting functions for micromagnetic simulations.

   Rafael L. Novak, UFSC Blumenau. rlnovak@gmail.com
   Created: 3oct19. Modified: 3oct19
"""

module plotMag

using PyPlot, PyCall
@pyimport mpl_toolkits.axes_grid1 as axgrid
using Images

export plotComps, plotVector, meshgrid, meshgrid2

function string_as_varname_function(s::AbstractString, v::Any)
    s = Symbol(s)
    @eval (($s) = ($v))
end

## Outro meshgrid ##
function meshgrid2(x::AbstractRange{T},y::AbstractRange{T}; rev = false) where T <: AbstractFloat
    """Returns X and Y, matrices with size = (length(x), length(y)) with the integer indexes of the X and Y coords"""
    X = permutedims(repeat(x,1,length(y)),[2,1])
    if rev
        Y = repeat(y,1,length(x))
    else
        Y = repeat(reverse(y),1,length(x))
    end
    return X,Y
end
####################

function meshgrid(x,y; dims = 1, rev = false)
    """Returns X and Y, matrices with size = (length(x), length(y)) with the integer indexes of the X and Y coords"""
    lx = length(x)
    ly = length(y)
    if dims == 2 && rev == true
        return (permutedims(repeat(x, 1, length(y))), reverse(repeat(y, 1,length(x)), dims = 1))
    elseif dims == 2 && rev == false
        return (permutedims(repeat(x, 1, length(y))), repeat(y, 1,length(x)))
    else
        return (repeat(x, outer=length(y)), repeat(y, inner=length(x)))
    end
end

function plotComps(mag::Array{Float64, 4}; zslice = 1, save = false, savename = "")
    fig, axs = PyPlot.subplots(nrows = 3, ncols = 1, figsize=(8,8))
    ny, nx, nz, c = size(mag)
    pltx = axs[1].imshow(reverse(mag[:,:,zslice,1], dims = 1), cmap = "bwr", vmin = -1.0, vmax = 1.0)
    plty = axs[2].imshow(reverse(mag[:,:,zslice,2], dims = 1), cmap = "bwr", vmin = -1.0, vmax = 1.0)
    pltz = axs[3].imshow(reverse(mag[:,:,zslice,3], dims = 1), cmap = "seismic", vmin = -1.0, vmax = 1.0)
    compsdict = Dict(1 => "x", 2 => "y", 3 => "z")
    for i in 1:3
        string_as_varname_function("divider"*string(i), axgrid.make_axes_locatable(axs[i]))
        axs[i].set_xlabel("x [nm]")
        axs[i].set_ylabel("y [nm]")
        axs[i].set_ylim(0,ny)
        axs[i].set_title("m"*compsdict[i])
    end
    PyPlot.colorbar(pltx, cax=divider1.append_axes("right", size="2%", pad=0.1))
    PyPlot.colorbar(plty, cax=divider2.append_axes("right", size="2%", pad=0.1))
    PyPlot.colorbar(pltz, cax=divider3.append_axes("right", size="2%", pad=0.1))
    if save
        if savename == ""
            savename = "compsMag"
        end
        PyPlot.savefig(savename*".png")
    end
end

function plotVector(mag::Array{Float64, 4}; comp = 3, zslice = 1, qStep = 20, save = false, savename = "")
    fig, axs = PyPlot.subplots(nrows = 1, ncols = 1, figsize=(12,8))
    ny, nx, nz, c = size(mag)
    mx = mag[:,:,zslice,1]
    my = mag[:,:,zslice,2]
    mz = mag[:,:,zslice,3]
    pltz = axs.imshow(reverse(mag[:,:,zslice,comp], dims = 1), interpolation = "bilinear", cmap = "bwr", vmax=1.0, vmin=-1.0)
    divider = axgrid.make_axes_locatable(axs)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    PyPlot.colorbar(pltz, cax=cax)
    axs.set_xlabel("x [nm]")
    axs.set_ylabel("y [nm]")
    axs.set_ylim(0,ny)
    
    #### Quiver - vector field plot ######
    quiverStep = qStep
    coordsx = range(1, nx, length = nx)
    coordsy = range(1, ny, length = ny)
    X, Y = meshgrid(coordsx, coordsy, dims = 2, rev = false)
    Xq = X[1:quiverStep:end,1:quiverStep:end]
    Yq = Y[1:quiverStep:end,1:quiverStep:end]
    mxq = mag[1:quiverStep:end, 1:quiverStep:end, zslice, 1]
    myq = mag[1:quiverStep:end, 1:quiverStep:end, zslice, 2]
    #mzq = mag[1:quiverStep:end,1:quiverStep:end,zslice,comp]
    magNorm = sqrt.(mxq.^2 + myq.^2)
    axs.quiver(Xq, Yq, reverse(mxq, dims = 1), reverse(myq, dims = 1), reverse(myq, dims = 1), pivot = "mid", cmap = "seismic", alpha = 1, scale = 60, width = 0.002, headwidth = 3.2, headlength = 4.2)
    if save
        if savename == ""
            savename = "vectorMag"
        end
        savefig(savename*".png")
    end
end

# function writevideo(fname, imgstack::Array{<:Color,3};
#                     overwrite=true, fps=30::UInt, options=``)
#     ow = overwrite ? `-y` : `-n`
#     h, w, nframes = size(imgstack)

#     open(`ffmpeg
#             -loglevel warning
#             $ow
#             -f rawvideo
#             -pix_fmt rgb24
#             -s:v $(h)x$(w)
#             -r $fps
#             -i pipe:0
#             $options
#             -vf "transpose=0"
#             -pix_fmt yuv420p
#             $fname`, "w") do out
#         for i = 1:nframes
#             write(out, convert.(RGB{N0f8}, clamp01.(imgstack[:,:,i])))
#         end
#     end
# end

## Nao roda se carregado no iJulia!! Corrigir as escalas no comps. Corrigir o plot dos precos do apartamento (datas!).
# Consertar o ArrayFire.
# Calcular os tensores de desmagnetizacao. FAZER PORT NO WORDPRESS!

end