module DemagKernel
# Calculates the demagnetization tensor (kernel) based on the input mesh geometry by brute force integration.
# Returns 3x3 matrix where each of the elements is a 3D array with mesh.size dimensions
# Code based on https://mumax.github.io/ A. Vansteenkiste at al., "The design and verification of Mumax3", AIP Advances 4 107133 (2014).
# Rafael L. Novak, rlnovak@gmail.com, jan2020, UFSC/Blumenau (Brazil).

using MeshGeometry

export demagKernel, calculateKernel

## Valeria a pena criar um struct para o kernel??

function demagKernel(mesh::Mesh, acc::Float64, cacheDir::String)
    """
    Kernel structure:
    
    | Kxx  Kxy  Kxz |
    | Kxy  Kyy  Kyz |
    | Kxz  Kzy  Kzz |
    
    where the Kij are Array{Float64,3}.

    Could I store it in linear fashion, as [--Kxx--Kyx--Kzx--Kxy--Kyy--Kzy--Kxz--Kyz--Kzz--]?
    Each Kij has Nx*Ny*Nz elements, where mesh.size = [Nx, Ny, Nz] + size increases due to zero padding.
    """
    ## TODO: Try to load kernel from temp directory, if file exists. Filename must be standardized according to mesh geometry.
    ## If file not present, calculate demagnetization tensor (kernel) by means of brute force integration of sources on
    ## the mesh (surface /volume charges on mesh cells).
    ## After calculating, the mesh array should be available for the simulation and must be saved for future use (use HDF5?).

    meshsize = mesh.gridsize
    meshPBC = mesh.pbc
    cellsizes = mesh.cellsize
    accuracy = 6.0 # Demag accuracy (divide cubes in at most N^3 points)
    kernel = calculateKernel(meshsize, meshPBC, cellsizes, accuracy)
    ## save(kernel, location/filename)
    return kernel
end

function calculateKernel(mesh::Mesh; accuracy=6.0)
    ###############################
    # Index -> Component convention
    X = 1::Int # 1 -> X
    Y = 2::Int # 2 -> Y
    Z = 3::Int # 3 -> Z
    ################################
    meshsize = mesh.gridsize
    meshPBC = mesh.pbc
    cellsize = mesh.cellsize
	# Add zero-padding in non-PBC directions
    size_padded = padSize(meshsize, meshPBC)
    println("size_padded -> ",size_padded)

    @assert size_padded[Z] > 0 && size_padded[Y] > 0 && size_padded[X] > 0
    @assert cellsize[X] > 0 && cellsize[Y] > 0 && cellsize[Z] > 0
    @assert meshPBC[X] >= 0 && meshPBC[Y] >= 0 && meshPBC[Z] >= 0
    @assert accuracy > 0

    # Initialize only upper diagonal part. The rest is symmetric due to reciprocity.
    kernel = Array{Array{Float64,3},2}(undef,3,3)
    kernel[X,X] = zeros(size_padded[X],size_padded[Y],size_padded[Z])
    kernel[X,Y] = zeros(size_padded[X],size_padded[Y],size_padded[Z])
    kernel[X,Z] = zeros(size_padded[X],size_padded[Y],size_padded[Z])
    kernel[Y,Y] = zeros(size_padded[X],size_padded[Y],size_padded[Z])
    kernel[Y,Z] = zeros(size_padded[X],size_padded[Y],size_padded[Z])
    kernel[Z,Z] = zeros(size_padded[X],size_padded[Y],size_padded[Z])

	# Field (destination) loop ranges
    r1, r2 = kernelRanges(size_padded, meshPBC)
    println("r1 -> ",r1)
    println("r2 -> ",r2)
	# smallest cell dimension is our typical length scale
	L = cellsize[X]
    if cellsize[Y] < L
        L = cellsize[Y]
    end
    if cellsize[Z] < L
        L = cellsize[Z]
    end
	# Start brute integration
	# 9 nested loops, does that stress you out?
	# Fortunately, the 5 inner ones usually loop over just one element.

    for s in 0:2 # source index Ksdxyz --> K[s,d,x,y,z]?
        println("Calculating demag kernel, line $s.")
        ##### MODIFY FOR 1-BASED INDEXING !!!!!! ####
        u, v, w = s, (s+1)%3, (s+2)%3 # u = direction of source (s), v & w are the orthogonal directions
        u += 1
        v += 1
        w += 1
        ########################
        R = Array{Float64,1}(undef,3) # field cell center positions
        R2 = Array{Float64,1}(undef,3) # source cell center positions
        pole = Array{Float64,1}(undef,3) # position of point charge on the surface
        #points = 1 # Int, counts used integration points, not used in any calculation.... should be zero-initialized?

        for z in r1[Z]:r2[Z]
            zw = wrap(z, size_padded[Z])
            # skip one half, reconstruct from symmetry later
            # check on wrapped index instead of loop range so it also works for PBC
            if zw > size_padded[Z]/2
                continue
            end
            R[Z] = z * cellsize[Z]

            for y in r1[Y]:r2[Y]
                yw = wrap(y, size_padded[Y])
                if yw > size_padded[Y]/2
                    continue
                end
                R[Y] = y * cellsize[Y]

                for x in r1[X]:r2[X]
                    xw = wrap(x, size_padded[X])
                    if xw > size_padded[X]/2
                        continue
                    end
                    R[X] = x * cellsize[X]

                    # choose number of integration points depending on how far we are from source.
                    dx, dy, dz = delta(x)*cellsize[X], delta(y)*cellsize[Y], delta(z)*cellsize[Z]
                    d = sqrt(dx*dx + dy*dy + dz*dz)
                    if d == 0
                        d = L
                    end
                    maxSize = d / accuracy # maximum acceptable integration size

                    nv = round(Int64, max(cellsize[v]/maxSize, 1.0) + 0.5)
                    nw = round(Int64, max(cellsize[w]/maxSize, 1.0) + 0.5)
                    nx = round(Int64, max(cellsize[X]/maxSize, 1.0) + 0.5)
                    ny = round(Int64, max(cellsize[Y]/maxSize, 1.0) + 0.5)
                    nz = round(Int64, max(cellsize[Z]/maxSize, 1.0) + 0.5)
                    # Stagger source and destination grids.
                    # Massively improves accuracy, see note.
                    nv *= 2
                    nw *= 2

                    @assert nv > 0 && nw > 0 && nx > 0 && ny > 0 && nz > 0

                    scale = 1.0 / nv*nw*nx*ny*nz
                    surface = cellsize[v] * cellsize[w] # the two directions perpendicular to direction s
                    charge = surface * scale
                    pu1 = cellsize[u]/2.0 # positive pole center
                    pu2 = -pu1             # negative pole center

                    # Do surface integral over source cell, accumulate  in B
                    B = Array{Float64, 1}(undef, 3)
                    for i in 1:nv
                        pv = -(cellsize[v] / 2.0) + cellsize[v]/(2*nv) + i*(cellsize[v]/nv)
                        pole[v] = pv
                        for j in 1:nw
                            pw = -(cellsize[w] / 2.0) + cellsize[w]/(2*nw) + j*(cellsize[w]/nw)
                            pole[w] = pw
                            # Do volume integral over destination cell
                            for α in 1:nx
                                rx = R[X] - cellsize[X]/2.0 + cellsize[X]/(2*nx) + (cellsize[X]/nx)*α
                                for β in 1:ny
                                    ry = R[Y] - cellsize[Y]/2.0 + cellsize[Y]/(2*ny) + (cellsize[Y]/ny)*β
                                    for γ in 1:nz
                                        rz = R[Z] - cellsize[Z]/2.0 + cellsize[Z]/(2*nz) + (cellsize[Z]/nz)*γ
                                        #points++
                                        pole[u] = pu1
                                        R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
                                        r = sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
                                        qr = charge / (4 * π * r * r * r)
                                        bx = R2[X] * qr
                                        by = R2[Y] * qr
                                        bz = R2[Z] * qr

                                        pole[u] = pu2
                                        R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
                                        r = sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
                                        qr = -charge / (4 * π * r * r * r)
                                        B[X] += (bx + R2[X]*qr) # addition ordered for accuracy
                                        B[Y] += (by + R2[Y]*qr)
                                        B[Z] += (bz + R2[Z]*qr)
                                    end
                                end
                            end
                        end
                    end
                    for d in (s+1):3 # destination index Ksdxyz --> k[s,d,x,y,z] ?
                        println("zw = ",zw)
                        println("yw = ",yw)
                        println("xw = ",xw)
                        kernel[s+1][d][zw][yw][xw] += B[d] # += needed in case of PBC
                    end
                end
            end
        end
    end

	# Reconstruct skipped parts from symmetry (X)
	for z in 1:size_padded[Z]
		for y in 1:size_padded[Y]
			for x in (size_padded[X]/2 + 1):size_padded[X]
				x2 = size_padded[X] - x
				kernel[X][X][z][y][x] = kernel[X][X][z][y][x2]
				kernel[X][Y][z][y][x] = -kernel[X][Y][z][y][x2]
				kernel[X][Z][z][y][x] = -kernel[X][Z][z][y][x2]
				kernel[Y][Y][z][y][x] = kernel[Y][Y][z][y][x2]
				kernel[Y][Z][z][y][x] = kernel[Y][Z][z][y][x2]
				kernel[Z][Z][z][y][x] = kernel[Z][Z][z][y][x2]
            end
		end
	end

	# Reconstruct skipped parts from symmetry (Y)
	for z in 1:size_padded[Z]
		for y in (size_padded[Y]/2 + 1):size_padded[Y]
			y2 = size_padded[Y] - y
			for x in 1:size_padded[X]
				kernel[X][X][z][y][x] = kernel[X][X][z][y2][x]
				kernel[X][Y][z][y][x] = -kernel[X][Y][z][y2][x]
				kernel[X][Z][z][y][x] = kernel[X][Z][z][y2][x]
				kernel[Y][Y][z][y][x] = kernel[Y][Y][z][y2][x]
				kernel[Y][Z][z][y][x] = -kernel[Y][Z][z][y2][x]
				kernel[Z][Z][z][y][x] = kernel[Z][Z][z][y2][x]

            end
		end
	end

	# Reconstruct skipped parts from symmetry (Z)
	for z in (size_padded[Z]/2 + 1):size_padded[Z]
		z2 = size_padded[Z] - z
		for y in 1:size_padded[Y]
			for x in 1:size_padded[X]
				kernel[X][X][z][y][x] = kernel[X][X][z2][y][x]
				kernel[X][Y][z][y][x] = kernel[X][Y][z2][y][x]
				kernel[X][Z][z][y][x] = -kernel[X][Z][z2][y][x]
				kernel[Y][Y][z][y][x] = kernel[Y][Y][z2][y][x]
				kernel[Y][Z][z][y][x] = -kernel[Y][Z][z2][y][x]
				kernel[Z][Z][z][y][x] = kernel[Z][Z][z2][y][x]
            end
		end
	end

	# for 2D these elements are zero:
	if size_padded[Z] == 1
		kernel[X][Z] = zeros(size_padded)
		kernel[Y][Z] = zeros(size_padded)
    end
	# make result symmetric for tools that expect it so.
	kernel[Y][X] = kernel[X][Y]
	kernel[Z][X] = kernel[X][Z]
	kernel[Z][Y] = kernel[Y][Z]
    return kernel
end


# Returns the size after zero-padding, taking into account periodic boundary conditions.
# In a certain direction, there is no padding in case of PBC (it should wrap around).
# Without PBC there should be zero padding up to at least 2*N - 1. In that case there
# is a trade-off: for large N, padding up to 2*N can be much more efficient since
# power-of-two sized FFT's are ludicrously fast on CUDA. However for very small N,
# in particular N=1, we should not over-pad.
function padSize(meshsize, periodic)
    ###############################
    # Index -> Component convention
    X = 1::Int # 1 -> X
    Y = 2::Int # 2 -> Y
    Z = 3::Int # 3 -> Z
    ################################
    SMALL_N = 5::Int
	padded = Array{Int64, 1}(undef,3)
	for (i,s) in enumerate(meshsize)
		if periodic[i] != 0
			padded[i] = s
			continue
        end
		if i != Z || s > SMALL_N # for some reason it only works for Z, perhaps we assume even FFT size elsewhere?
			# large N: zero pad * 2 for FFT performance
			padded[i] = 2*s
		else
			# small N: minimal zero padding for memory/performance ## Zero-padding even if meshsize[Z] == 1??
			padded[i] = 2*s - 1
        end
	end
	return padded
end

# Use 2N-1 padding instead of 2N for sizes up to SMALL_N.
# 5 seems a good choice since for all n<=5, 2*n-1 only has
# prime factors 2,3,5,7 (good CUFFT performance).
# starting from 6 it becomes problematic so we use 2*n.

# integration ranges for kernel. size=kernelsize, so padded for no PBC, not padded for PBC
function kernelRanges(meshsize, pbc)
    ###############################
    # Index -> Component convention
    X = 1::Int # 1 -> X
    Y = 2::Int # 2 -> Y
    Z = 3::Int # 3 -> Z
    ################################
    r1 = Array{Int64,1}(undef,3)
    r2 = Array{Int64,1}(undef,3)
    ## 3fev20 -> I eliminated the -1 inside the parentheses bec. of 1-based indexing.
	for c in 1:3
		if pbc[c] == 0
			r1[c], r2[c] = -(meshsize[c]+1) ÷ 2, (meshsize[c]+1) ÷ 2
		else
			r1[c], r2[c] = -(meshsize[c]*pbc[c]), (meshsize[c]*pbc[c]) # no /2 here, or we would take half right and half left image
        end
    end
	# support for 2D simulations (thickness 1)
	if meshsize[Z] == 1 && pbc[Z] == 0
		r2[Z] = 1
    end
	return r1, r2
end

# closest distance between cells, given center distance d.
# if cells touch by just even a corner, the distance is zero.
function delta(d::Int)
	if d < 1 # Was 0 because of 0-based indexing. CHECK!
		d = -d
    end
	if d > 1 # Was 0 because of 0-based indexing. CHECK!
		d -= 1
    end
	return Float64(d)
end

# Wraps an index to [0, max] by adding/subtracting a multiple of max.
function wrap(number::Int, max::Int)
	if number < 0 # Was 0 because of 0-based indexing. CHECK! # Now I am trying 1 (3fev20).
		number += max
    end
	if number >= max
		number -= max
    end
	return number
end

end