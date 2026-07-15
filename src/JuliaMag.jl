"""
    JuliaMag

Micromagnetic simulation in pure Julia, on CPU and (later) GPU.

The magnetization is a reduced vector field `m = M / Msat` stored as an
`Array{T,4}` of shape `(3, Nx, Ny, Nz)`. Every routine dispatches on the array
type, so the same code runs on a `CuArray` once GPU methods are added.

Inspired by mumax3 (Vansteenkiste et al., AIP Advances 4, 107133 (2014)).
"""
module JuliaMag

using Random: AbstractRNG
import Random
using LinearAlgebra: mul!

# --- Constants -------------------------------------------------------------
include("constants.jl")
export μ0, μB, kB, qe, γLL

# --- Geometry --------------------------------------------------------------
include("mesh.jl")
export Mesh, cellvolume, volume, worldsize, isperiodic, padsize, dataregion

# --- Material --------------------------------------------------------------
include("params.jl")
export Material, exchangelength

# --- Magnetization ---------------------------------------------------------
include("magnetization.jl")
export zeromag, uniform, uniform!, randommag, randommag!, vortex, vortex!, normalize!, average

# --- Effective-field terms -------------------------------------------------
include("exchange.jl")
include("anisotropy.jl")
include("dmi.jl")
include("zeeman.jl")
export exchange!, anisotropy!, dmi!, zeeman!

# --- Demagnetization -------------------------------------------------------
include("demag_kernel.jl")
export DemagKernel, demagkernel
include("demag_field.jl")
export DemagPlan, demagfield!

# --- Dynamics --------------------------------------------------------------
include("effective_field.jl")
export World, setexternalfield!, effectivefield!
include("llg.jl")
export torque!, maxtorque
include("solver.jl")
export Solver, step!, runtime!, relax!
include("minimizer.jl")
export Minimizer, minimizestep!, minimize!
include("integrator.jl")
export Integrator, advance!, relaxate!, currenttime, state

end # module
