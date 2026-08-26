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
export togpu, tocpu

# --- Shapes (geometry) -----------------------------------------------------
include("shape.jl")
export Shape, translate, scale, rotz, rotx, roty, mirror, repeat_shape
export Cuboid, Rect, Square, Cylinder, Circle, Ellipsoid, Ellipse, Cone,
       Superball, XRange, YRange, ZRange, Layer, Layers, Universe, Empty,
       Triangle, Line, Line2D, Cell
export shapeunion, shapeintersect, shapediff, shapecomplement, shapexor

# --- Regions ---------------------------------------------------------------
include("regions.jl")
export Regions, defregion!, defregioncell!, regionlist, regionvolume, cellcenter

# --- Per-region material parameters ----------------------------------------
include("region_params.jl")
export RegionParams, setregion!, setgeometry!, clearempty!

# --- Polycrystalline grains (Voronoi) --------------------------------------
include("voronoi.jl")
export voronoi!, randomanisotropy!

# --- Initial configurations ------------------------------------------------
include("config.jl")
export Config, setconfig, setconfig!
export UniformConfig, VortexConfig, AntiVortexConfig, NeelSkyrmionConfig,
       BlochSkyrmionConfig, VortexWallConfig, TwoDomainConfig, RandomConfig

# --- OVF I/O ---------------------------------------------------------------
include("ovf.jl")
export loadovf, saveovf, meshfromovf

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

# --- Energies --------------------------------------------------------------
include("energy.jl")
export fieldenergy, exchangeenergy, anisotropyenergy, dmienergy, demagenergy,
       zeemanenergy, totalenergy

# --- Spin-transfer torque --------------------------------------------------
include("stt.jl")
export zhanglitorque!, slonczewskitorque!

# --- Feature trackers ------------------------------------------------------
include("trackers.jl")
export vortexcore, skyrmionpos, domainwallpos, topologicalcharge

# --- Data output -----------------------------------------------------------
include("output.jl")
export Quantity, DataTable, tableadd!, tablesave!, writetable, tableheader
export q_time, q_m, q_m_region, q_energy, q_exchangeenergy, q_demagenergy,
       q_zeemanenergy, q_anisenergy, q_maxtorque, q_Bext, q_vortexcore,
       q_skyrmionpos, q_dwpos, q_topocharge
export average_region

# --- Material library ------------------------------------------------------
include("materials.jl")
export material, materialnames

# --- Simulation wrapper ----------------------------------------------------
include("simulation.jl")
export Simulation, setmag!, savequantities!, savenow!, run!, runcurrent!

# --- Finite-temperature (Langevin) dynamics --------------------------------
include("thermal.jl")
export thermalfield!, runthermal!

end # module
