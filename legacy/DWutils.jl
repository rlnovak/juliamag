module DWutils

import DifferentialEquations:ODEProblem, solve
import Base:show

# Defines the functions q_dot and psi_dot from my 1D DW propagation model
#
#  Version 1.0 16Nov16. Last modified 26Mar20
#  Rafael L. Novak, UFSC, Brazil. rlnovak@gmail.com

export params, evolve!, dwqΨ!, fieldRampUp!, fieldRampDown!, histeresis!
export Geometry, Material, Simulation, State, Cobalt, Stripe, stdField, stdSim
# export appliedH

## Geometry definition:
struct Geometry{T}
    Lx::T
    Ly::T
    Lz::T
end

function Geometry(Lx, Ly, Lz)
    if typeof(Lx) != Float64
        convert(Float64, Lx)
    end
    if typeof(Ly) != Float64
        convert(Float64, Ly)
    end
    if typeof(Lz) != Float64
        convert(Float64, Lz)
    end
    Geometry(Lx, Ly, Lz)
end

function Geometry(l::Array{T,1} where T <: AbstractFloat)
    if eltype(l) != Float64
        convert.(Float64, l)
    end
    return Geometry(l[1], l[2], l[3])
end

function Base.show(io::IO, g::Geometry)
    println("Geometry:")
    println("Lx = $(g.Lx) cm")
    println("Ly = $(g.Ly) cm")
    println("Lz = $(g.Lz) cm")
end

## Material definition:
struct Material
    α::Float64
    γ::Float64
    δ::Float64
    Ms::Float64
end

function Base.show(io::IO, m::Material)
    println("Material:")
    println("α = $(m.α)")
    println("γ = $(m.γ)")
    println("δ = $(m.δ)")
    println("Ms = $(m.Ms)")
end

# struct AppliedH
#     initialH::Float64
#     finalH::Float64
#     direction::Int64
#     step::Float64
#     function appliedH(init, final, dir::Char, step)
#         if dir == 'x' direction = 1
#         elseif dir == 'y' direction = 2
#         elseif dir == 'z' direction = 3
#         end
#         new AppliedH(init, final, direction, step)
#     end
# end

## Simulation state definition:
struct State
    pos::Float64
    angle::Float64
    field::Float64
    time::Float64
    Δ::Float64
    up::Bool
end

function Base.show(io::IO, s::State)
    print("pos \t angle \t field \t time \t Δ \t up \n")
    print("$(s.pos) \t $(s.angle) \t $(s.field) \t $(s.time) \t $(s.Δ) \t $(s.up) \n")
end

## Simulation definition:
mutable struct Simulation
    mat::Material
    geom::Geometry
    Hk::Float64
    # H::AppliedH
    initialH::Float64 # Modify constructor for Int field arguments!
    finalH::Float64
    stepH::Float64
    state::State
    tolerance::Float64
    results::Array{Float64,2} # Matrix cols.: 1 -> HZ 2 -> pos 3 -> pos/Lx 4 -> angle 5 -> time
end

function Base.show(io::IO, s::Simulation)
    println("1D micromagnetic simulation:")
    println("Material -> $(s.mat)")
    println("Geometry -> $(s.geom)")
    println("Anisotropy field -> $(s.Hk)")
    println("Initial field -> $(s.initialH)")
    println("Final field -> $(s.finalH)")
    println("Field step -> $(s.stepH)")
    println("DW position tolerance -> $(s.tolerance)")
end

function Simulation(mat, geom, state)
    Hk = 4*mat.Ms*geom.Lz*log(2)/mat.δ
    Simulation(mat, geom, Hk, 0.0, 1000.0, 10.0, state, 1e-9, [0.0 500e-7 0.25 0.75 0.0]) # Default field parameters
    # Mehorar init do results, com as verdadeiras conds. iniciais passadas como parametro para a função.
end

function Simulation(mat, geom, initialH, finalH, stepH, state, tol) # Outer constructor -> field parameters passed as input.
    Hk = 4*mat.Ms*geom.Lz*log(2)/mat.δ
    return Simulation(mat, geom, Hk, initialH, finalH, stepH, state, tol, [0.0 500e-7 0.25 0.75 0.0])
    # Mehorar init do results, com as verdadeiras conds. iniciais passadas como parametro para a função.
end

## Default variables for testing:
cobalt = Material(0.5, 1.76086e7, 6.3e-7, 1135)
stripe = Geometry([2000e-7, 512e-7, 2e-7])
# stdField = AppliedH(0.0, 600.0, 'z', 10.0)
initState = State(20e-9, 0.00002, 0.0, 0.0, 0.0, true)
stdSim = Simulation(cobalt, stripe, 0.0, 600.0, 10.0, initState, 1e-9)
dwPosition(sim::Simulation) = sim.state.pos

## Definitions
# α = 0.5
# # CGS constants
# γ = 1.76086e7 # MHz/Oe according to Thomas et al. Nature
# δ = 6.3e-7 # cm
# Ms = 1135 # emu/cm3
# Ly = 512e-7 # cm
# Lz = 2e-7 # cm
# Hk =  4*Ms*Lz*log(2)/δ # Anisotropy field in Oe
# # inv_α = 1/(1+α^2)

# Global: applied out-of-plane field
# global HZ
####################################

# function Δ(sim::Simulation)
#     aux = u[end] - u[end-1]
#     return aux
# end

# function dwqΨ!(du, u, p, t)
#     # Parameters: α, γ, δ, HZ, Hk, Ms, Ly, Lz
#     α, γ, δ, HZ, Hk, Ms, Ly, Lz = p
#     ####### CGS, wrong! Correct these CGS equations! #######
#     du[1] = (1.0/(1.0+α^2))*(α*γ*δ*HZ + 0.5*γ*δ*Hk*sin(2*u[2]) - α*γ*δ/(2*Ms*Ly*Lz)*dPot(u[1]))
#     du[2] = (1.0/(1.0+α^2))*(γ*HZ - 0.5*γ*α*Hk*sin(2*u[2]) - γ/(2*Ms*Ly*Lz)*dPot(u[1]))
#     ########################################################
#     ## SI ##
#     # The potential term is too high! I multiplied by 2e-9 to reduce it and get
#     # more sensible results
#     #fn(1) =  443.7367*HZ - 3.5276563e7*sin(2*qPsi(2)) - 2e-9*2.3862e17*dPot(qPsi(1));
#     #fn(2) = 1.4087e11*HZ + 2.7997e15*sin(2*qPsi(2)) - 2e-9*7.5755e25*dPot(qPsi(1));
# end
function dwqΨ!(du, u, p, t)
    α, γ, δ, HZ, Hk, Ms, Ly, Lz = p
    ####### CGS, wrong! Correct these CGS equations! #######
    du[1] = (1.0/(1.0+α^2))*(α*γ*δ*HZ + 0.5*γ*δ*Hk*sin(2*u[2]) - α*γ*δ/(2*Ms*Ly*Lz)*dPot(u[1]))
    du[2] = (1.0/(1.0+α^2))*(γ*HZ - 0.5*γ*α*Hk*sin(2*u[2]) - γ/(2*Ms*Ly*Lz)*dPot(u[1]))
end

function params(sim::Simulation)
    α = sim.mat.α
    γ = sim.mat.γ
    δ = sim.mat.δ
    H = sim.state.field
    Hk = sim.Hk
    Ms = sim.mat.Ms
    Ly = sim.geom.Ly
    Lz = sim.geom.Lz
    return [α, γ, δ, H, Hk, Ms, Ly, Lz]
end

function dPot(x)
    # Defines the potential derivative from my 1D DW propagation model
    # The potential is derived from a fit to micromagnetic simulation results.
    #
    #  Version 1.0 16Nov16. Last modified 26Mar20
    #  Rafael L. Novak, UFSC, Brazil. rlnovak@gmail.com

    # Functions in CGS. Dimensions e-7 -> e-9 for SI

        if (x < 710e-7) || (x > 1350e-7)
            pn = 0.0
            return pn
        #elseif (x >= 680e-9) && (x < 710e-9)
            #pn = -7.4733e-4*x + 5.0819e-10;
        #elseif (x >= 710e-9) && (x <= 1350e-9)
        else
            # CGS expression
            pn = -7.29753e-6 + 0.07084*x + (-0.481616*exp(-1.5536e10*(x-0.000109)^2)*(x-0.000109))
            return pn
            # 0.5 because I believe the potential was twice the original value...
            # SI expression
            #pn = -7.2975e-11 + 7.085e-5*x + (-4.33099e-4*exp(-1.1785e14*(x-1.09e-6)^2)*(x-1.09e-6));
        #else
            #pn = -7.4633e-4*x + 1.0299e-9;
        end
    end

## Time evolution of simulation.
# The simulation will evolve every tmax interval until the domain stops moving
# under the applied field.
function evolve!(sim::Simulation; tmax = 5e-10)
    tspan = (0.0, tmax) # What time span to use ????
    p = params(sim)
    currentState = sim.state
    rampUp = currentState.up
    currentTime = currentState.time
    u0 = [currentState.pos, currentState.angle]
    prob = ODEProblem(dwqΨ!, u0, tspan, p)
    sol = solve(prob) # sol.u is an Array{Array{Float64,1},1}. Each element of the outer array is an inner array with [pos, angle]
    m = size(sol.u,1)
    i = m
    Lx = sim.geom.Lx
    newPos = sol.u[i][1]
    while newPos > Lx
        i -= 1
        newPos = sol.u[i][1]
    end
    newΨ = sol.u[i][2]
    newTime = sol.t[i] # sol.t is an Array{Float64,1}
    if i > 1
        δ = abs(newPos-sol.u[i-1][1])
    else
        δ = abs(sol.u[2][1]-sol.u[1][1])
    end
    H = p[4]
    sim.state = State(newPos, newΨ, H, currentTime+newTime, δ, rampUp)
end

# function updateResults!(sim::Simulation)
#     sim.results = vcat(sim.results, [sim.state.field newPos newPos/sim.geom.Lx newΨ newTime])
# end

# function resetDelta!(sim::Simulation)
#     currentState = sim.state
#     pos = currentState.pos
#     angle = currentState.angle
#     field = currentState.field
#     time = currentState.time
#     newState = State(pos, angle, field, time, 1.0)
#     sim.state = newState
# end

function setField!(sim::Simulation, field)
    currentState = sim.state
    pos = currentState.pos
    angle = currentState.angle
    time = currentState.time
    δ = currentState.Δ
    currentField = currentState.field
    if field - currentField > 0
        rampUp = true
    elseif field - currentField < 0
        rampUp = false
    else
        rampUp = currentState.up
    end
    sim.state = State(pos, angle, field, time, δ, rampUp)
end

#positiveRamp(fields) = (fields[2]-field[1] > 0) ? true : false

function fieldRampUp!(sim::Simulation) # usar kwargs para passar plot. if plot, push!(...)
    # push!(resultPlot, [sim.results[end,1]], [sim.results[end,2]])
    println(sim.state)
    pos = sim.state.pos
    Lx = sim.geom.Lx
    field = sim.initialH
    setField!(sim, field)
    fieldStep = sim.stepH
    dwTarget = 0.95*Lx
    #counter = 0
    #maxCounts = 2000
    while pos < dwTarget #&& counter < maxCounts
        evolve!(sim)
        # if counter % 200 == 0
        #     println("------------------------------")
        #     println(sim.state)
        #     println("counter = $(counter)")
        #     println("------------------------------")
        # end
        if sim.state.Δ < sim.tolerance
            pos = sim.state.pos
            Ψ = sim.state.angle
            time = sim.state.time
            sim.results = vcat(sim.results, [sim.state.field pos pos/sim.geom.Lx Ψ time])
            println("Ramp up -> Results appended.")
            field += fieldStep
            println(field)
            setField!(sim, field)
            #counter = 0
        else
            pos = sim.state.pos
            #println(pos)
            #counter += 1
        end
    end
end

function fieldRampDown!(sim::Simulation) # usar kwargs para passar plot. if plot, push!(...)
    # push!(resultPlot, [sim.results[end,1]], [sim.results[end,2]])
    println(sim.state)
    pos = sim.state.pos
    #println("pos = $(pos)")
    Lx = sim.geom.Lx
    #println("Lx = $(Lx)")
    field = -sim.stepH # sim.finalH originally.
    #println("field = $(field)")
    setField!(sim, field)
    fieldStep = sim.stepH
    dwTarget = 0.05*Lx
    #println("dwTarget = $(dwTarget)")
    #counter = 0
    #maxCounts = 2000
    while pos > dwTarget #&& counter < maxCounts
        #println("Iter. while loop... calling evolve()")
        evolve!(sim)
        # if counter % 200 == 0
        #     println("------------------------------")
        #     println(sim.state)
        #     println("counter = $(counter)")
        #     println("------------------------------")
        # end
        #println("evolve() finished executing...")
        if sim.state.Δ < sim.tolerance
            #println("Entered if... ")
            pos = sim.state.pos
            Ψ = sim.state.angle
            time = sim.state.time
            sim.results = vcat(sim.results, [sim.state.field pos pos/sim.geom.Lx Ψ time])
            println("Ramp down -> Results appended.")
            field -= fieldStep
            println(field)
            setField!(sim, field)
            #counter = 0
        else
            #println("Entered else...")
            pos = sim.state.pos
            #println("pos = $(pos)")
            #println(pos)
            #counter += 1
        end
    end
end

## Terminar!

# function plotRamp!(resultsPlot, sim::Simulation)
#     # Tem ifexists, ou algo assim??
#     try
#         push!(resultPlot, [sim.results[:,1], [sim.results[:,2]])
#     catch e
#         println("The resultsPlot variable is not defined!")
#     end
# end

function histeresis!(sim::Simulation)
    ## Histerese
    # De +initH a +Hmax
    println("Ramp from $(sim.initialH) to $(sim.finalH)")
    fieldRampUp!(sim)
    # De H = Hmax-stepH a -Hmax
    println("Ramp from $(sim.finalH) to -$(sim.finalH)")
    fieldRampDown!(sim)
    # De H = -Hmax+stepH a +Hmax
    println("Ramp from -$(sim.finalH) to $(sim.finalH)")
    fieldRampUp!(sim)
end

# Tem que fazer evolve! ate a parede parar. Isso tem que ser avaliado,e quando a parede parar
# a gente grava o results. A recursão de evolve está dando stack overflow

## Empirical magnetostatic potential from the overlying permalloy disk

end

"""
Garantir que rampUp ou rampDown cheguem aos valores extremos de campo. Não estão chegando!

"""
