# Data table output, following mumax3 (engine/table.go).
#
# A DataTable is an ordered list of Quantities; each row appends the current
# value of every quantity, tab-separated, with a header of "name (unit)" columns.
# The user builds the table with tableadd!, writes rows on demand with
# tablesave!, or automatically every `autosave` seconds of simulated time.
#
# A Quantity is any callable `q(world, m, t) -> Vector{Float64}` carrying a name,
# a unit, its component count, and whether it can be evaluated on a GPU state
# (`device`); a Simulation on the GPU evaluates the others on a host copy of m. The library quantities below cover what mumax3
# tabulates: time, averaged magnetization (total and per region), the energies,
# the maximum torque, the applied field and current, and the vortex-core,
# skyrmion, and domain-wall positions and topological charge.

"""
    Quantity(name, unit, ncomp, f; device=false)

A tabulatable quantity: `name`, `unit`, number of components `ncomp`, and a
function `f(world, m, t)` returning `ncomp` Float64 values. `device = true` declares
that `f` also works when `m` and `world` live on the GPU; otherwise a GPU
[`Simulation`](@ref) evaluates it on a host copy of the state (with a host World).
"""
struct Quantity
    name::String
    unit::String
    ncomp::Int
    f::Function
    device::Bool
end
Quantity(name, unit, ncomp, f; device::Bool = false) = Quantity(name, unit, ncomp, f, device)

(q::Quantity)(world, m, t) = q.f(world, m, t)

# --- Library quantities ----------------------------------------------------

"Simulated time [s]."
q_time() = Quantity("t", "s", 1, (w, m, t) -> [Float64(t)]; device = true)

"""
Spatially averaged magnetization ⟨mx,my,mz⟩ over the magnet — mumax3's table `m`:
cells weighted by their fill, empty cells excluded ([`average_magnet`](@ref)). For a
scalar `Material` (no geometry) this is the plain mesh average.
"""
q_m() = Quantity("m", "", 3, (w, m, t) -> collect(Float64.(average_magnet(m, w.material))); device = true)

"Averaged magnetization over one region."
function q_m_region(id::Integer)
    Quantity("m_region$id", "", 3, (w, m, t) -> collect(Float64.(average_region(m, w.material, id))))
end

"Total energy [J]."
q_energy() = Quantity("E_total", "J", 1, (w, m, t) -> [Float64(totalenergy(m, w))])
"Exchange energy [J]."
q_exchangeenergy() = Quantity("E_exch", "J", 1, (w, m, t) -> [Float64(exchangeenergy(m, w.mesh, w.material))])
"Demag energy [J]."
q_demagenergy() = Quantity("E_demag", "J", 1,
    (w, m, t) -> [w.demagplan === nothing ? 0.0 : Float64(demagenergy(m, w.demagplan, w.mesh, w.material))])
"Zeeman energy [J]."
q_zeemanenergy() = Quantity("E_zeeman", "J", 1, (w, m, t) -> [Float64(zeemanenergy(m, w))])
"Anisotropy energy [J]."
q_anisenergy() = Quantity("E_anis", "J", 1, (w, m, t) -> [Float64(anisotropyenergy(m, w.mesh, w.material))])

"""
Maximum torque over all cells [T], as mumax3's `maxTorque`: max |dm/dt| / γLL of the
full right-hand side ([`llgrhs!`](@ref)), i.e. the LLG torque plus the spin-transfer
torques of a current set with `setcurrent!`.
"""
function q_maxtorque()
    Quantity("maxTorque", "T", 1, function (w, m, t)
        dm = similar(m)
        llgrhs!(dm, m, w)
        [Float64(maxtorque(dm)) / γLL]
    end; device = true)
end

"Applied field [T], averaged over the mesh (the uniform part and the terms of `addfield!`)."
q_Bext() = Quantity("B_ext", "T", 3, function (w, m, t)
    b = collect(Float64.(w.Bext))
    for q in w.Bterms
        n = length(q.map) ÷ 3
        b .+= q.c .* [Float64(sum(Float64, view(q.map, c:c, :, :, :))) / n for c in 1:3]
    end
    b
end; device = true)
"Current density [A/m²], averaged over the mesh (the uniform part and the terms of `addcurrent!`)."
q_J() = Quantity("J", "A/m2", 3, function (w, m, t)
    w.current === nothing && return zeros(3)
    b = collect(Float64.(w.current.J))
    for q in w.Jterms
        n = length(q.map) ÷ 3
        b .+= q.c .* [Float64(sum(Float64, view(q.map, c:c, :, :, :))) / n for c in 1:3]
    end
    b
end; device = true)

"Vortex core position (x,y,z) [m] and polarity."
q_vortexcore() = Quantity("vortex", "m", 4, (w, m, t) -> collect(Float64.(vortexcore(m, w.mesh))); device = true)
"Skyrmion position (x,y,z) [m]."
q_skyrmionpos() = Quantity("skyrmion", "m", 3, (w, m, t) -> collect(Float64.(skyrmionpos(m, w.mesh))); device = true)
"Domain-wall position (x,y,z) [m]."
q_dwpos() = Quantity("dwpos", "m", 3, (w, m, t) -> collect(Float64.(domainwallpos(m, w.mesh))); device = true)
"Topological charge."
q_topocharge() = Quantity("Q", "", 1, (w, m, t) -> [Float64(topologicalcharge(m, w.mesh))]; device = true)

# --- DataTable -------------------------------------------------------------

"""
    DataTable(; autosave=0.0)

An output table. Add quantities with [`tableadd!`](@ref) (or the `columns`
keyword), then append rows with [`tablesave!`](@ref). If `autosave > 0`, the
run loop appends a row every `autosave` seconds of simulated time, with mumax3's
rule (engine/autosave.go): a row is written after the first step at which
`t − start ≥ (count + 1)·autosave`, counting from the time the interval was set
(so the first row is the state at that time).
"""
mutable struct DataTable
    quantities::Vector{Quantity}
    autosave::Float64          # save interval [s]; 0 disables
    nextsave::Float64          # kept for compatibility (unused)
    rows::Vector{Vector{Float64}}
    start::Float64             # time the autosave interval was set
    count::Int                 # autosaves done since `start` (−1: none yet)
    path::Union{Nothing,String}   # stream rows to this file as they are saved
    written::Bool                 # the header has been written to `path`
end

function DataTable(; autosave = 0.0, columns = Quantity[], path = nothing, start = 0.0)
    qs = isempty(columns) ? [q_time(), q_m()] : collect(columns)
    DataTable(qs, Float64(autosave), 0.0, Vector{Float64}[], Float64(start), -1, path, false)
end

# mumax3's autosave.needSave: period != 0 && (t − start) − count·period ≥ period. A
# relative slack of 1e-9·period absorbs the rounding of a time accumulated in fixed
# steps, so that rows land on the step that reaches the save time.
_needsave(period, start, count, t) =
    period != 0 && (t - start) - count * period >= period * (1 - 1e-9)

"Add a quantity as one or more columns."
function tableadd!(tbl::DataTable, q::Quantity)
    (tbl.written || !isempty(tbl.rows)) &&
        throw(ArgumentError("tableadd!: the table already has rows; add every column before the first row is saved (as mumax3's TableAdd)"))
    push!(tbl.quantities, q)
    return tbl
end

"Column header strings, one per scalar component."
function tableheader(tbl::DataTable)
    hdr = String[]
    for q in tbl.quantities
        if q.ncomp == 1
            push!(hdr, q.unit == "" ? q.name : "$(q.name) ($(q.unit))")
        else
            for c in 1:q.ncomp
                base = "$(q.name)$(('x','y','z','w')[min(c,4)])"
                push!(hdr, q.unit == "" ? base : "$base ($(q.unit))")
            end
        end
    end
    return hdr
end

"""
    tablesave!(tbl, world, m, t)

Evaluate every quantity and append one row.
"""
function tablesave!(tbl::DataTable, world, m, t)
    row = Float64[]
    for q in tbl.quantities
        append!(row, q(world, m, t))
    end
    _pushrow!(tbl, row)
    return tbl
end

# Append a row, and stream it to tbl.path if set. The first row of a table starts
# the file afresh (header first), as mumax3 does: a rerun into the same output
# directory replaces its table instead of appending to the old one.
function _pushrow!(tbl::DataTable, row)
    push!(tbl.rows, row)
    if tbl.path !== nothing
        open(tbl.path, tbl.written ? "a" : "w") do io
            tbl.written || println(io, "# ", join(tableheader(tbl), "\t"))
            println(io, join(row, "\t"))
        end
        tbl.written = true
    end
    return tbl
end

"""
    writetable(tbl, path)

Write the table to `path` as a tab-separated file with a header line.
"""
function writetable(tbl::DataTable, path::AbstractString)
    open(path, "w") do io
        println(io, "# ", join(tableheader(tbl), "\t"))
        for row in tbl.rows
            println(io, join(row, "\t"))
        end
    end
    return path
end
