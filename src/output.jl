# Data table output, following mumax3 (engine/table.go).
#
# A DataTable is an ordered list of Quantities; each row appends the current
# value of every quantity, tab-separated, with a header of "name (unit)" columns.
# The user builds the table with tableadd!, writes rows on demand with
# tablesave!, or automatically every `autosave` seconds of simulated time.
#
# A Quantity is any callable `q(world, m, t) -> Vector{Float64}` carrying a name,
# a unit, and its component count. The library quantities below cover what mumax3
# tabulates: time, averaged magnetization (total and per region), the energies,
# the maximum torque, the applied field and current, and the vortex-core,
# skyrmion, and domain-wall positions and topological charge.

"""
    Quantity

A tabulatable quantity: `name`, `unit`, number of components `ncomp`, and a
function `f(world, m, t)` returning `ncomp` Float64 values.
"""
struct Quantity
    name::String
    unit::String
    ncomp::Int
    f::Function
end

(q::Quantity)(world, m, t) = q.f(world, m, t)

# --- Library quantities ----------------------------------------------------

"Simulated time [s]."
q_time() = Quantity("t", "s", 1, (w, m, t) -> [Float64(t)])

"Spatially averaged magnetization ⟨mx,my,mz⟩ over the whole sample."
q_m() = Quantity("m", "", 3, (w, m, t) -> collect(Float64.(average(m))))

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
q_zeemanenergy() = Quantity("E_zeeman", "J", 1, (w, m, t) -> [Float64(zeemanenergy(m, w.Bext, w.mesh, w.material))])
"Anisotropy energy [J]."
q_anisenergy() = Quantity("E_anis", "J", 1, (w, m, t) -> [Float64(anisotropyenergy(m, w.mesh, w.material))])

"Maximum torque over all cells [rad/s]."
function q_maxtorque()
    Quantity("maxTorque", "rad/s", 1, function (w, m, t)
        B = w._Bbuf
        effectivefield!(B, m, w)
        dm = similar(m)
        torque!(dm, m, B, damping(w.material))
        [Float64(maxtorque(dm))]
    end)
end

"Applied (Zeeman) field [T]."
q_Bext() = Quantity("B_ext", "T", 3, (w, m, t) -> collect(Float64.(w.Bext)))

"Vortex core position (x,y,z) [m] and polarity."
q_vortexcore() = Quantity("vortex", "m", 4, (w, m, t) -> collect(Float64.(vortexcore(m, w.mesh))))
"Skyrmion position (x,y,z) [m]."
q_skyrmionpos() = Quantity("skyrmion", "m", 3, (w, m, t) -> collect(Float64.(skyrmionpos(m, w.mesh))))
"Domain-wall position (x,y,z) [m]."
q_dwpos() = Quantity("dwpos", "m", 3, (w, m, t) -> collect(Float64.(domainwallpos(m, w.mesh))))
"Topological charge."
q_topocharge() = Quantity("Q", "", 1, (w, m, t) -> [Float64(topologicalcharge(m, w.mesh))])

# --- DataTable -------------------------------------------------------------

"""
    DataTable(; autosave=0.0)

An output table. Add quantities with [`tableadd!`](@ref) (or the `columns`
keyword), then append rows with [`tablesave!`](@ref). If `autosave > 0`, the
run loop appends a row every `autosave` seconds of simulated time.
"""
mutable struct DataTable
    quantities::Vector{Quantity}
    autosave::Float64          # save interval [s]; 0 disables
    nextsave::Float64          # next simulated time to auto-save at
    rows::Vector{Vector{Float64}}
end

function DataTable(; autosave = 0.0, columns = Quantity[])
    qs = isempty(columns) ? [q_time(), q_m()] : collect(columns)
    DataTable(qs, Float64(autosave), 0.0, Vector{Float64}[])
end

"Add a quantity as one or more columns."
tableadd!(tbl::DataTable, q::Quantity) = (push!(tbl.quantities, q); tbl)

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
    push!(tbl.rows, row)
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
