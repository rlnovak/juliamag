# A periodic stripe (infinite wire) with a Néel skyrmion driven by a current.
#
# Interfacial DMI plus perpendicular anisotropy stabilizes a Néel skyrmion on a
# thin stripe. Periodic boundary conditions along x make the stripe an infinite
# wire: when the skyrmion reaches one end it re-enters from the other and keeps
# moving, instead of being deflected or annihilated at an edge.
#
# A Zhang–Li spin current along +x drives the skyrmion against the current
# (electron flow), i.e. toward −x, with a small transverse skyrmion-Hall
# deflection. We drive it for 3 ns, save four OVF snapshots at the three regular
# sub-intervals (t = 0, 1, 2, 3 ns) showing the displacement (including the
# wrap-around re-entry), track the centroid every 20 ps, and fit the longitudinal
# velocity from the slope of the unwrapped x(t).
#
# Run:  julia --project=examples examples/skyrmion_drive.jl

using JuliaMag
using Printf
using Plots

# Unwrap a periodic coordinate stream of period L: undo the ±L jumps at re-entry
# so x(t) is a continuous line whose slope is the true velocity.
function unwrap(xs, L)
    out = copy(xs)
    for i in 2:length(out)
        d = out[i] - out[i-1]
        d >  L/2 && (out[i:end] .-= L)
        d < -L/2 && (out[i:end] .+= L)
    end
    return out
end

function main()
    mesh = Mesh((150, 100, 1), (2e-9, 2e-9, 1e-9); pbc = (1, 0, 0))  # 300×200×1 nm, periodic x
    Lx   = 150 * 2e-9
    mat  = Material(
        Msat  = 5.8e5,     # A/m
        Aex   = 1.5e-11,   # J/m
        alpha = 0.3,       # damping
        Ku    = 8e5,       # J/m³ perpendicular anisotropy…
        anisU = (0, 0, 1), # …along z
        Dind  = 3.0e-3,    # J/m² interfacial DMI (Néel skyrmion)
        pol   = 1.0,       # current spin polarization
        xi    = 0.2,       # Zhang–Li non-adiabaticity
    )
    sim = Simulation(mesh, mat; demag = true)

    # Seed a Néel skyrmion at the centre and relax it to equilibrium.
    setmag!(sim, NeelSkyrmionConfig(mesh; charge = 1, pol = -1))
    relax!(sim; stopdm = 1e-6)
    xs0, ys0, _ = skyrmionpos(sim.m, mesh)
    @printf("relaxed skyrmion at (%.1f, %.1f) nm, Q = %.2f\n",
            xs0 * 1e9, ys0 * 1e9, topologicalcharge(sim.m, mesh))

    # Table columns become: t, mx, my, mz, skyrmion x,y,z, Q  (t and ⟨m⟩ are
    # always the first columns, so skyrmion x is column 5).
    savequantities!(sim, q_skyrmionpos(), q_topocharge())

    # Drive with a current for 3 ns in three equal 1-ns intervals, saving an OVF
    # at each snapshot time (t = 0, 1, 2, 3 ns). The skyrmion re-enters the stripe
    # through the periodic seam.
    J        = (2e12, 0.0, 0.0)          # A/m² along +x
    duration = 3e-9
    nsnaps   = 3
    tsnap    = duration / nsnaps
    outdir   = @__DIR__

    saveovf(joinpath(outdir, "skyrmion_00.ovf"), sim.m, mesh; title = "skyrmion t=0.0ns")
    savenow!(sim)
    for s in 1:nsnaps
        runcurrent!(sim, J, tsnap; every = 20e-12)
        xsn, ysn, _ = skyrmionpos(sim.m, mesh)
        saveovf(joinpath(outdir, @sprintf("skyrmion_%02d.ovf", s)), sim.m, mesh;
                title = @sprintf("skyrmion t=%.2fns", sim.t * 1e9))
        @printf("t = %.2f ns  →  (x, y) = (%.1f, %.1f) nm\n", sim.t * 1e9, xsn * 1e9, ysn * 1e9)
    end
    writetable(sim.table, joinpath(outdir, "skyrmion_track.txt"))

    # Velocity from the unwrapped fine (20 ps) track: least-squares slope of x(t).
    t  = getindex.(sim.table.rows, 1)             # column 1 = time
    xw = unwrap(getindex.(sim.table.rows, 5), Lx) # column 5 = skyrmion x, unwrapped
    tt = t .- t[1]; xx = xw .- xw[1]
    v  = sum(tt .* xx) / sum(tt .^ 2)             # m/s, through the origin
    @printf("skyrmion velocity vx = %.1f m/s  (net Δx = %.1f nm over %.1f ns)\n",
            v, (xw[end] - xw[1]) * 1e9, duration * 1e9)

    # Trajectory figure (unwrapped, so the re-entry is a straight line).
    plt = plot(t .* 1e9, xw .* 1e9; xlabel = "t (ns)", ylabel = "skyrmion x, unwrapped (nm)",
               title = @sprintf("Skyrmion drift on an infinite wire: vx ≈ %.0f m/s", v),
               titlefontsize = 10, legend = false, lw = 2)
    out = joinpath(outdir, "skyrmion_drive.png")
    savefig(plt, out)
    println("Wrote figure → ", out)
end

main()
