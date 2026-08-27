# Stage-2 test problems, adapted from magnum.np demos (see docs/stage2_problems.md).
# Runs the problems JuliaMag can do today, saving a data table and a figure for
# each. Runs on CPU here; the same physics runs on the GPU via togpu (see the
# Colab notebook's Stage-2 section).
#
# Run:  julia --project=examples examples/stage2_local.jl

using JuliaMag
using Printf
using Statistics
include(joinpath(@__DIR__, "makie_shim.jl"))
gr()

const OUT = @__DIR__

# --- Problem 1: Langevin — free macrospin at finite T (magnum.np demos/langevin) --
# An isotropic macrospin (no anisotropy) in a field along x. In equilibrium the
# time-averaged ⟨mx⟩ follows the classical Langevin function
#   L(ξ) = coth(ξ) − 1/ξ,   ξ = μ0 Ms V H / (kB T),
# the same benchmark as Leliaert et al. (magnum.np's demo). We fix a set of ξ
# (as the demo does) and check ⟨mx⟩ ≈ L(ξ). No anisotropy, so this isolates the
# thermal field's fluctuation-dissipation statistics.
function langevin_demo()
    println("\n[1/3] Langevin: free macrospin at finite T…"); flush(stdout)
    μ0 = JuliaMag.μ0; kB = JuliaMag.kB
    # An ensemble of UNCOUPLED macrospins (Aex = 0), exactly as magnum.np's demo
    # (64^3 uncoupled cells): the spatial average over the ensemble at one instant
    # equals the thermodynamic ⟨mx⟩ = L(ξ), so no long time-averaging is needed.
    n = (16, 16, 16)                                        # 4096 spins
    mesh = Mesh(n, (10e-9, 10e-9, 10e-9))
    Ms = 1e6
    mat = Material(Msat = Ms, Aex = 0.0, alpha = 0.5)       # no exchange, no Ku
    V  = JuliaMag.cellvolume(mesh)
    langevin(ξ) = coth(ξ) - 1/ξ
    ξs = (1.0, 2.0, 5.0, 10.0, 20.0)                        # μ0 Ms V H/(kB T)
    T  = 300.0
    tbl = open(joinpath(OUT, "stage2_langevin.txt"), "w")
    println(tbl, "# Langevin: ensemble of uncoupled macrospins, <mx> vs L(ξ). T=$(T) K")
    println(tbl, "# xi\tH[A/m]\tmx_sim\tL(xi)")
    xs = Float64[]; sim_mx = Float64[]
    for ξ in ξs
        H = ξ * kB * T / (μ0 * Ms * V)                      # field giving this ξ
        sim = Simulation(mesh, mat; demag = false)
        setmag!(sim, RandomConfig())                        # start disordered → equilibrates from both sides
        JuliaMag.setexternalfield!(sim.world, (μ0 * H, 0.0, 0.0))   # B = μ0 H [T]
        # Relax the ensemble under the thermal field, then average over spins and a
        # few snapshots (ensemble average dominates; time-averaging just de-noises).
        runthermal!(sim, 2e-9, T; dt = 1e-15)               # long thermalization
        acc = Float64[]
        for _ in 1:40
            runthermal!(sim, 40 * 1e-15, T; dt = 1e-15)
            push!(acc, average(sim.m)[1])                   # spatial mean over 4096 spins
        end
        mx = mean(acc); L = langevin(ξ)
        push!(xs, ξ); push!(sim_mx, mx)
        @printf(tbl, "%.2f\t%.4e\t%.4f\t%.4f\n", ξ, H, mx, L)
        @printf("  ξ=%.1f  ⟨mx⟩=%.3f  L(ξ)=%.3f  (Δ=%.3f)\n", ξ, mx, L, mx-L)
    end
    close(tbl)
    ξfine = range(0.5, 22; length = 100)
    plt = plot(ξfine, langevin.(ξfine); lw = 2, label = "Langevin L(ξ)",
               xlabel = "ξ = μ₀MsVH/kBT", ylabel = "⟨mx⟩", title = "Langevin: free macrospin at 300 K",
               titlefontsize = 10, legend = :bottomright)
    scatter!(plt, xs, sim_mx; ms = 5, label = "JuliaMag (thermal)")
    savefig(plt, joinpath(OUT, "stage2_langevin.png"))
    println("  wrote stage2_langevin.txt / .png")
end

# --- Problem 2: domain-wall depinning (magnum.np demos/sp_domainwall_pinning) -
# A soft region (x<0) and a hard region (x≥0). A wall sits at the interface; ramp
# a field and find the depinning field where the soft region reverses.
function depinning_demo()
    println("\n[2/3] Domain-wall depinning at a soft/hard interface…"); flush(stdout)
    N = 80
    mesh = Mesh((N, 1, 1), (1e-9, 1e-9, 1e-9))
    μ0 = JuliaMag.μ0
    # magnum.np sp_domainwall_pinning: easy axis along y; soft (x<0) vs hard (x≥0),
    # 4× contrast in Ms/Ku/A; field ramped along +y from 1.4/μ0 to 1.8/μ0.
    soft = Material(Msat = 0.25/μ0, Aex = 0.25e-11, alpha = 1.0, Ku = 1e5, anisU = (0, 1, 0))
    rp = RegionParams(mesh, soft)                      # region 0 = soft
    setregion!(rp, 1; Msat = 1.0/μ0, Aex = 1.0e-11, Ku = 1e6, anisU = (0, 1, 0))
    defregion!(rp, 1, XRange(0.0, 1e3))                # x ≥ 0 → hard
    # Initial state: soft canted near +y (sin.3, cos.3, 0), hard along −y — a wall
    # pinned at the material step.
    m0 = setconfig(mesh, (x,y,z) -> x < 0 ? (sin(0.3), cos(0.3), 0.0) : (0.0, -1.0, 0.0))
    sim = Simulation(mesh, rp; demag = false); setmag!(sim, m0)

    tbl = open(joinpath(OUT, "stage2_depinning.txt"), "w")
    println(tbl, "# Domain-wall depinning — <my> vs applied field along y (soft/hard interface)")
    println(tbl, "# B[T]\tmy_total\tmy_soft\tmy_hard")
    Bs = range(1.40, 1.80; length = 81)                # tesla along +y (as B = μ0 H)
    mytot = Float64[]
    for B in Bs
        JuliaMag.setexternalfield!(sim.world, (0.0, B, 0.0))
        relax!(sim; stopdm = 1e-6)
        mt = average(sim.m)[2]
        ms = average_region(sim.m, rp, 0)[2]; mh = average_region(sim.m, rp, 1)[2]
        push!(mytot, mt)
        @printf(tbl, "%.4f\t%.5f\t%.5f\t%.5f\n", B, mt, ms, mh)
    end
    close(tbl)
    # Depinning field: the largest jump in ⟨my⟩ as the wall unpins and sweeps.
    d = diff(mytot); ji = argmax(d); Hdep = collect(Bs)[ji]
    @printf("  depinning field ≈ %.3f T (largest ⟨my⟩ jump)\n", Hdep)
    plt = plot(collect(Bs), mytot; lw = 2, marker = :circle, ms = 2, legend = false,
               xlabel = "μ₀Hy (T)", ylabel = "⟨my⟩", title = "Domain-wall depinning (soft/hard)",
               titlefontsize = 10)
    vline!(plt, [Hdep]; ls = :dash, color = :gray)
    savefig(plt, joinpath(OUT, "stage2_depinning.png"))
    println("  wrote stage2_depinning.txt / .png")
end

# --- Problem 3: bulk-DMI spin spiral (magnum.np demos/dmi, bulk case) ---------
# A bulk-DMI chain with no anisotropy relaxes to a helix of period L = 4πA/D. We
# seed a helix at that period on a periodic chain and relax; the period is
# preserved. This validates the bulk DMI without the chirality-dependent DMI
# boundary condition (a listed future feature) that the interfacial-DMI
# edge-canting demo (sp_DMI) needs.
function dmi_spiral_demo()
    println("\n[3/3] Bulk-DMI spin spiral (period L = 4piA/D)..."); flush(stdout)
    A = 13e-12; D = 3e-3
    L = 4π * A / D                                  # equilibrium helix period [m]
    cx = 2e-9; N = max(40, round(Int, 2L / cx))     # ~2 periods, periodic along x
    mesh = Mesh((N, 1, 1), (cx, cx, cx); pbc = (1, 0, 0))
    mat  = Material(Msat = 8e5, Aex = A, alpha = 0.5, Dbulk = D)
    k = 2π / L
    m0 = setconfig(mesh, (x, y, z) -> (cos(k*x), 0.0, sin(k*x)))   # Bloch helix
    sim = Simulation(mesh, mat; demag = false); setmag!(sim, m0)
    relax!(sim; stopdm = 1e-7)
    # The helix relaxes into whichever plane the DMI chirality selects; measure the
    # period from the total winding of the in-plane phase along x (unwrapped),
    # using the two components with the largest variance.
    comps = [[sim.m[c,i,1,1] for i in 1:N] for c in 1:3]
    vars = [sum(abs2, c .- sum(c)/N) for c in comps]
    a, b = sortperm(vars; rev=true)[1:2]                # the two winding components
    θ = atan.(comps[b], comps[a])
    total = 0.0
    for i in 1:N-1
        d = θ[i+1]-θ[i]; d > π && (d -= 2π); d < -π && (d += 2π); total += d
    end
    turns = abs(total) / (2π)
    Lmeas = turns > 0 ? (N*cx) / turns : NaN
    @printf("  seeded L=%.1f nm, measured L=%.1f nm (%.2f turns over %.0f nm)\n",
            L*1e9, Lmeas*1e9, turns, N*cx*1e9)
    tbl = open(joinpath(OUT, "stage2_dmi_spiral.txt"), "w")
    println(tbl, "# Bulk-DMI spin spiral — relaxed m(x). seeded L=$(L) m, measured L=$(Lmeas) m")
    println(tbl, "# x[nm]\tmx\tmy\tmz")
    xs = Float64[]; mxs = Float64[]; mzs = Float64[]
    for i in 1:N
        x = (i - (N+1)/2) * cx * 1e9
        push!(xs, x); push!(mxs, sim.m[1,i,1,1]); push!(mzs, sim.m[3,i,1,1])
        @printf(tbl, "%.1f\t%.5f\t%.5f\t%.5f\n", x, sim.m[1,i,1,1], sim.m[2,i,1,1], sim.m[3,i,1,1])
    end
    close(tbl)
    plt = plot(xs, [mxs mzs]; lw = 2, label = ["mx" "mz"], xlabel = "x (nm)", ylabel = "m",
               title = @sprintf("Bulk-DMI helix: L=%.0f nm", Lmeas*1e9), titlefontsize = 10)
    savefig(plt, joinpath(OUT, "stage2_dmi_spiral.png"))
    println("  wrote stage2_dmi_spiral.txt / .png")
end

function main()
    println("JuliaMag Stage-2 problems (adapted from magnum.np demos)")
    langevin_demo()
    depinning_demo()
    dmi_spiral_demo()
    println("\nStage-2 done. Tables + figures in examples/.")
end

main()
