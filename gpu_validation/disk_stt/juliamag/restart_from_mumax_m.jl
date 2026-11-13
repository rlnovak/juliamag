# Restart JuliaMag from a mumax3 snapshot (stt_es4_noZL_snap.mx3) and log the
# maximum torque in the core region (r < 20 nm) and in the vortex body (r ≥ 20 nm).
# Tells whether JuliaMag's own dynamics sustain the body torque that mumax3 shows
# after ~13 ns, or let it decay.
#
# Run:  julia -t auto --project=<repo> restart_from_mumax_m.jl OUTDIR IDX DURATION [LOG]

using JuliaMag, Printf

const DIR = ARGS[1]
const IDX = parse(Int, ARGS[2])
const DUR = parse(Float64, ARGS[3])
const LOG = length(ARGS) >= 4 ? ARGS[4] : "restart_torque.txt"

D, THICK, CELL = 128e-9, 16e-9, 2e-9
mesh = Mesh((64, 64, 8), (CELL, CELL, CELL))
mat  = Material(Msat = 800e3, Aex = 10.8e-12, alpha = 0.01,
                pol = 0.37, lambda = 1.0, epsilonPrime = 0.01)
rp   = RegionParams(mesh, mat)
setregion!(rp, 0; Msat = 0.0)
setgeometry!(rp, Cylinder(D, 2 * THICK); id = 1, edgesmooth = 4)
world = World(mesh, rp; demag = true)

m = Float64.(loadovf(joinpath(DIR, @sprintf("m%06d.ovf", IDX)))[1])
clearempty!(m, rp)
JC    = 0.040 / (π * D^2 / 4)
FIXED = (cos(20π/180), sin(20π/180), 0.0)

function rhs!(dm, mm)
    B = world._Bbuf
    effectivefield!(B, mm, world)
    torque!(dm, mm, B, mat.alpha)
    slonczewskitorque!(dm, mm, mesh, mat, JC, FIXED, THICK; add = true)
    return dm
end
k1 = similar(m); k2 = similar(m); k3 = similar(m); k4 = similar(m); tmp = similar(m)
function rk4!(mm, dt, n)
    for _ in 1:n
        rhs!(k1, mm); @. tmp = mm + 0.5dt*k1; normalize!(tmp); rhs!(k2, tmp)
        @. tmp = mm + 0.5dt*k2; normalize!(tmp); rhs!(k3, tmp)
        @. tmp = mm + dt*k3;    normalize!(tmp); rhs!(k4, tmp)
        @. mm = mm + (dt/6)*(k1 + 2k2 + 2k3 + k4); normalize!(mm)
    end
    return mm
end

R = [hypot((i - 32.5) * CELL, (j - 32.5) * CELL) for i in 1:64, j in 1:64, k in 1:8]
inside = rp.regions.id .== 1
core = inside .& (R .< 20e-9); body = inside .& (R .>= 20e-9)
dm = similar(m)
function tq()
    rhs!(dm, m)
    n = dropdims(sqrt.(sum(abs2, dm; dims = 1)); dims = 1) ./ γLL
    return maximum(n .* core), maximum(n .* body), n[24, 48, 4]
end

dt, every = 5e-14, 1e-12
io = open(LOG, "w")
println(io, "# t_since_restart (s)\tmaxTorque_core r<20nm (T)\tmaxTorque_body r>=20nm (T)\t|τ| cell (24,48,4) (T)\tmx\tmy\tmz")
function row(t)
    c, b, s = tq(); a = average_magnet(m, rp)
    @printf(io, "%.6e\t%.6e\t%.6e\t%.6e\t%.7f\t%.7f\t%.7f\n", t, c, b, s, a...)
    flush(io)
    return c, b
end
let (c, b) = row(0.0)
    @printf("t=0: core %.4e  body %.4e\n", c, b); flush(stdout)
end
t0 = time()
for n in 1:round(Int, DUR / every)
    rk4!(m, dt, round(Int, every / dt))
    c, b = row(n * every)
    n % 10 == 0 && (@printf("t=%.0f ps: core %.4e  body %.4e  (%.0f s)\n", n, c, b, time() - t0); flush(stdout))
end
close(io)
