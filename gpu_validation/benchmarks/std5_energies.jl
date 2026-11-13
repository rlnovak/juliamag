# muMAG Standard Problem 5 — JuliaMag with time series + energies, for the mumax3
# cross-check. Same setup as examples/stdproblem5.jl (vortex + Zhang-Li STT),
# logging t, mx,my,mz and the four energies every 2.5 ps over 1 ns.

using JuliaMag
using Printf

const OUT = raw"C:\Users\rlnov\AppData\Local\Temp\claude\c--Users-rlnov-Projetos-mumag\bfbbd7a3-2b9f-4849-996b-9f44d5632691\scratchpad"

mesh = Mesh((32, 32, 4), (100e-9/32, 100e-9/32, 10e-9/4))
mat  = Material(Msat = 800e3, Aex = 13e-12, alpha = 0.1, pol = 1.0, xi = 0.05)
world = World(mesh, mat; demag = true)

m = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1))
mn = Minimizer(world, m; stopdm = 1e-6); minimize!(mn; maxsteps = 20_000); m = mn.m
@printf("relaxed ⟨m⟩ = (%.5f, %.5f, %.5f)\n", average(m)...); flush(stdout)

const J = (1e12, 0.0, 0.0)
function rhs!(dm, mm)
    B = world._Bbuf
    effectivefield!(B, mm, world)
    torque!(dm, mm, B, mat.alpha)
    zhanglitorque!(dm, mm, mesh, mat, J; add = true)
    return dm
end
k1=similar(m);k2=similar(m);k3=similar(m);k4=similar(m);tmp=similar(m)
function rk4!(mm, dt, n)
    for _ in 1:n
        rhs!(k1,mm); @. tmp=mm+0.5dt*k1; normalize!(tmp); rhs!(k2,tmp)
        @. tmp=mm+0.5dt*k2; normalize!(tmp); rhs!(k3,tmp)
        @. tmp=mm+dt*k3; normalize!(tmp); rhs!(k4,tmp)
        @. mm=mm+(dt/6)*(k1+2k2+2k3+k4); normalize!(mm)
    end
    return mm
end

dt=5e-14; save_dt=2.5e-12; total=1e-9
nchunk=round(Int,total/save_dt); per=round(Int,save_dt/dt)

tbl=open(joinpath(OUT,"std5_juliamag_table.txt"),"w")
println(tbl,"# JuliaMag std5 (post exchange fix) — t mx my mz E_demag E_exch E_Zeeman E_total")
function logrow!(t)
    a=average(m)
    Ed=demagenergy(m,world.demagplan,mesh,mat); Ex=exchangeenergy(m,mesh,mat)
    Ez=zeemanenergy(m,world.Bext,mesh,mat); Et=totalenergy(m,world)
    @printf(tbl,"%.6e\t%.6f\t%.6f\t%.6f\t%.6e\t%.6e\t%.6e\t%.6e\n",t,a[1],a[2],a[3],Ed,Ex,Ez,Et)
end
logrow!(0.0)
for c in 1:nchunk
    rk4!(m,dt,per); logrow!(c*save_dt)
end
close(tbl)
@printf("done. final ⟨m⟩ = (%.5f, %.5f, %.5f)\n", average(m)...)
