# GPU demag verification + benchmark — run on a CUDA machine.
#
# Checks that the GPU demagnetization field and the full GPU effective field
# (exchange + anisotropy + demag + Zeeman, assembled through `togpu(world)`)
# agree with the CPU to floating-point precision, then times the full effective
# field on both across a range of mesh sizes. The demag FFT convolution runs on
# the GPU via CUFFT; because the whole package dispatches on the array type, the
# only difference between the paths is Array vs CuArray.
#
# Setup:  julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.add("CUDA")'
# Run:    julia --project=. examples/gpu_demag_check.jl

using JuliaMag
using CUDA
using Printf

if !CUDA.functional()
    error("No functional CUDA GPU found.")
end
println("CUDA device: ", CUDA.name(CUDA.device()))

maxreldiff(g, c) = maximum(abs.(Array(g) .- c)) / (maximum(abs, c) + eps())

# --- Accuracy: demag field and full effective field, CPU vs GPU --------------
println("\nAccuracy (max rel diff CPU vs GPU):")
for (nm, (nx, ny, nz), cs) in [
        ("std4",   (100, 25, 1), (5e-9, 5e-9, 3e-9)),
        ("cube3D", (32, 32, 16), (4e-9, 4e-9, 4e-9)),
        ("film",   (128, 128, 1), (4e-9, 4e-9, 4e-9))]
    mesh = Mesh((nx, ny, nz), cs)
    mat  = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02, Ku = 5e4, anisU = (0, 0, 1))
    m    = randommag!(zeromag(mesh))

    # demag only
    ker  = JuliaMag.demagkernel(Float64, mesh)
    plan = JuliaMag.DemagPlan(ker, mesh, mat.Msat)
    Bc = similar(m); demagfield!(Bc, m, plan)
    Bg = similar(togpu(m)); demagfield!(Bg, togpu(m), togpu(plan))
    d_demag = maxreldiff(Bg, Bc)

    # full effective field
    wc = World(mesh, mat; demag = true, Bext = (0.02, -0.01, 0.0))
    fill!(Bc, 0); effectivefield!(Bc, m, wc)
    wg = togpu(wc); mg = togpu(m); Bg2 = similar(mg)
    effectivefield!(Bg2, mg, wg)
    d_eff = maxreldiff(Bg2, Bc)

    @printf("  %-7s cells=%6d   demag=%.2e   effective field=%.2e\n",
            nm, nx*ny*nz, d_demag, d_eff)
end

# --- Performance: full effective field with demag, CPU vs GPU ----------------
println("\nTiming — full effective field with demag (ms/call):")
println("  cells     CPU(ms)   GPU(ms)   speedup")
mat = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02, Ku = 5e4, anisU = (0, 0, 1))
for (nx, ny, nz) in [(64,64,1), (128,128,1), (256,256,1), (512,512,1), (256,256,8), (1024,1024,1)]
    mesh = Mesh((nx, ny, nz), (4e-9, 4e-9, 4e-9))
    wc = World(mesh, mat; demag = true, Bext = (0.02, -0.01, 0.0))
    m  = randommag!(zeromag(mesh)); Bc = similar(m)
    effectivefield!(Bc, m, wc); tc = @elapsed for _ in 1:30; effectivefield!(Bc, m, wc); end
    wg = togpu(wc); mg = togpu(m); Bg = similar(mg)
    CUDA.@sync effectivefield!(Bg, mg, wg)
    tg = @elapsed (for _ in 1:30; effectivefield!(Bg, mg, wg); end; CUDA.synchronize())
    @printf("  %8d  %7.2f   %7.2f   %.2fx\n", nx*ny*nz, tc/30*1000, tg/30*1000, tc/tg)
end

println("\nGPU demag check done.")
