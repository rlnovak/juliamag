# GPU verification script — run on a machine with a CUDA GPU (e.g. Google Colab).
#
# Loads JuliaMag with the CUDA extension and checks that each GPU field/torque
# method agrees with the CPU version to floating-point tolerance, then times a
# demag-free effective-field evaluation on both. Because the whole package
# dispatches on the array type, the ONLY difference between the two paths is
# whether the magnetization lives in an Array or a CuArray.
#
# Setup on Colab (Julia + GPU runtime):
#   using Pkg; Pkg.add(url="https://github.com/rlnovak/juliamag"); Pkg.add("CUDA")
# then:  julia --project=. examples/gpu_check.jl

using JuliaMag
using CUDA
using Printf
using LinearAlgebra: norm

if !CUDA.functional()
    error("No functional CUDA GPU found. Run this on a GPU machine (e.g. Colab GPU runtime).")
end
println("CUDA device: ", CUDA.name(CUDA.device()))

mesh = Mesh((64, 64, 4), (4e-9, 4e-9, 4e-9))
mat  = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02, Ku = 5e5, anisU = (0, 0, 1))

# Random state on CPU and its GPU copy.
m_cpu = randommag!(zeromag(mesh))
m_gpu = togpu(m_cpu)

maxreldiff(a, cpu) = maximum(abs.(tocpu(a) .- cpu)) / (maximum(abs, cpu) + eps())

println("\nField-by-field CPU vs GPU agreement:")

# Exchange
Bc = similar(m_cpu); exchange!(Bc, m_cpu, mesh, mat)
Bg = similar(m_gpu); exchange!(Bg, m_gpu, mesh, mat)
@printf("  exchange:   max rel diff = %.2e\n", maxreldiff(Bg, Bc))

# Anisotropy
fill!(Bc, 0); anisotropy!(Bc, m_cpu, mesh, mat)
fill!(Bg, 0); anisotropy!(Bg, m_gpu, mesh, mat)
@printf("  anisotropy: max rel diff = %.2e\n", maxreldiff(Bg, Bc))

# Zeeman
Bext = (0.05, -0.02, 0.0)
zeeman!(Bc, Bext); zeeman!(Bg, Bext)
@printf("  zeeman:     max rel diff = %.2e\n", maxreldiff(Bg, Bc))

# Torque
dmc = similar(m_cpu); torque!(dmc, m_cpu, Bc, mat.alpha)
dmg = similar(m_gpu); torque!(dmg, m_gpu, Bg, mat.alpha)
@printf("  torque:     max rel diff = %.2e\n", maxreldiff(dmg, dmc))

# Average and normalize
ac = average(m_cpu); ag = average(m_gpu)
@printf("  average:    Δ = %.2e\n", maximum(abs.(collect(ac) .- collect(ag))))
mc = copy(m_cpu); normalize!(mc)
mg = copy(m_gpu); normalize!(mg)
@printf("  normalize:  max rel diff = %.2e\n", maxreldiff(mg, mc))

# Timing (exchange only, the per-step hot loop).
println("\nTiming (exchange!, 100 calls):")
exchange!(Bc, m_cpu, mesh, mat); tc = @elapsed for _ in 1:100; exchange!(Bc, m_cpu, mesh, mat); end
exchange!(Bg, m_gpu, mesh, mat); CUDA.@sync exchange!(Bg, m_gpu, mesh, mat)
tg = @elapsed (for _ in 1:100; exchange!(Bg, m_gpu, mesh, mat); end; CUDA.synchronize())
@printf("  CPU: %.1f ms   GPU: %.1f ms   speedup: %.1fx\n", tc*10, tg*10, tc/tg)

println("\nGPU check done.")
