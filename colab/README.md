# JuliaMag GPU benchmark on Google Colab

`juliamag_gpu_benchmark.ipynb` runs the JuliaMag CUDA extension on a Colab GPU:
installs Julia and the package, verifies GPU correctness against the CPU
(fields, demag, DMI, spin-transfer torques, and the solvers), runs µMAG Standard
Problem 4 entirely on the GPU, and benchmarks the full effective field and a
relax step CPU vs GPU across mesh sizes — then plots the speedup and lets you
download the results.

## Run it

1. Open the notebook in Colab:
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlnovak/juliamag/blob/main/colab/juliamag_gpu_benchmark.ipynb)
2. **Runtime → Change runtime type**: set **Runtime type = Python 3** and
   **Hardware accelerator = GPU** (T4/L4/A100 if available). The notebook is a
   Python notebook that shells out to Julia — on a Julia kernel the very first
   cell (`!nvidia-smi`) fails with `UndefVarError: nvidia not defined`.
3. **Runtime → Run all.** The first two install cells take a few minutes
   (Julia download + CUDA.jl artifacts + precompilation).

### Plotting note

The figures are drawn with **CairoMakie**, not Plots.jl. Colab's NVIDIA driver
ships a `libglapi` without `_glapi_tls_Current`, which breaks the
`Libglvnd_jll → GLFW_jll → GR_jll` chain that Plots.jl loads unconditionally
(`Plots.load_default_backend()` hardcodes `:gr`, so no preference or environment
variable avoids it). CairoMakie is pure Cairo and touches no OpenGL.

`examples/makie_shim.jl` provides the small Plots-compatible API the drivers use
(`plot`, `plot!`, `scatter!`, `hline!`, `vline!`, `savefig`), so the driver
scripts read the same as before. It keeps CairoMakie inside a module because
CairoMakie exports `Mesh`, which would otherwise collide with JuliaMag's `Mesh`.

## What to expect

The physics is validated to floating-point precision (~1e-15) regardless of the
GPU — that is already established on a Quadro P1000 (see `gpu_validation/` in the
repo). The reason to run here is **speed**: the effective field with the FFT
demag reaches ~3× on the modest P1000, so a current datacenter GPU should do
considerably better. The isolated derivative kernels (exchange, DMI, STT) are
memory-bound and benefit less; the win comes from the demag FFT in the full loop.

Outputs: `colab_benchmark.csv`, `colab_speedup.png`, and the Standard Problem 4
GPU table and figure.
