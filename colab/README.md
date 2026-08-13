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
2. **Runtime → Change runtime type → GPU** (T4/L4/A100 if available).
3. **Runtime → Run all.** The first two install cells take a few minutes
   (Julia download + CUDA.jl artifacts + precompilation).

## What to expect

The physics is validated to floating-point precision (~1e-15) regardless of the
GPU — that is already established on a Quadro P1000 (see `gpu_validation/` in the
repo). The reason to run here is **speed**: the effective field with the FFT
demag reaches ~3× on the modest P1000, so a current datacenter GPU should do
considerably better. The isolated derivative kernels (exchange, DMI, STT) are
memory-bound and benefit less; the win comes from the demag FFT in the full loop.

Outputs: `colab_benchmark.csv`, `colab_speedup.png`, and the Standard Problem 4
GPU table and figure.
