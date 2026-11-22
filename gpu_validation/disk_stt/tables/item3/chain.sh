#!/bin/bash
cd /home/rlnovak/JuliaMag/item3 || exit 1
st() { echo "$(date "+%F %T") step: $*"; "$@"; rc=$?; echo "$(date "+%F %T") step rc=$rc"; }
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag3 /home/rlnovak/juliamag3/gpu_validation/disk_stt/juliamag/disk_40mA_sim.jl /home/rlnovak/JuliaMag/item3/smoke Float32 rk45 1e-5 2e-11
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag3 /home/rlnovak/juliamag3/gpu_validation/test_gpu.jl
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag3 /home/rlnovak/juliamag3/gpu_validation/disk_stt/juliamag/disk_40mA_sim.jl /home/rlnovak/JuliaMag/item3/f32_rk45 Float32 rk45 1e-5 40e-9
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag3 /home/rlnovak/juliamag3/gpu_validation/disk_stt/juliamag/disk_40mA_sim.jl /home/rlnovak/JuliaMag/item3/f64_rk45 Float64 rk45 1e-5 40e-9
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag3 /home/rlnovak/juliamag3/gpu_validation/disk_stt/juliamag/disk_40mA_sim.jl /home/rlnovak/JuliaMag/item3/f32_rk4_1ns Float32 rk4 1e-5 1e-9
