#!/bin/bash
cd /home/rlnovak/JuliaMag/stepper_bench || exit 1
st() { echo "$(date "+%F %T") step: $*"; "$@"; rc=$?; echo "$(date "+%F %T") step rc=$rc"; }
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_6c53fbb /home/rlnovak/juliamag_6c53fbb/gpu_validation/test_gpu.jl
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_6c53fbb /home/rlnovak/juliamag_6c53fbb/gpu_validation/disk_stt/juliamag/bench_stepper.jl Float32
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag3 /home/rlnovak/juliamag_6c53fbb/gpu_validation/disk_stt/juliamag/bench_stepper.jl Float32
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_6c53fbb /home/rlnovak/juliamag_6c53fbb/gpu_validation/disk_stt/juliamag/bench_stepper.jl Float32
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag3 /home/rlnovak/juliamag_6c53fbb/gpu_validation/disk_stt/juliamag/bench_stepper.jl Float32
echo "$(date "+%F %T") done"
