#!/bin/bash
cd '/home/rlnovak/JuliaMag/thermal_loop/phase2' || exit 1
st() { echo "$(date "+%F %T") step: $*"; "$@"; rc=$?; echo "$(date "+%F %T") step rc=$rc"; }
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 300 1 /home/rlnovak/JuliaMag/thermal_loop/phase2/dtcheck_T300_dt1e-13.txt 1e-9 1e-13 0.5 gpu Float32 31
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 300 1 /home/rlnovak/JuliaMag/thermal_loop/phase2/dtcheck_T300_dt5e-14.txt 1e-9 5e-14 0.5 gpu Float32 31
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 1 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T50_seed1.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 300 101 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T300_seed101.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 3 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T50_seed3.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 300 103 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T300_seed103.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 5 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T50_seed5.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 300 105 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T300_seed105.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 7 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T50_seed7.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 300 107 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T300_seed107.txt 2e-9 1e-13 0.5 gpu Float32 
echo "$(date "+%F %T") done"
