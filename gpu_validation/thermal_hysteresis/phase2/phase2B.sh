#!/bin/bash
cd '/home/rlnovak/JuliaMag/thermal_loop/phase2' || exit 1
st() { echo "$(date "+%F %T") step: $*"; "$@"; rc=$?; echo "$(date "+%F %T") step rc=$rc"; }
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 2 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T50_seed2.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 300 102 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T300_seed102.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 4 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T50_seed4.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 300 104 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T300_seed104.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 6 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T50_seed6.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 300 106 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T300_seed106.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 8 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T50_seed8.txt 2e-9 1e-13 0.5 gpu Float32 
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 300 108 /home/rlnovak/JuliaMag/thermal_loop/phase2/loop_T300_seed108.txt 2e-9 1e-13 0.5 gpu Float32 
echo "$(date "+%F %T") done"
