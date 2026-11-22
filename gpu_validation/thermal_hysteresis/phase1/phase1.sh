#!/bin/bash
# Phase 1 (commit 309980f): dwell / dt convergence of the finite-T vortex loop,
# virgin branch 0 -> 150 mT (31 field steps), T = 50 K, seed 1, GPU Float32.
cd /home/rlnovak/JuliaMag/thermal_loop || exit 1
st() { echo "$(date "+%F %T") step: $*"; "$@"; rc=$?; echo "$(date "+%F %T") step rc=$rc"; }
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 1 /home/rlnovak/JuliaMag/thermal_loop/conv_dwell0.5ns.txt 0.5e-9 1e-13 0.5 gpu Float32 31
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 1 /home/rlnovak/JuliaMag/thermal_loop/conv_dwell1ns.txt   1e-9   1e-13 0.5 gpu Float32 31
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 1 /home/rlnovak/JuliaMag/thermal_loop/conv_dwell2ns.txt   2e-9   1e-13 0.5 gpu Float32 31
st /home/rlnovak/opt/julia-1.11.9/bin/julia --project=/home/rlnovak/juliamag_309980f /home/rlnovak/juliamag_309980f/examples/disc_hysteresis_thermal.jl 50 1 /home/rlnovak/JuliaMag/thermal_loop/conv_dwell1ns_dt5e-14.txt 1e-9 5e-14 0.5 gpu Float32 31
