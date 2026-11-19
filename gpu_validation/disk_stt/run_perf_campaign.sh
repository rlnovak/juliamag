#!/bin/bash
# Performance-rewrite validation campaign (run through run_when_gpu_idle.sh, which
# picks an idle GPU and sets CUDA_VISIBLE_DEVICES). Stops at the first failure and
# then shows the tail of the failing step's log.
#   1. GPU vs CPU agreement suite (Float64 + Float32)
#   2. cost breakdown of the disc right-hand side (bench_rhs.jl)
#   3. Float64 regression: 1 ns of the 40 ns disc driver, vs the pre-rewrite table
#   4. Float32: the full 40 ns disc series, vs mumax3 (float32)
set -e
JL="$HOME/opt/julia-1.11.9/bin/julia --project=$HOME/juliamag"
R=$HOME/juliamag
OUT=$HOME/JuliaMag/bench/perf
DRV="$R/gpu_validation/disk_stt/juliamag/disk_d1=128nm_t=16nm_40mA.jl"
mkdir -p "$OUT"; cd "$OUT"
echo "   host $(hostname), GPU $CUDA_VISIBLE_DEVICES"
step() {   # step LOG command... : run, show the log tail on failure
    local log=$1; shift
    "$@" > "$log" 2>&1 || { echo "FAILED (rc=$?), tail of $log:"; tail -20 "$log"; exit 1; }
}
echo "== $(date '+%F %T') 1. test_gpu";  step test_gpu2.log $JL $R/gpu_validation/test_gpu.jl
tail -1 test_gpu2.log
echo "== $(date '+%F %T') 2. bench_rhs"; step bench_rhs_new.log $JL $R/gpu_validation/disk_stt/juliamag/bench_rhs.jl
grep -E "===|RHS total|RK4 step|predicted|allocations" bench_rhs_new.log || true
echo "== $(date '+%F %T') 3. Float64 1 ns"
step jm64_1ns.log $JL "$DRV" 1e-9 jm64_1ns.txt Float64
tail -2 jm64_1ns.log
echo "== $(date '+%F %T') 4. Float32 40 ns"
step jm32_40ns.log $JL "$DRV" 40e-9 jm32_40ns.txt Float32
tail -2 jm32_40ns.log
echo "== $(date '+%F %T') campaign done"
