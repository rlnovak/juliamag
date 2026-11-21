#!/bin/bash
# Re-run of the validation problems after the edge-fill / demag-energy / tracker
# fixes and the GPU kernel rewrite (commit 36a619c). One invocation per host:
#
#   setsid nohup bash run_reval.sh labmac03 > ~/JuliaMag/reval/master_labmac03.log 2>&1 &
#   setsid nohup bash run_reval.sh labmac04 > ~/JuliaMag/reval/master_labmac04.log 2>&1 &
#
# Optional second argument SMOKE=1 runs every job for a few seconds of simulated
# time / under a wall-clock timeout instead (a check that each driver starts).
#
# Work tree: ~/JuliaMag/reval/{examples,bench,disk,logs}. The examples are run
# from a fresh copy of the repo's examples/ (the drivers write next to themselves),
# so the repo checkout is never modified. CPU jobs run detached in parallel, one
# process each; GPU chains wait for an idle GPU via run_when_gpu_idle.sh and run
# their steps in sequence on it. Every job logs to logs/<name>.log, ending with
# "done rc=N".
set -u
ROLE=${1:?usage: run_reval.sh labmac03|labmac04 [SMOKE]}
SMOKE=${2:-0}
JULIA=$HOME/opt/julia-1.11.9/bin/julia
# JM_REPO: the code tree to run (default ~/juliamag); JM_WORK: the work tree;
# CPU_ONLY=1 skips the GPU chains.
R=${JM_REPO:-$HOME/juliamag}
PROJ=$R
W=${JM_WORK:-$HOME/JuliaMag/reval}
CPU_ONLY=${CPU_ONLY:-0}
[ "$SMOKE" = 1 ] && W=$HOME/JuliaMag/reval_smoke
WAIT=$R/gpu_validation/run_when_gpu_idle.sh
GYRO=$R/gpu_validation/disk_stt/juliamag/gyro_lean.jl
DRV="$R/gpu_validation/disk_stt/juliamag/disk_d1=128nm_t=16nm_40mA.jl"
BEN=$R/gpu_validation/benchmarks

mkdir -p "$W/examples" "$W/bench" "$W/disk" "$W/logs"
export JM_OUT=$W/bench          # output directory of the std4/std5 energy drivers
# Only the drivers and the reference tables (OOMMF, mumax3): no old JuliaMag result
# table is copied, so a driver that fails leaves no stale table behind.
cp "$R"/examples/*.jl "$R"/examples/stdprob2_oommf.txt "$R"/examples/stdprob4a_oommf.odt    "$R"/examples/stdproblem4_mumax3.txt "$W/examples/"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # one core per job
cd "$W"
echo "$(date '+%F %T') reval $ROLE on $(hostname), SMOKE=$SMOKE, work tree $W"
echo "repo md5 of src+ext: $(cat "$R"/src/*.jl "$R"/ext/*.jl | md5sum | cut -d' ' -f1)"

# Simulated times (seconds) — full runs or smoke.
if [ "$SMOKE" = 1 ]; then T7=2e-11; T40=1e-11; CPU_TIMEOUT=240; else T7=7e-9; T40=40e-9; CPU_TIMEOUT=0; fi

# cpu NAME DIR script [args...]: run detached in DIR on the CPU, log to logs/NAME.log.
# In SMOKE mode the job is killed after CPU_TIMEOUT s (rc=124 is then the expected
# outcome: the driver started and ran without error until the timeout).
cpu() {
    local name=$1 dir=$2; shift 2
    local log="$W/logs/$name.log"
    (
        cd "$dir" || exit 1
        echo "$(date '+%F %T') start on $(hostname) (CPU): $*"
        if [ "$CPU_TIMEOUT" -gt 0 ]; then
            CUDA_VISIBLE_DEVICES= timeout "$CPU_TIMEOUT" "$JULIA" --project="$PROJ" "$@"
        else
            CUDA_VISIBLE_DEVICES= "$JULIA" --project="$PROJ" "$@"
        fi
        rc=$?                  # capture before $(date) resets $?
        echo "$(date '+%F %T') done rc=$rc"
    ) > "$log" 2>&1 < /dev/null &
    echo "  cpu job $name: pid $!"
}

# gpuchain NAME "step1" "step2" ...: wait for an idle GPU, then run the steps in
# sequence on it (each step a julia script + args, run in $W/disk); a failing step
# does not stop the chain. Log: logs/NAME.log.
gpuchain() {
    local name=$1; shift
    local script="$W/logs/$name.chain.sh"
    {
        echo "#!/bin/bash"
        echo "cd '$W/disk' || exit 1"
        echo "rc_all=0"
        for s in "$@"; do
            echo "echo \"\$(date '+%F %T') step: $s\""
            echo "$JULIA --project=$PROJ $s; rc=\$?; echo \"\$(date '+%F %T') step rc=\$rc\"; [ \$rc -eq 0 ] || rc_all=\$rc"
        done
        echo "exit \$rc_all"
    } > "$script"
    setsid nohup bash "$WAIT" "$W/logs/$name.log" bash "$script" > /dev/null 2> "$W/logs/$name.waiter.err" < /dev/null &
    echo "  gpu chain $name: waiter pid $!"
}

case "$ROLE" in
labmac03)
    E=$W/examples
    cpu std2        "$E" "$E/stdproblem2.jl"
    cpu std2hyst    "$E" "$E/stdproblem2_hysteresis.jl"
    cpu std4        "$E" "$E/stdproblem4.jl"
    cpu std5        "$E" "$E/stdproblem5.jl"
    cpu std4en "$W/bench" "$BEN/std4_energies.jl"
    cpu std5en "$W/bench" "$BEN/std5_energies.jl"
    cpu stage2      "$E" "$E/stage2_local.jl"
    cpu thermal     "$E" "$E/thermal_demagnetization.jl"
    cpu dischyst    "$E" "$E/disc_hysteresis.jl"
    cpu skyrmion    "$E" "$E/skyrmion_drive.jl"
    # GPU (labmac04's GPUs are taken by another project's long queue, so both
    # chains run here, ~2.3 h and ~4.4 h):
    #   A: the time-step convergence partner (EdgeSmooth 4, Slonczewski only, 7 ns,
    #      dt = 5e-14), standard problem 4 on the GPU, then the three like-for-like
    #      7 ns runs against mumax3 (EdgeSmooth 0).
    #   B: the 40 ns EdgeSmooth-4 series in Float64, then the dt = 2.5e-14 run.
    [ "$CPU_ONLY" = 1 ] || gpuchain gpuA "$GYRO 4 5e-14 $T7 stt stt_es4_dt5.txt" \
                  "$E/stdproblem4_gpu.jl" \
                  "$GYRO 0 5e-14 $T7 stt jm_stt_es0_dt5.txt" \
                  "$GYRO 0 5e-14 $T7 sttzl sttzl_es0_dt5.txt" \
                  "$GYRO 0 5e-14 $T7 free free_es0_dt5.txt"
    [ "$CPU_ONLY" = 1 ] || gpuchain gpuB "'$DRV' $T40 $W/disk/jm_es4_40ns.txt Float64" \
                  "$GYRO 4 2.5e-14 $T7 stt stt_es4_dt25.txt"
    ;;
labmac04)
    # CPU: static fields on mumax3's relaxed m (needs the mumax3 OVF outputs, which
    # live on this host).
    cpu fieldcmp "$W/disk" "$R/gpu_validation/disk_stt/juliamag/fieldcmp.jl" "$HOME/JuliaMag/bench/fields"
    ;;
*) echo "unknown role $ROLE" >&2; exit 2 ;;
esac
echo "$(date '+%F %T') all jobs launched"
