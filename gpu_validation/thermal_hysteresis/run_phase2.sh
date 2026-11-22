#!/bin/bash
# Phase 2 of the finite-temperature vortex hysteresis (examples/disc_hysteresis_thermal.jl):
# full loops 0 -> +150 -> -150 -> +150 mT, dwell 2 ns per 5 mT step, Heun dt 1e-13 s,
# alpha 0.5, GPU Float32; 8 realizations at 50 K (seeds 1-8) and 8 at 300 K (seeds
# 101-108, distinct from the 50 K ones so that the two ensembles share no noise).
#
# Phase 1 (phase1/) fixed dwell = 2 ns and dt = 1e-13 s at 50 K. The dt check is
# repeated here at 300 K first (virgin branch, 31 steps, dwell 1 ns, dt 1e-13 vs
# 5e-14 s). It ends ~1 h into chain A, ~2 h before the first 300 K loop (chain B)
# starts; nothing stops the chains automatically: check it and kill them by hand if
# it fails (kill -- -<pgid of the waiter>). A failing step does not stop a chain
# either: read the "step rc=" lines of logs/phase2?.log.
#
# Two chains, one per GPU, each behind run_when_gpu_idle.sh; the loops alternate
# 50 K / 300 K so that a partial ensemble is balanced. About 3.1 h per loop.
# Code: commit 309980f, unpacked in ~/juliamag_309980f (md5-checked, see README).
#
# Usage (on labmac03): bash run_phase2.sh [SMOKE=1 env: 2 field steps per run]
set -u
[ "${SMOKE:-0}" = 0 ] && SMOKE=      # SMOKE=0 means no smoke
R=$HOME/juliamag_309980f
JL=$HOME/opt/julia-1.11.9/bin/julia
DRV=$R/examples/disc_hysteresis_thermal.jl
WAIT0=$R/gpu_validation/run_when_gpu_idle.sh
W=$HOME/JuliaMag/thermal_loop/phase2${SMOKE:+_smoke}
WAIT=$W/logs/run_when_gpu_idle.sh     # LF copy: a git archive made on Windows has CRLF
NS=${SMOKE:+2}                         # empty = whole sweep
[ -e "$W/logs/phase2A.sh" ] && { echo "$W already launched; move it away first" >&2; exit 1; }
for f in "$JL" "$DRV" "$WAIT0"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done
mkdir -p "$W/logs"
tr -d '\r' < "$WAIT0" > "$WAIT"
bash -n "$WAIT" || { echo "waiter copy does not parse" >&2; exit 1; }

# run T SEED OUT DWELL DT [NSTEPS]: one driver call, written as a chain line.
run() {
    local n=${6:-$NS}
    echo "st $JL --project=$R $DRV $1 $2 $W/$3 $4 $5 0.5 gpu Float32 $n"
}

chain() {
    local name=$1; shift
    local script=$W/logs/$name.sh
    {
        echo "#!/bin/bash"
        echo "cd '$W' || exit 1"
        echo 'st() { echo "$(date "+%F %T") step: $*"; "$@"; rc=$?; echo "$(date "+%F %T") step rc=$rc"; }'
        for line in "$@"; do echo "$line"; done
        echo 'echo "$(date "+%F %T") done"'
    } > "$script"
    setsid nohup bash "$WAIT" "$W/logs/$name.log" bash "$script" > /dev/null 2> "$W/logs/$name.waiter.err" < /dev/null &
    echo "chain $name: waiter pid $!, script $script"
}

NDT=${NS:-31}                          # virgin branch for the dt check
A=( "$(run 300 1 dtcheck_T300_dt1e-13.txt 1e-9 1e-13 "$NDT")"
    "$(run 300 1 dtcheck_T300_dt5e-14.txt 1e-9 5e-14 "$NDT")" )
B=()
for i in 1 2 3 4 5 6 7 8; do
    l50=$(run 50 $i loop_T50_seed$i.txt 2e-9 1e-13)
    l300=$(run 300 $((100 + i)) loop_T300_seed$((100 + i)).txt 2e-9 1e-13)
    if [ $((i % 2)) = 1 ]; then A+=("$l50" "$l300"); else B+=("$l50" "$l300"); fi
done
chain phase2A "${A[@]}"
chain phase2B "${B[@]}"
