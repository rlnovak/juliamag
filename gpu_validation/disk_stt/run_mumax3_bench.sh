#!/bin/bash
# mumax3 timing reference on the same machine as the JuliaMag campaign (run through
# run_when_gpu_idle.sh, which sets CUDA_VISIBLE_DEVICES; mumax3 then sees one GPU, 0).
#   1. fixed-step RK4 50 fs, 0.5 ns from the 17 ns snapshot (JuliaMag's algorithm)
#   2. the 40 ns production run with NEval (mumax3's own adaptive RK45)
# Needs $DIR/stt_es4_noZL_snap.out/m000001.ovf (from stt_es4_noZL_snap.mx3).
# Private kernel cache: a cache shared with another user may be unwritable, and the
# kernel computation would then land in the measured wall time.
set -e
MUMAX=$HOME/mumax3/mumax3.11.1_linux_cuda12.0/mumax3
DIR=$HOME/JuliaMag/bench/disk40
SRC=$HOME/juliamag/gpu_validation/disk_stt/mumax3
CACHE=/tmp/mumax3cache_$USER
mkdir -p "$CACHE"
cd "$DIR"
echo "   host $(hostname), GPU $CUDA_VISIBLE_DEVICES"
[ -f stt_es4_noZL_snap.out/m000001.ovf ] || { echo "missing snapshot"; exit 2; }
for f in bench_rk4fixed_0p5ns bench_40ns_neval; do
    cp "$SRC/$f.mx3" .
    echo "== $(date '+%F %T') mumax3 $f"
    t0=$(date +%s%N)
    "$MUMAX" -gpu 0 -http "" -cache "$CACHE" "$f.mx3" > "$f.log" 2>&1 || { tail -20 "$f.log"; exit 1; }
    t1=$(date +%s%N)
    printf "   wall %d.%03d s; %s\n" $(( (t1 - t0) / 1000000000 )) $(( (t1 - t0) / 1000000 % 1000 )) \
           "$(grep -E "steps" "$f.log" | tail -1 || true)"
done
echo "== $(date '+%F %T') mumax3 bench done"
