#!/usr/bin/env bash
# 40 ns production series (EdgeSmooth 4, Slonczewski only): JuliaMag on one GPU,
# mumax3 on the other. labmac04's GPUs are shared with other jobs, so each job
# first waits until its GPU has had no compute process for 3 consecutive checks
# 60 s apart (a back-to-back queue elsewhere leaves only ~1 s gaps).
#
# Usage (from the run directory holding both scripts):
#   nohup ./run_queue_disk40.sh > run_queue_disk40.log 2>&1 < /dev/null &
# Writes jm_es4_40ns.{txt,log}, stt_es4_noZL_40ns.{out,log} and *.done markers.
set -u
J=~/opt/julia-1.11.9/bin/julia
P=--project=$HOME/juliamag
MX=~/mumax3/mumax3.11.1_linux_cuda12.0/mumax3
JM_GPU=${JM_GPU:-0}
MX_GPU=${MX_GPU:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID        # same numbering as nvidia-smi -i
cd "$(dirname "$(readlink -f "$0")")"
stamp() { date '+%F %T'; }

# Preflight, before any wait: inputs, executables, and a JuliaMag that has the
# functions the driver needs (the project must be synced with the repo).
for f in disk_d1=128nm_t=16nm_40mA.jl stt_es4_noZL_40ns.mx3 "$J" "$MX"; do
  [ -e "$f" ] || { echo "$(stamp) preflight: missing $f"; exit 1; }
done
$J $P -e 'using JuliaMag; @assert isdefined(JuliaMag, :average_magnet)' \
  || { echo "$(stamp) preflight: JuliaMag env at $P is out of date"; exit 1; }
nvidia-smi -L > /dev/null || { echo "$(stamp) preflight: nvidia-smi fails"; exit 1; }

# Busy = a compute process on the GPU. A failing nvidia-smi stops the job instead of
# reading as busy (wait forever) or idle (start on top of another run).
gpu_busy() {
  local o
  o=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) \
    || { echo "$(stamp) nvidia-smi failed on GPU $1" >&2; exit 2; }
  [ -n "$o" ]
}
wait_gpu() {
  local ok=0
  while [ $ok -lt 3 ]; do
    if gpu_busy "$1"; then ok=0; else ok=$((ok + 1)); fi
    [ $ok -lt 3 ] && sleep 60
  done
}

( wait_gpu "$JM_GPU"
  echo "$(stamp) start JuliaMag 40 ns on GPU $JM_GPU"
  CUDA_VISIBLE_DEVICES=$JM_GPU $J $P disk_d1=128nm_t=16nm_40mA.jl 40e-9 jm_es4_40ns.txt \
      > jm_es4_40ns.log 2>&1 < /dev/null
  rc=$?; echo "$(stamp) JuliaMag done rc=$rc"; echo $rc > jm.done ) &

( wait_gpu "$MX_GPU"
  echo "$(stamp) start mumax3 40 ns on GPU $MX_GPU"
  $MX -gpu "$MX_GPU" -http "" -cache /tmp/mumax3cache -f stt_es4_noZL_40ns.mx3 \
      > stt_es4_noZL_40ns.log 2>&1 < /dev/null
  rc=$?; echo "$(stamp) mumax3 done rc=$rc"; echo $rc > mumax.done ) &

wait
echo "$(stamp) queue finished"
