#!/bin/bash
# Run a command on the first GPU that has been idle for IDLE_CHECKS consecutive
# checks 60 s apart (default 10, i.e. ~10 min) — so jobs queue behind other users'
# runs on a shared machine. The window must be longer than any pause inside the
# other queues (on labmac03 a follow-up queue polls every 300 s). The command runs
# with CUDA_VISIBLE_DEVICES set to that GPU's UUID (NVIDIA does not guarantee that
# nvidia-smi's index order matches CUDA's; the UUID is unambiguous).
#
# Several of these waiters can run at once: a GPU is claimed with flock on
# /tmp/run_when_gpu_idle.<uuid>.lock, so two of them never start on the same GPU.
# The kernel drops the lock when the holder dies (even on SIGKILL), and the job
# inherits the descriptor, so the claim lasts exactly as long as the job.
# While the job runs, every compute process on its GPU is logged each minute to
# LOG.gpuapps, so a foreign job landing on the same GPU (spoiling timings) shows up.
#
# Start it detached:  setsid nohup run_when_gpu_idle.sh LOG cmd args... &
# Stop it with:       kill -- -<pgid>   (setsid gives it its own process group)
# Usage: [IDLE_CHECKS=n] run_when_gpu_idle.sh LOG command args...
[ $# -ge 2 ] || { echo "usage: $0 LOG command [args...]" >&2; exit 2; }
LOG=$1; shift
NEED=${IDLE_CHECKS:-10}
[[ $NEED =~ ^[0-9]+$ ]] && [ "$NEED" -ge 1 ] || { echo "bad IDLE_CHECKS=$NEED" >&2; exit 2; }
mkdir -p "$(dirname "$LOG")"
SMI() { timeout 30 nvidia-smi "$@"; }

mapfile -t UUIDS < <(SMI --query-gpu=uuid --format=csv,noheader)
NGPU=${#UUIDS[@]}
[ "$NGPU" -ge 1 ] || { echo "no GPU visible to nvidia-smi" >&2; exit 2; }
for u in "${UUIDS[@]}"; do [[ $u == GPU-* ]] || { echo "bad GPU uuid '$u'" >&2; exit 2; }; done
lockfile() { echo "/tmp/run_when_gpu_idle.$1.lock"; }
echo "$(date '+%F %T') waiting for an idle GPU ($NEED checks): $*" >> "$LOG"

# Busy if another waiter holds its claim, a compute process is listed on it, or
# (in case other users' processes are hidden from this user) more than 500 MiB of
# its memory are in use. Any failing or unparsable query counts as busy, so a
# broken query never launches on a busy GPU.
busy() {
    local u=${UUIDS[$1]} apps mem
    flock -n "$(lockfile "$u")" true 2>/dev/null || return 0
    apps=$(SMI --query-compute-apps=gpu_uuid --format=csv,noheader) || return 0
    grep -q "$u" <<< "$apps" && return 0
    mem=$(SMI --query-gpu=memory.used --format=csv,noheader,nounits -i "$u") || return 0
    [[ $mem =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || return 0
    [ "$mem" -gt 500 ]
}

idle=()
for ((g = 0; g < NGPU; g++)); do idle[$g]=0; done
while true; do
    for ((g = 0; g < NGPU; g++)); do
        if busy $g; then idle[$g]=0; else idle[$g]=$((idle[$g] + 1)); fi
        if [ "${idle[$g]}" -ge "$NEED" ]; then
            uuid=${UUIDS[$g]}
            exec 9> "$(lockfile "$uuid")"
            flock -n 9 || { exec 9>&-; idle[$g]=0; continue; }   # claimed meanwhile
            echo "$(date '+%F %T') GPU $g ($uuid) idle, starting on $(hostname): $*" >> "$LOG"
            ( while sleep 60; do
                  echo "$(date '+%F %T') $(SMI --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep "$uuid" | tr '\n' ';')"
              done ) >> "$LOG.gpuapps" 2>&1 9>&- &
            mon=$!
            CUDA_VISIBLE_DEVICES=$uuid "$@" >> "$LOG" 2>&1
            rc=$?
            kill $mon 2>/dev/null
            echo "$(date '+%F %T') done rc=$rc" >> "$LOG"
            exit $rc
        fi
    done
    sleep 60
done
