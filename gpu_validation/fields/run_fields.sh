#!/bin/bash
# The four problems in mumax3 3.11 and in JuliaMag, on one GPU (through run_when_gpu_idle.sh).
#   bash gpu_validation/fields/run_fields.sh REPO OUT
REPO=$1; OUT=$2
MUMAX=$HOME/mumax3/mumax3.11.1_linux_cuda12.0/mumax3
JULIA=$HOME/opt/julia-1.11.9/bin/julia
mkdir -p $OUT/mumax3 $OUT/juliamag
for c in pulse regions zhangli_half nanocontact; do
    echo "== $(date '+%F %T') mumax3 $c"
    cp $REPO/gpu_validation/fields/mumax3/$c.mx3 $OUT/mumax3/
    (cd $OUT/mumax3 && mkdir -p /tmp/mumax3cache_$USER; $MUMAX -gpu 0 -http "" -cache /tmp/mumax3cache_$USER $c.mx3 > $c.log 2>&1)
    echo "== $(date '+%F %T') juliamag $c"
    $JULIA --project=$REPO $REPO/gpu_validation/fields/fields_juliamag.jl $c $OUT/juliamag > $OUT/juliamag/$c.log 2>&1
done
echo "== $(date '+%F %T') done"
