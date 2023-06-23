# set -x

thedir=$(ls -t1 | head -n 1)
if [[ $thedir == "Makefile" ]]; then
  thedir=$(ls -t1 | head -n 2 | tail -n 1)
fi
echo thedir = $thedir
if [[ $1 == "gpu" ]]; then
  if [[ -d "$thedir"/gpu1 ]]; then
    python ../src/diff.py reference3/gpu1/out/out1 $thedir/gpu1/out/output.out.snapshot_00001 
    # diff "$thedir"/gpu1/out/output.out.snapshot_00003 $1/gpu1/out/output.out.snapshot_00003 
  fi
else
  if [[ -d "$thedir"/cpu1 ]]; then
    python ../src/diff.py reference-cpu/cpu1/out/output.out.snapshot_00001 $thedir/cpu1/out/output.out.snapshot_00001 
    # diff "$thedir"/cpu1/out/output.out.snapshot_00003 $1/cpu1/out/output.out.snapshot_00003 
  fi
fi