
thedir=$(ls -t1 | head -n 1)
if [[ -d "$thedir"/gpu1 ]]; then
  cd "$thedir"/gpu1
  sbatch job.bash
elif [[ -d "$thedir"/cpu1 ]]; then
  cd "$thedir"/cpu1
  sbatch job.bash
else
  echo "Error: could not find a gpu1 or cpu1 directory in $thedir"
  exit 1
fi
