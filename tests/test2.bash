# run tests for this code
# Usage: ./test1.bash

set -e 

# create a folder with current date and time in EST time
# date and time format in EST: YYYY-MM-DD_HH:MM:SS_EST
date_time=$(TZ=":US/Eastern" date +"%Y-%m-%d_%H:%M:%S_EST")
folder_name="output/test2/test_$date_time"
mkdir -p "$folder_name"
# cd "$folder_name"

for power in {1..8}
do
  N=$((2**($power+6)))
  echo "N = $N"
  input="../input/0${power}-nbody-p${N}_m1.in"
  echo $input
  # output_dir="${folder_name}/0${power}-nbody-p${N}_m1.out"
  output_dir="${folder_name}/N${N}"
  mkdir -p $output_dir
  # cd $output
  ../src/gravidy-cpu -i "$input" -o "${output_dir}/output.gravidy" -t 5 2>&1 >> "${output_dir}/log.txt"
  # ../../../../../src/gravidy-cpu -i ../../../../input/$input -o output.gravidy -t 50 2>&1 >> log.txt
  # cd ..
done

python src/benchmark.py ${folder_name} "${folder_name}/benchmark.jpg"
