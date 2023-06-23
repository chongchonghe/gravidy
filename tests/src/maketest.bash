#!/bin/bash

set -e

platform=$1

date_time=$(TZ=":US/Eastern" date +"%Y-%m-%d_%H:%M:%S_EST")
folder_name="test_$date_time"
mkdir -p "$folder_name"
cd "$folder_name"

mkdir -p "${platform}1"
cd "${platform}1"

# create job.bash
if [[ "$platform" == "gpu" ]]; then
  cp ../../../src/job_template1_gpu.bash job.bash
else
  cp ../../../src/job_template1_cpu.bash job.bash
fi


# list the most recent file with the pattern test_*

