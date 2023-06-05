# run tests for this code
# Usage: ./test1.bash

set -e 

# create a folder with current date and time in EST time
# date and time format in EST: YYYY-MM-DD_HH:MM:SS_EST
date_time=$(TZ=":US/Eastern" date +"%Y-%m-%d_%H:%M:%S_EST")
folder_name="test_$date_time"
mkdir -p $folder_name

# run the code
cd $folder_name
pwd 2>&1 >> log.txt
# ../src/gravidy-cpu
