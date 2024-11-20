#!/bin/bash

# Define the directory and model id combinations
declare -A combinations=(
    ["iter_0/net_1"]="0,4,5,7,11"
    ["iter_0/net_10"]="0,1,4,6,8,11"
    ["iter_0/net_13"]="0,2,3,6,11"
    ["iter_0/net_19"]="1,2,3,6,11"
    ["iter_0/net_2"]="4,11"
    ["iter_0/net_23"]="0,1,2,5,11"
    ["iter_0/net_24"]="0,1,2,6,11"
    ["iter_0/net_3"]="6,11"
    ["iter_0/net_4"]="2,11"
    ["iter_0/net_8"]="2,8,11"
    ["iter_2/net_10"]="2,3,4,11"
    ["iter_2/net_11"]="1,6,7,11"
    ["iter_2/net_13"]="0,1,2,7,11"
    ["iter_2/net_15"]="0,1,2,3,5,7,11"
    ["iter_2/net_16"]="1,4,7,9,11"
    ["iter_2/net_2"]="1,3,4,11"
    ["iter_2/net_5"]="4,8,11"
    ["iter_2/net_7"]="0,1,5,8,9,11"
    ["iter_2/net_8"]="1,2,3,4,5,7,11"
    ["iter_2/net_9"]="0,5,7,9,11"
    ["iter_3/net_0"]="2,3,4,6,9,11"
    ["iter_3/net_1"]="2,3,4,6,8,11"
    ["iter_3/net_10"]="1,3,4,5,8,11"
    ["iter_3/net_11"]="1,3,4,5,7,11"
    ["iter_3/net_12"]="1,2,4,6,9,11"
    ["iter_3/net_13"]="1,2,4,6,8,11"
    ["iter_3/net_14"]="1,2,4,6,7,11"
    ["iter_3/net_15"]="1,2,4,5,9,11"
    ["iter_3/net_16"]="1,2,4,5,8,11"
    ["iter_3/net_17"]="1,2,4,5,7,11"
    ["iter_3/net_18"]="1,2,3,6,9,11"
    ["iter_3/net_19"]="1,2,3,6,8,11"
    ["iter_3/net_2"]="2,3,4,6,7,11"
    ["iter_3/net_20"]="1,2,3,6,7,11"
    ["iter_3/net_21"]="1,2,3,5,9,11"
    ["iter_3/net_22"]="1,2,3,5,8,11"
    ["iter_3/net_23"]="1,2,3,5,7,11"
    ["iter_3/net_24"]="1,2,3,4,9,11"
    ["iter_3/net_25"]="1,2,3,4,8,11"
    ["iter_3/net_26"]="1,2,3,4,7,11"
    ["iter_3/net_3"]="2,3,4,5,9,11"
    ["iter_3/net_4"]="2,3,4,5,8,11"
    ["iter_3/net_5"]="2,3,4,5,7,11"
    ["iter_3/net_6"]="1,3,4,6,9,11"
    ["iter_3/net_7"]="1,3,4,6,8,11"
    ["iter_3/net_8"]="1,3,4,6,7,11"
    ["iter_3/net_9"]="1,3,4,5,9,11"
    ["iter_4/net_0"]="0,1,2,3,4,11"
    ["iter_4/net_1"]="0,1,7,8,11"
    ["iter_4/net_10"]="0,1,3,7,8,11"
    ["iter_4/net_11"]="1,2,5,6,11"
    ["iter_4/net_12"]="0,2,4,5,8,9,11"
    ["iter_4/net_13"]="0,1,2,3,4,7,9,11"
    ["iter_4/net_14"]="1,2,5,11"
    ["iter_4/net_15"]="0,2,3,4,11"
    ["iter_4/net_16"]="2,4,5,8,9,11"
    ["iter_4/net_17"]="0,1,3,4,7,11"
    ["iter_4/net_18"]="3,5,7,9,11"
    ["iter_4/net_3"]="0,1,2,7,11"
    ["iter_4/net_4"]="0,1,4,11"
    ["iter_4/net_5"]="0,1,2,7,9,11"
    ["iter_4/net_6"]="1,8,11"
    ["iter_4/net_8"]="2,3,5,6,9,11"
    ["iter_4/net_9"]="0,4,6,9,11"
    ["iter_5/net_0"]="6,11"
    ["iter_5/net_11"]="3,4,11"
    ["iter_5/net_2"]="8,11"
    ["iter_5/net_4"]="2,3,5,6,7,8,9,11"
    ["iter_5/net_5"]="2,4,11"
    ["iter_5/net_6"]="0,1,4,5,7,9,11"
    ["iter_5/net_7"]="2,7,11"
    ["iter_6/net_0"]="2,3,4,5,6,9,11"
    ["iter_6/net_1"]="0,2,4,5,7,8,11"
    ["iter_6/net_2"]="0,2,5,6,7,8,11"
    ["iter_6/net_3"]="1,2,3,4,6,7,8,11"
    ["iter_6/net_4"]="0,1,2,3,5,7,8,11"
    ["iter_6/net_5"]="0,3,4,7,8,9,11"
    ["iter_6/net_6"]="0,1,2,3,4,7,11"
    ["iter_6/net_7"]="0,1,4,6,7,8,9,11"
    ["iter_6/net_8"]="0,1,3,4,6,8,11"
    ["iter_6/net_9"]="0,1,3,4,5,6,7,11"
    ["iter_7/net_3"]="9,11"
    ["iter_7/net_4"]="1,3,11"
)

# Function to run the python script
run_script() {
    local relpath=$1
    local ids=$2
    local logfile="outputs-eenn/nas_logs/${relpath//\//_}_$ids.log"
    mkdir -p outputs-eenn/nas_logs
    echo "Starting: python main_stream_eenn_nas.py -relpath $relpath -id $ids"
    python main_stream_eenn_nas.py -relpath "$relpath" -id "$ids" &> "$logfile"
    echo "Finished: python main_stream_eenn_nas.py -relpath $relpath -id $ids"
}

# Export the function so it can be used by parallel
export -f run_script

# Initialize conda for the current shell session
source ~/anaconda3/etc/profile.d/conda.sh && conda activate stream

# Iterate through the combinations and run them in batches of 12
batch_size=12
batch=()

for relpath in "${!combinations[@]}"; do
    ids="${combinations[$relpath]}"
    batch+=("$relpath,$ids")
    if [ ${#batch[@]} -eq $batch_size ]; then
        for item in "${batch[@]}"; do
            IFS=',' read -r relpath ids <<< "$item"
            run_script "$relpath" "$ids" &
        done
        wait
        batch=()
    fi
done

# Run any remaining jobs in the last batch
if [ ${#batch[@]} -gt 0 ]; then
    for item in "${batch[@]}"; do
        IFS=',' read -r relpath ids <<< "$item"
        run_script "$relpath" "$ids" &
    done
    wait
fi

echo "All jobs completed."