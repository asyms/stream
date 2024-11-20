#!/bin/bash

# Initialize conda for the current shell session
source ~/anaconda3/etc/profile.d/conda.sh

# Launch python main_stream_eenn.py with -id 3,5,8 and combinations -pb and -pc:
# -pb 32 -pc 32
# -pb 8 -pc 8
# -pb 8 -pc 4
# -pb 4 -pc 8
# -pb 4 -pc 4 
conda activate stream && 
python main_stream_eenn.py -id 3,5,8 -pb 32 -pc 32 &> outputs-eenn/precision_logs/eenn_precision_3_5_8_32_32.log &
python main_stream_eenn.py -id 3,5,8 -pb 8 -pc 8 &> outputs-eenn/precision_logs/eenn_precision_3_5_8_8_8.log &
python main_stream_eenn.py -id 3,5,8 -pb 8 -pc 4 &> outputs-eenn/precision_logs/eenn_precision_3_5_8_8_4.log &
python main_stream_eenn.py -id 3,5,8 -pb 4 -pc 8 &> outputs-eenn/precision_logs/eenn_precision_3_5_8_4_8.log &
python main_stream_eenn.py -id 3,5,8 -pb 4 -pc 4 &> outputs-eenn/precision_logs/eenn_precision_3_5_8_4_4.log &
