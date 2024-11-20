#!/bin/bash

# Initialize conda for the current shell session
source ~/anaconda3/etc/profile.d/conda.sh

conda activate stream && 
# Launch python main_stream_eenn.py with -pb 8 and -pc 8 and:
# -id 0,1,5
python main_stream_eenn.py -id 0,1,5 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_0_1_5_8_8.log &
# -id 0,2,4
python main_stream_eenn.py -id 0,2,4 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_0_2_4_8_8.log &
# -id 0,2,5
python main_stream_eenn.py -id 0,2,5 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_0_2_5_8_8.log &
# -id 0,2,6
python main_stream_eenn.py -id 0,2,6 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_0_2_6_8_8.log &
# -id 0,2,7
python main_stream_eenn.py -id 0,2,7 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_0_2_7_8_8.log &
# -id 0,4,6
python main_stream_eenn.py -id 0,4,6 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_0_4_6_8_8.log &
# -id 0,4,7
python main_stream_eenn.py -id 0,4,7 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_0_4_7_8_8.log &
# -id 0,4,8
python main_stream_eenn.py -id 0,4,8 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_0_4_8_8_8.log &
# -id 1,2,4
python main_stream_eenn.py -id 1,2,4 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_1_2_4_8_8.log &
# -id 1,4,6
python main_stream_eenn.py -id 1,4,6 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_1_4_6_8_8.log &
# -id 2,3,4
python main_stream_eenn.py -id 2,3,4 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_2_3_4_8_8.log &
# -id 2,3,6
python main_stream_eenn.py -id 2,3,6 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_2_3_6_8_8.log &
# -id 3,5,8
python main_stream_eenn.py -id 3,5,8 -pb 8 -pc 8 &> outputs-eenn/focus_logs/eenn_precision_3_5_8_8_8.log &
