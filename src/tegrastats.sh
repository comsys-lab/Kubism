#!/bin/bash

# Define the log file for tegrastats
LOGFILE="/home/du6293/Power_Efficiency/tegrastats_log_$1.txt"

# tegrastats interval: 1000ms
tegrastats --interval 1000 --logfile $LOG_FILE &

TEGRSTATS_PID=$!

echo $TEGRSTATS_PID > "/home/du6293/Power_Efficiency/tegrastats_pid_$1.txt"
