#!/bin/bash

# Define the log file for tegrastats
LOGFILE="/home/du6293/Power_Efficiency/tegrastats_log_$1.txt"

# tegrastats를 1초 간격으로 실행하여 VDD_GPU_SOC와 VDD_CPU_CV 값을 필터링하여 저장
tegrastats --interval 1000 --logfile $LOG_FILE &

# 실행된 tegrastats PID를 기록
TEGRSTATS_PID=$!

# PID를 따로 파일에 저장해 나중에 종료할 수 있도록 함
echo $TEGRSTATS_PID > "/home/du6293/Power_Efficiency/tegrastats_pid_$1.txt"