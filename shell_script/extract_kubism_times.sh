#!/bin/bash

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Usage: $0 <log_file>"
  echo "Error: File '$INPUT_FILE' not found."
  exit 1
fi

awk '
/^\[Current Iteration = / {iteration=$3}
/\(6\) Early Skip Ratio Marking kernel execution time\(1_1\):/ {early_skip=$8}
/\(7\) Calculation Skip Ratio execution time:/ {calc_skip=$6}
/\(6\) Local Filter Marking kernel execution time:/ {local_filter=$7}
/\(8\) CPU-GPU Data Point Partition kernel execution time:/ {partition=$8}
/\(9-3-4\) kernel start ~ sync start:/ {rwd_sync=$7; case_type=1}
/\(9-4-4\) Only Heterogeneous kernel start ~ sync start:/ {rwd_sync=$7; case_type=2}
/\(9-3-5\) Host To Device Data Transfer Time:/ {h2d=$7; if (case_type==1) {
    printf("Iteration %d (Case 1): EarlySkip=%.6f ms CalcSkip=%.6f ms LocalFilter=%.6f ms Partition=%.6f ms Sync=%.6f ms H2D=%.6f ms\n", iteration, early_skip, calc_skip, local_filter, partition, rwd_sync, h2d)
}}
/\(9-4-5\) Host To Device Data Transfer Time:/ {h2d=$7; if (case_type==2) {
    printf("Iteration %d (Case 2): EarlySkip=%.6f ms CalcSkip=%.6f ms Partition=%.6f ms Sync=%.6f ms H2D=%.6f ms\n", iteration, early_skip, calc_skip, partition, rwd_sync, h2d)
}}
' "$INPUT_FILE"
