#!/bin/bash

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Usage: $0 <log_file>"
  echo "Error: File '$INPUT_FILE' not found."
  exit 1
fi

echo "Early Skip Ratio Marking kernel execution time(1_1):"
grep "(6) Early Skip Ratio Marking kernel execution time(1_1):" "$INPUT_FILE" | awk '{print $(NF-1)}'

echo "Calculation Skip Ratio execution time:"
grep "(7) Calculation Skip Ratio execution time:" "$INPUT_FILE" | awk '{print $(NF-1)}'

echo "Local Filter Marking kernel execution time:"
grep "(6) Local Filter Marking kernel execution time:" "$INPUT_FILE" | awk '{print $(NF-1)}'

echo "CPU-GPU Data Point Partition kernel execution time:"
grep "(8) CPU-GPU Data Point Partition kernel execution time:" "$INPUT_FILE" | awk '{print $(NF-1)}'

echo "Device to Host Data Transfer Time:"
grep "(9-3-3) Device to Host Data Transfer Time:" "$INPUT_FILE" | awk '{print $(NF-1)}'

echo "kernel start ~ sync start:"
grep "(9-3-4) kernel start ~ sync start:" "$INPUT_FILE" | awk '{print $(NF-1)}'

echo "Host To Device Data Transfer Time:"
grep "(9-3-5) Host To Device Data Transfer Time:" "$INPUT_FILE" | awk '{print $(NF-1)}'
