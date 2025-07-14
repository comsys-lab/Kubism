#!/bin/bash

# 입력 파일: 첫 번째 인자
INPUT_FILE="$1"

# 파일이 존재하는지 확인
if [ ! -f "$INPUT_FILE" ]; then
  echo "Usage: $0 <log_file>"
  echo "Error: File '$INPUT_FILE' not found."
  exit 1
fi

# 해당 라인에서 숫자(ms) 값만 추출
grep "(6) Local Filter Baseline kernel execution time" "$INPUT_FILE" | \
awk '{for(i=1;i<=NF;i++) if($i ~ /ms$/) print $(i-1)}'
