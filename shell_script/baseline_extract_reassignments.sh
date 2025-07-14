#!/bin/bash

# 입력 파일: 첫 번째 인자
INPUT_FILE="$1"

# 파일이 존재하는지 확인
if [ ! -f "$INPUT_FILE" ]; then
  echo "Usage: $0 <log_file>"
  echo "Error: File '$INPUT_FILE' not found."
  exit 1
fi

# Iteration X: N Reassignments 에서 N 값만 추출
grep -E "Iteration [0-9]+: [0-9]+ Reassignments" "$INPUT_FILE" | \
awk '{print $3}'
