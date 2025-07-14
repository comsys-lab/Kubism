#!/bin/bash

# 입력 파일: 첫 번째 인자
INPUT_FILE="$1"

# 파일 존재 여부 확인
if [ ! -f "$INPUT_FILE" ]; then
  echo "Usage: $0 <log_file>"
  echo "Error: File '$INPUT_FILE' not found."
  exit 1
fi

# Current iteration Skip Ratio = 수치만 추출
grep "Current iteration Skip Ratio =" "$INPUT_FILE" | \
awk '{for(i=1;i<=NF;i++) if($i=="Skip" && $(i+1)=="Ratio" && $(i+2)=="=") print $(i+3)}'
