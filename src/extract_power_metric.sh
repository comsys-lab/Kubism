#!/bin/bash

# 로그 파일들이 있는 디렉토리 경로
log_dir="/home/du6293/kmcuda/src/Power_2M_256/Kubism"

# 출력 파일 경로 지정
output_file="/home/du6293/output1.txt"

# 출력 파일 초기화 (기존 내용 덮어쓰기)
> "$output_file"

# 로그 파일들을 순차적으로 처리
for i in {1..29}; do
    file_path="$log_dir/tegra_log_$i.txt"

    # 파일이 존재하는지 확인
    if [[ -f "$file_path" ]]; then
        # 입력 파일을 읽어서 마지막 줄 추출
        last_line=$(tail -n 1 "$file_path")

        # VDD_GPU_SOC 값 추출 (단위 제거)
        vdd_gpu_soc=$(echo "$last_line" | grep -oP 'VDD_GPU_SOC \d+mW/\K\d+')

        # VDD_CPU_CV 값 추출 (단위 제거)
        vdd_cpu_cv=$(echo "$last_line" | grep -oP 'VDD_CPU_CV \d+mW/\K\d+')

        # 출력 파일에 추가 (단위 없이)
        echo "File: tegra_log_$i.txt" >> "$output_file"
        echo "VDD_GPU_SOC: $vdd_gpu_soc" >> "$output_file"
        echo "VDD_CPU_CV: $vdd_cpu_cv" >> "$output_file"
        echo "" >> "$output_file"  # 빈 줄 추가
    else
        echo "File tegra_log_$i.txt not found" >> "$output_file"
    fi
done
