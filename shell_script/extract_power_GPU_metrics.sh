#!/bin/bash

DIR_PATH="$1"

if [ ! -d "$DIR_PATH" ]; then
  echo "Usage: $0 <directory_path>"
  echo "Error: Directory '$DIR_PATH' not found."
  exit 1
fi

for FILE in "$DIR_PATH"/tegra_log_{5..80}.txt; do
  if [ -f "$FILE" ]; then
    echo "Processing $FILE"
    grep "VDD_GPU_SOC" "$FILE" | while read -r line; do
      gpu_val=$(echo "$line" | sed -E 's/.*VDD_GPU_SOC [0-9]+mW\/([0-9]+)mW.*/\1/')
      echo "$gpu_val"
    done
  else
    echo "File $FILE not found, skipping."
  fi
done
