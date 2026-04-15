#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_video_path> <output_video_path>"
    exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2

ffmpeg -y -i "$INPUT_PATH" \
       -c:v h264 \
       -c:a aac \
       "$OUTPUT_PATH"

echo "$OUTPUT_PATH"
