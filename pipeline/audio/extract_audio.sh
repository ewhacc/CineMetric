#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_video_path> <output_audio_path>"
    exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2

ffmpeg -y -i "$INPUT_PATH" \
       -q:a 0 \
       -map a \
       -ac 1 \
       "$OUTPUT_PATH"

echo "$OUTPUT_PATH"
