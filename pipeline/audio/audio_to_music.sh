#!/bin/bash

if [ "$#" -lt 2 ]; then
    exit 1
fi

AUDIO_PATH=$1
OUTPUT_DIR=$2

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "출력 디렉토리가 존재하지 않아 새로 생성합니다: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi


cd uvr

python separate.py \
--audio_path "$AUDIO_PATH" \
--output_dir "$OUTPUT_DIR"

cd -

