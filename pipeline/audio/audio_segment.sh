#!/bin/bash

if [ "$#" -lt 2 ]; then
    exit 1
fi

AUDIO_PATH=$1
OUTPUT_DIR=$2
FORMAT=${3:-"csv"} # 세 번째 인자가 없으면 기본값으로 'csv' 사용

cd TVSM/inference/
python inference.py \
    --audio_path "$AUDIO_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --format "$FORMAT"
cd -
echo ""
