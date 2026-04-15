i#!/bin/bash

if [ -z "$1" ]; then
    exit 1
fi

VIDEO_PATH=$1
SHOT_RESULT_DIR=$2:-"./results"}

mkdir -p "$SHOT_RESULT_DIR"

cd SceneSeg/pre 
python ShotDetection/shotdetect.py \
    --print_result \
    --save_keyf \
    --save_keyf_txt \
    --video_path "$VIDEO_PATH" \
    --save_data_root_path "$SHOT_RESULT_DIR"
cd -
