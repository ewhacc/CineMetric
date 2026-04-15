#!/bin/bash

if [ "$#" -lt 2 ]; then
    exit 1
fi

MUSIC_PATH=$1
OUTPUT_DIR=$2

cd Music2Emotion
python inference.py \
--audio_path "$MUSIC_PATH" \
--output_dir "$OUTPUT_DIR"
cd -

