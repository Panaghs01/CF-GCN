#!/bin/bash

mkdir -p "MASKS"

counterM=1

for folder in CEMS/*/; do
    if [[ -d "$folder" ]]; then
        if [[ -d "$folder/flood_mask" ]]; then
            for img in "$folder/flood_mask"/*; do
                ext="${img##*.}"
                cp "$img" "MASKS/s2_$(printf %06d $counterM).$ext"
                counterM=$((counterM+1))
            done
        fi
    fi
done

for folder in DFO/*/*/; do
    if [[ -d "$folder" ]]; then
        if [[ -d "$folder/flood_mask" ]]; then
            for img in "$folder/flood_mask"/*; do
                ext="${img##*.}"
                cp "$img" "MASKS/s2_$(printf %06d $counterM).$ext"
                counterM=$((counterM+1))
            done
        fi
    fi
done
