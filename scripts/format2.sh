#!/bin/bash

# Usage: ./collect_flood_images.sh /path/to/senforflood

ROOT=$1
OUT_A="A"   # pre-event
OUT_B="B"   # post-event
OUT_MASK="mask"  # flood mask
mkdir -p "$OUT_A" "$OUT_B" "$OUT_MASK"

count=1
for folder in "$ROOT"/*/EMS*/; do

    if [[ -d "$folder" ]]; then
        # Pre-event (before flood)
        if [[ -d "$folder/s2_before_flood" ]]; then
            count=1
            for img in "$folder/s2_before_flood"/*; do
                ext="${img##*.}"
                cp "$img" "$OUT_A/S2_$(printf "%06d" $count).$ext"
                count=$((count+1))
            done
        fi

        # Post-event (during flood)
        if [[ -d "$folder/s2_during_flood" ]]; then
            count=1
            for img in "$folder/s2_during_flood"/*; do
                ext="${img##*.}"
                cp "$img" "$OUT_B/S2_$(printf "%06d" $count).$ext"
                count=$((count+1))
            done
        fi
        if [[ -d "$folder/flood_mask" ]]; then
            count=1
            for img in "$folder/flood_mask"/*; do
                ext="${img##*.}"
                cp "$img" "$OUT_MASK/S2_$(printf "%06d" $count).$ext"
                count=$((count+1))
            done
        fi
    fi
done
