#!/bin/bash

mkdir -p "A" "B" "MASKS"

modes=('before' 'during')
counterA=1
counterB=1
counterM=1
for mode in ${modes[@]}; do
    for folder in CEMS/*/; do
        if [[ -d "$folder" ]]; then
            echo "inside folder: $folder/s2_${mode}_flood"
            if [[ -d "$folder/s2_${mode}_flood" ]]; then
                for img in "$folder/s2_${mode}_flood"/*; do
                    ext="${img##*.}"
                    if [[ $mode == 'before' ]]; then
                        cp "$img" "A/s2_$(printf %06d $counterA).$ext"
                        counterA=$((counterA+1))
                    else
                        cp "$img" "B/s2_$(printf %06d $counterB).$ext"
                        counterB=$((counterB+1))
                    fi
                    
                done
            fi
            if [[ -d "$folder/flood_mask" ]]; then
                for img in "$folder/flood_mask"/*; do
                    ext="${img##*.}"
                    cp "$img" "MASKS/s2_$(printf %06d $counterM)_mask.$ext"
                    counterM=$((counterM+1))
                done
            fi
        fi
    done

    for folder in DFO/*/*/; do
        if [[ -d "$folder" ]]; then
            echo "inside folder: $folder/s2_${mode}_flood"
            if [[ -d "$folder/s2_${mode}_flood" ]]; then
                for img in "$folder/s2_${mode}_flood"/*; do
                    ext="${img##*.}"
                    if [[ $mode == 'before' ]]; then
                        cp "$img" "A/s2_$(printf %06d $counterA).$ext"
                        counterA=$((counterA+1))
                    else
                        cp "$img" "B/s2_$(printf %06d $counterB).$ext"
                        counterB=$((counterB+1))
                    fi
                done
            fi
            if [[ -d "$folder/flood_mask" ]]; then
                for img in "$folder/flood_mask"/*; do
                    ext="${img##*.}"
                    cp "$img" "MASKS/s2_$(printf %06d $counterM)_mask.$ext"
                    counterM=$((counterM+1))
                done
            fi
        fi
    done
done