#!/bin/bash

mkdir -p "A" "B" "MASKS" "C" "D"

sat=$1  # "S1" or "S2"
root=$2
mask=$3
modes=('before' 'during')
counterA=1
counterB=1
counterM=1

if [[ $sat == 's1' ]]; then
    bef='C'
    dur='D'
else
    bef='A'
    dur='B'
fi

for mode in ${modes[@]}; do
    for folder in ${root}/CEMS/*/; do
        if [[ -d "$folder" ]]; then
            echo "inside folder: $folder/${sat}_${mode}_flood"
            if [[ -d "$folder/${sat}_${mode}_flood" ]]; then
                for img in "$folder/${sat}_${mode}_flood"/*; do
                    ext="${img##*.}"

                    if [[ $mode == 'before' ]]; then
                    
                        cp "$img" "${bef}/s2_$(printf %06d $counterA).$ext"
                        counterA=$((counterA+1))
                    else
                        cp "$img" "${dur}/s2_$(printf %06d $counterB).$ext"
                        counterB=$((counterB+1))
                    fi
                    
                done
            fi
            if [[ $mask == 'no' ]];then
                continue
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

    for folder in ${root}/DFO/*/*/; do
        if [[ -d "$folder" ]]; then
            echo "inside folder: $folder/${sat}_${mode}_flood"
            if [[ -d "$folder/${sat}_${mode}_flood" ]]; then
                for img in "$folder/${sat}_${mode}_flood"/*; do
                    ext="${img##*.}"
                    if [[ $mode == 'before' ]]; then
                        cp "$img" "${bef}/s2_$(printf %06d $counterA).$ext"
                        counterA=$((counterA+1))
                    else
                        cp "$img" "${dur}/s2_$(printf %06d $counterB).$ext"
                        counterB=$((counterB+1))
                    fi
                done
            fi
            if [[ $mask == 'no' ]];then
                continue
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