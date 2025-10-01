#!/bin/bash

rand=$(shuf -i 1-100 -n 1)
echo "${rand}"
echo "${rand}"
for line in $(cat full_list.txt); do  
    if [[ $rand -gt 20 ]]; then
        echo "${line}" >> train.txt
    else
        echo "${line}" >> test.txt
    fi
    rand=$(shuf -i 1-100 -n 1)
done