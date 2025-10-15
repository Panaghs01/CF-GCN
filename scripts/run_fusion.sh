#!/usr/bin/env bash

gpus=-1
checkpoint_root=checkpoint_fusion
data_name=SenForFlood

img_size=512
batch_size=8
lr=0.001
max_epochs=200
net_G=base_GCN_with_fusion
lr_policy=linear
dataset=CDDataset_fusion
split=train10
split_val=test10
project_name=CD_cut_f1_${net_G}_${data_name}_b${batch_size}_${split}_${split_val}_${max_epochs}_${lr_policy}

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} \
 --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --dataset ${dataset} \
  --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}