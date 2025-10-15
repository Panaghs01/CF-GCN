#!/usr/bin/env bash

gpus=0
checkpoint_root=checkpoint_fusion
data_name=SenForFlood

img_size=256
batch_size=6
lr=0.001
max_epochs=100
net_G=base_GCN_with_fusion
lr_policy=linear
dataset=CDDataset_fusion
split=train
split_val=test
project_name=CD_fusion

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} \
 --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --dataset ${dataset} \
  --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} \
   --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --accumulation_steps 4