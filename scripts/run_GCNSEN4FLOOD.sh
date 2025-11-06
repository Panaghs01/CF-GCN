#!/usr/bin/env bash

gpus=0
checkpoint_root=checkpoint_fusion
data_name=SenForFlood

img_size=512
batch_size=2
lr=0.001
max_epochs=200
net_G=base_GCN
lr_policy=linear
dataset=CDDataset
split=train
split_val=test
project_name=CD_S1_CFGCN_base
loss=ce_dice
accumulation_steps=16
python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} \
 --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} \
   --dataset ${dataset} --gpu_ids ${gpus}\
   --max_epochs ${max_epochs} --project_name ${project_name} \
   --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --accumulation_steps ${accumulation_steps}\
   --loss ${loss}