#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
learning_rate=1e-3
DIR="/data1/cs330/project/train/model3"

python /data1/cs330/project/Train_simple.py --learning_rate ${learning_rate} --log_dir ${DIR} --maxEpoch 30
