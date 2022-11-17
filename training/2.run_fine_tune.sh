#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
learning_rate=1e-4
DIR="/data1/cs330/project/fine_tune/model1_tune"
MODEL_WEIGHT="/data1/cs330/project/train/model1"

python /data1/cs330/project/fine_tune.py --fine_tune_weights ${MODEL_WEIGHT} --learning_rate ${learning_rate} --log_dir ${DIR} --maxEpoch 15
