#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
learning_rate=1e-4
DIR="/data1/cs330/project/tdl_model"
fine_tune_weights_list=["/data1/cs330/project/weight_matrix/model1",'/data1/cs330/project/weight_matrix/model2','/data1/cs330/project/weight_matrix/model3']

python /data1/cs330/project/td_train.py --learning_rate ${learning_rate} \
--log_dir ${DIR} \
--maxEpoch 30 \
--fine_tune_weights_list ${fine_tune_weights_list}
