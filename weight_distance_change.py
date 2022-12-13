import random
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import tensorflow as tf
import os
import sys
import model as model_path
import glob
from numpy.linalg import norm


learning_rate=0.001

def get_weight_matrix(fine_tune_weights):
    #print(fine_tune_weights)

    #======model weight matrxi embedding:
    base_model=model_path.Model_transfer()
    checkpoint = tf.train.Checkpoint(base_model)
    checkpoint.restore(fine_tune_weights).expect_partial()

    opt=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    base_model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["categorical_accuracy"])
    base_model.build(input_shape=(None,28,28,3))

    #======just use last conv layer
    weigth_matrix=[]
    for layer in base_model.layers[2:3]:
        weights=layer.get_weights()[0]
        weigth_np=np.array(weights,dtype=object).astype('float32')
        weigth_np_b=np.swapaxes(weigth_np,3,0)
        weigth_np_r=np.reshape(weigth_np_b,(weigth_np_b.shape[0],weigth_np_b.shape[1]*weigth_np_b.shape[2]*weigth_np_b.shape[3]))
        weigth_np_flat=weigth_np.flatten()

        weights_bias=layer.get_weights()[1]
        weigth_np_bias=np.array(weights_bias,dtype=object).astype('float32')
    #print(">>>>>final shape",weigth_np_flat.shape)
    return weigth_np_flat,weigth_np_bias

def get_fine_tune_epoch_weights(fine_tune_weights):
    weight_list=[]
    bias_list=[]
    #print(fine_tune_weights)
    for weight in fine_tune_weights:
        weigth,bias=get_weight_matrix(weight)
        weight_list.append(weigth)
        bias_list.append(bias)
    
    weight_np=np.array(weight_list,dtype=object).astype('float32')
    bias_np=np.array(bias_list,dtype=object).astype('float32')
    print("your fine tune weight list shape",weight_np.shape)
    return weight_np,bias_np

def get_distance(weight_orginal,bias_original,weight_np_list,bias_np_list):
    distance_weight=[]
    distance_bias=[]
    for tune_weight in weight_np_list:
        w_d=norm(weight_orginal-tune_weight)
        distance_weight.append(w_d)

    for tune_bias in bias_np_list:
        w_bb=norm(bias_original-tune_bias)
        distance_bias.append(w_bb)

    distance_w=np.array(distance_weight,dtype=object).astype('float32')
    distance_b=np.array(distance_bias,dtype=object).astype('float32')
    print("your distance shape",distance_w.shape)
    return distance_w,distance_b


if __name__ == "__main__":
    original_weights='/data1/cs330/project/train/model9/weights/weights.30'
    fine_tune_path='/data1/cs330/project/fine_tune/model9/weights'
    out_path="/data1/cs330/project/weight_distance"
    weight_orginal,bias_original=get_weight_matrix(original_weights)
    print(">>>>>>sucess got inital weight")
    folder_name=os.listdir(fine_tune_path)
    dir_list=[os.path.join(fine_tune_path,f) for f in folder_name]
    dir_list.sort()
    print(">>>>>>>your fine sorted list",dir_list)
    weight_np_list,bias_np_list=get_fine_tune_epoch_weights(dir_list)
    distance_w,distance_b=get_distance(weight_orginal,bias_original,weight_np_list,bias_np_list)
    np.save(os.path.join(out_path,"weight_distance_model9.npy"),distance_w)
    np.save(os.path.join(out_path,"bias_distance_model9.npy"),distance_b)


