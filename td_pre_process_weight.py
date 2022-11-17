import random
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import tensorflow as tf
import os
import sys
import model as model_path
import glob


learning_rate=0.001

def get_weight_matrix_input(fine_tune_weights_list):
    weigth_matrix=[]
    for fine_tune_weights in fine_tune_weights_list:
        print(fine_tune_weights)

        #======model weight matrxi embedding:
        base_model=model_path.Model_transfer()
        checkpoint = tf.train.Checkpoint(base_model)
        checkpoint.restore(fine_tune_weights).expect_partial()

        opt=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        base_model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["categorical_accuracy"])
        base_model.build(input_shape=(None,28,28,3))
        # base_model.summary()

        #======just use last conv layer
        for layer in base_model.layers[2:3]:
            # print(layer.name)
            weights=layer.get_weights()[0]
            weigth_np=np.array(weights,dtype=object)
            print(weigth_np.shape)
            weigth_np_b=np.swapaxes(weigth_np,3,0)
            print(weigth_np_b.shape)
            weigth_np_r=np.reshape(weigth_np_b,(weigth_np_b.shape[0],weigth_np_b.shape[1]*weigth_np_b.shape[2]*weigth_np_b.shape[3]))
            print(weigth_np_r.shape)
            #weigth_np_flat=weigth_np.flatten()
            #print(weigth_np_flat.shape)
        
        weigth_matrix.append(weigth_np_r)
    weigth_matrix_np=np.array(weigth_matrix,dtype=object).astype('float32')
    print(">>>>>final shape",weigth_matrix_np.shape)
    return weigth_matrix_np

path="/data1/cs330/project/weight_matrix"
folder_name=os.listdir(path)

dir_list=[os.path.join(path,f) for f in folder_name]
print(dir_list)
weigth_matrix_np=get_weight_matrix_input(dir_list)
np.save("/data1/cs330/project/data/weight_matrix.npy",weigth_matrix_np)