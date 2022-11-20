import random
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import tensorflow as tf
import os
import sys
import model as model_path




def get_weight_matrix_input_predict(fine_tune_weights):
    learning_rate=0.001
    #======model weight matrxi embedding:
    base_model=model_path.Model_transfer()
    checkpoint = tf.train.Checkpoint(base_model)
    checkpoint.restore(fine_tune_weights).expect_partial()

    opt=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    base_model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["categorical_accuracy"])
    base_model.build(input_shape=(None,28,28,3))

    #======just use last conv layer
    for layer in base_model.layers[2:3]:
        weights=layer.get_weights()[0]
        weigth_np=np.array(weights,dtype=object)
        print(weigth_np.shape)
        weigth_np_b=np.swapaxes(weigth_np,3,0)
        print(weigth_np_b.shape)
        weigth_np_r=np.reshape(weigth_np_b,(weigth_np_b.shape[0],weigth_np_b.shape[1]*weigth_np_b.shape[2]*weigth_np_b.shape[3]))
        print(weigth_np_r.shape)
    weigth_np_r=np.expand_dims(weigth_np_r,axis=0).astype('float32')
    print(">>>>>final shape",weigth_np_r.shape)
    return weigth_np_r

def predict(args):
    x_weights_matrix =args.x_weights_matrix
    data_matrix =args.data_matrix
    choice_net_weights=args.choice_net_weights

    X_test=np.load(x_weights_matrix)
    X_test2=np.load(data_matrix)
    #load model
    print(">>>>your fist",X_test.shape)
    print(">>>>your 2nd X",X_test2.shape)

    model=model_path.Model_ChoiceNet_simple()
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(choice_net_weights).expect_partial()

    pred=model.predict([X_test,X_test2],verbose=0)
    print(">>>>your prediction",pred)


def main():
    parser = ArgumentParser(description="Train parameters ")

    parser.add_argument('--x_weights_matrix', type=str, default="/data1/cs330/project/data/x_feature_test/weight_training_matrix.npy",
                        help="path for the reducd weight matrix")
    
    parser.add_argument('--choice_net_weights', type=str, default="/data1/cs330/project/tdl_model/tdl_2input/weights",
                        help="weights for choicenet")

    parser.add_argument('--data_matrix', type=str, default="/data1/cs330/project/data/x_feature_test/x_test_new.npy",
                        help="dataset duplicate for number of network")
    
    args = parser.parse_args()

    predict(args)


if __name__ == "__main__":
    main()

