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
    fine_tune_weights =args.fine_tune_weights
    choice_net_weights=args.choice_net_weights

    X_test=get_weight_matrix_input_predict(fine_tune_weights)

    #load model
    model=model_path.Model_ChoiceNet_simple()
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(choice_net_weights).expect_partial()

    pred=model.predict(X_test,verbose=0)
    print(">>>>your prediction",pred)


def main():
    parser = ArgumentParser(description="Train parameters ")

    parser.add_argument('--fine_tune_weights', type=str, default="/data1/cs330/project/weight_matrix/model1",
                        help="path for the reducd weight matrix")
    
    parser.add_argument('--choice_net_weights', type=str, default="/data1/cs330/project/tdl_model/weight_net/weights",
                        help="weights for choicenet")
    
    args = parser.parse_args()

    predict(args)


if __name__ == "__main__":
    main()

