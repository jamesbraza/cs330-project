import random
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import tensorflow as tf
import os
import sys
import model as model_path



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
            weigth_np_flat=weigth_np.flatten()
            #print(weigth_np_flat.shape)
        
        weigth_matrix.append(weigth_np_flat)
    weigth_matrix_np=np.array(weigth_matrix,dtype=object).astype('float32')
    # print(weigth_matrix_np.shape)
    return weigth_matrix_np


def train_model(args):
    learning_rate = args.learning_rate 
    max_epoch = args.maxEpoch 
    batch_size= args.batch_size
    weight_matrix_path=args.weight_matrix_path
    y_matrix=args.y_matrix
    log_dir=args.log_dir

    X_weight=np.load(weight_matrix_path)
    Y_matrix=np.load(y_matrix)

    X_weight=np.array(X_weight).astype(np.float32)
    Y_matrix=np.array(Y_matrix).astype(np.float32)

    print(X_weight.shape)
    print(Y_matrix.shape)

    X_train=X_weight[0:-1]
    Y_train=Y_matrix[0:-1]

    X_val=X_weight[-1:]
    Y_val=Y_matrix[-1:]

    model=model_path.Model_ChoiceNet_emb()

    opt=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,loss="mse",metrics=[tf.keras.metrics.MeanSquaredError()])
    model.build(input_shape=(None,18,8193))
    #model = Model(inputs=[base_model.input,base_model2.input], outputs=output)
    model.summary()

    model_save_callback = tf.keras.callbacks.ModelCheckpoint(log_dir +"/weights/", save_best_only=True,save_weights_only=False)
    train_log_callback = tf.keras.callbacks.CSVLogger(log_dir+"training.csv", separator=',')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    train_model = model.fit(X_train,Y_train,
                              epochs=max_epoch,
                              batch_size=1,
                              validation_data=(X_val,Y_val),
                              callbacks=[model_save_callback,
                                         train_log_callback,
                                         ],
                              verbose=1,
                              shuffle=True)

def main():
    parser = ArgumentParser(description="Train parameters ")

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Set learning rate")

    parser.add_argument('--maxEpoch', type=int, default=30,
                        help="Maximum epochs")
    
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Set the batch size for training")

    parser.add_argument('--weight_matrix_path', type=str, default="/data1/cs330/project/res_net_embedding/emb_resnet_x_all.npy",
                        help="path for the reducd weight matrix")
    
    parser.add_argument('--y_matrix', type=str, default="/data1/cs330/project/data/x_feature/y_train.npy",
                        help="path for the accuracy from fine-tuning val")

    
    parser.add_argument('--log_dir', type=str, default="/data1/cs330/project/test_tdl_weight_only",
                        help="log")

    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()

