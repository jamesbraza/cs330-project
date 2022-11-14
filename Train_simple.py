import random
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import tensorflow_addons as tfa
import tensorflow as tf
import os
import sys
import shared.param_p as param
import model as model_path


def train_test_split(X,Y,num_training=2500,num_validation=226):
    num_training=num_training
    num_validation=num_validation

    X_train = X[:num_training]
    Y_train=Y1[:num_training]

    X_val = X[num_training:num_training+num_validation]
    Y_val=Y1[num_training:num_training+num_validation]

    return X_train,Y_train,X_val,Y_val


def custom_loss(y_true, y_pred):
    diff = K.sum(K.square(y_pred - y_true),axis=0)
    bottom=K.sum(K.square(y_true),axis=0)
    pert= diff/bottom
    return K.sum(pert)

def train_model(args):
    learning_rate = args.learning_rate 
    max_epoch = args.maxEpoch 
    batch_size= param.batch_size
    X_path=args.x_path
    y_path=args.y_path

    X=np.load('/content/drive/My Drive/plant/X_features.npy')
    Y=np.load('/content/drive/My Drive/plant/X_features_test.npy')


    #=====load data and shuffle
    X_train,Y_train,X_val,Y_val=train_test_split(X,Y,num_training=2500,num_validation=226)
    
    #====for multi-gpu
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model=model_path.model()

        model.compile(optimizer='adam',loss=custom_loss)
        model.summary()

        model_save_callback = tf.keras.callbacks.ModelCheckpoint(".{epoch:02d}", period=1, save_weights_only=False)
        model_best_callback = tf.keras.callbacks.ModelCheckpoint("best_val_loss", monitor='val_loss', save_best_only=True, mode="min")
        train_log_callback = tf.keras.callbacks.CSVLogger("training.csv", separator=',')

        #====add tensorboard log
        folder_name=ochk_prefix.split("/")[-2]
        log_dir="runs/" + folder_name 
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        output = model(np.array(table_dataset_list[0].root.position_matrix[:20]))

        if args.fine_tune is not None:
            model.load_weights(args.fine_tune)
            logging.info("[INFO] Starting from model {}".format(args.chkpnt_fn))

    train_model = model.fit(X_train,Y_train,
                              epochs=max_epoch
                              validation_data=(X_test,Y_val),
                              callbacks=[model_save_callback,
                                         model_best_callback,
                                         train_log_callback,
                                         tensorboard_callback],
                              verbose=1,
                              shuffle=False)
    


def main():
    parser = ArgumentParser(description="Train parameters ")

    parser.add_argument('--fine_tune', type=str, default=None,
                        help="fine-tuning weights")

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Set learning rate")

    parser.add_argument('--maxEpoch', type=int, default=None,
                        help="Maximum epochs")
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Set the batch size for training")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    train_model(args)


if __name__ == "__main__":
    main()
