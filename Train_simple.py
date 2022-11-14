import random
import numpy as np
from argparse import ArgumentParser, SUPPRESS
import tensorflow as tf
import os
import sys
import model as model_path


def suffle_and_split(X,Y,training_num=0.7):
    max_len=Y.shape[0]
    r=np.random.permutation(max_len)
    X_shuffle=X[r,:]
    Y_shuffle=Y[r]
    Y_enc=tf.keras.utils.to_categorical(Y_shuffle)

    training_num=round(training_num*max_len)
    num_validation=round(max_len-training_num)

    X_train = X_shuffle[:training_num]
    Y_train=Y_enc[:training_num]

    X_val = X_shuffle[training_num:training_num+num_validation]
    Y_val=Y_enc[training_num:training_num+num_validation]
    return X_train,Y_train,X_val,Y_val


def custom_loss(y_true, y_pred):
    diff = K.sum(K.square(y_pred - y_true),axis=0)
    bottom=K.sum(K.square(y_true),axis=0)
    pert= diff/bottom
    return K.sum(pert)

def train_model(args):
    learning_rate = args.learning_rate 
    max_epoch = args.maxEpoch 
    batch_size= args.batch_size
    X_path=args.x_path
    y_path=args.y_path
    log_dir=args.log_dir

    X=np.load(X_path)
    Y=np.load(y_path)


    #=====load data and shuffle
    X_train,Y_train,X_val,Y_val=suffle_and_split(X,Y,training_num=0.7)
    
    #====for multi-gpu
    #strategy = tf.distribute.MirroredStrategy()
    #print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    #with strategy.scope():
    model=model_path.Model_transfer()

    opt=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["categorical_accuracy"])
    model.build(input_shape=(None,28,28,3))
    model.summary()

    model_save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir,period=1, save_weights_only=False)
    model_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir, monitor='val_loss', save_best_only=True, mode="min")
    train_log_callback = tf.keras.callbacks.CSVLogger("training.csv", separator=',')

    #====add tensorboard log
    # folder_name=log_dir.split("/")[-2]
    # log_dir_board="runs/" + folder_name 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if args.fine_tune is not None:
        model.load_weights(args.fine_tune)
        logging.info("[INFO] Starting from model {}".format(args.fine_tune))

    train_model = model.fit(X_train,Y_train,
                              epochs=max_epoch,
                              batch_size=batch_size,
                              validation_data=(X_val,Y_val),
                              callbacks=[model_save_callback,
                                         model_best_callback,
                                         train_log_callback,
                                         tensorboard_callback],
                              verbose=1,
                              shuffle=True)
    


def main():
    parser = ArgumentParser(description="Train parameters ")

    parser.add_argument('--fine_tune', type=str, default=None,
                        help="fine-tuning weights")

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Set learning rate")

    parser.add_argument('--maxEpoch', type=int, default=30,
                        help="Maximum epochs")
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Set the batch size for training")
    
    parser.add_argument('--x_path', type=str, default="/data1/cs330/project/project/tl3_x.npy",
                        help="x")

    parser.add_argument('--y_path', type=str, default="/data1/cs330/project/project/tl3_y.npy",
                        help="y")
    
    parser.add_argument('--log_dir', type=str, default="/data1/cs330/project/train/model2",
                        help="log")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    train_model(args)


if __name__ == "__main__":
    main()
