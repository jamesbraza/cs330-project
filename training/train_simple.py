import logging
import sys
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from models.core import TransferModel
from training.utils import shuffle_and_split


def lr_strategy(epoch):
    lr_start = 0.0001

    lr_max = 0.001
    lr_min = 0.00005
    lr_rampup_epochs = 10
    lr_sustain_epochs = 5
    lr_exp_decay = 0.8
    if epoch < lr_rampup_epochs:
        lr = lr_start + (lr_max - lr_min) / lr_rampup_epochs * epoch
    elif epoch < lr_rampup_epochs + lr_sustain_epochs:
        lr = lr_max
    else:
        lr = lr_min + (lr_max - lr_min) * lr_exp_decay ** (
            epoch - lr_sustain_epochs - lr_rampup_epochs
        )
    return lr


def train_model(args):
    learning_rate = args.learning_rate
    max_epoch = args.maxEpoch
    batch_size = args.batch_size
    X_path = args.x_path
    y_path = args.y_path
    log_dir = args.log_dir

    X = np.load(X_path)
    Y = np.load(y_path)

    # =====load data and shuffle
    X_train, Y_train, X_val, Y_val = shuffle_and_split(X, Y, training_split=0.7)

    model = TransferModel()

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt, loss="categorical_crossentropy", metrics=["categorical_accuracy"]
    )
    model.build(input_shape=(None, 28, 28, 3))
    model.summary()

    model_save_callback = tf.keras.callbacks.ModelCheckpoint(
        log_dir + "/weights/weights.{epoch:02d}", period=1, save_weights_only=False
    )
    train_log_callback = tf.keras.callbacks.CSVLogger("training.csv", separator=",")
    # ====add tensorboard log
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    if args.fine_tune is not None:
        model.load_weights(args.fine_tune)
        logging.info("[INFO] Starting from model {}".format(args.fine_tune))

    train_model = model.fit(
        X_train,
        Y_train,
        epochs=max_epoch,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[
            model_save_callback,
            train_log_callback,
            tensorboard_callback,
        ],
        verbose=1,
        shuffle=True,
    )


def main():
    parser = ArgumentParser(description="Train parameters ")
    parser.add_argument(
        "--fine_tune", type=str, default=None, help="fine-tuning weights"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Set learning rate"
    )
    parser.add_argument("--maxEpoch", type=int, default=30, help="Maximum epochs")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Set the batch size for training"
    )
    parser.add_argument(
        "--x_path", type=str, default="/data1/cs330/project/project/tl3_x.npy", help="x"
    )
    parser.add_argument(
        "--y_path", type=str, default="/data1/cs330/project/project/tl3_y.npy", help="y"
    )
    parser.add_argument(
        "--log_dir", type=str, default="/data1/cs330/project/train/model2", help="log"
    )
    args = parser.parse_args()
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    train_model(args)


if __name__ == "__main__":
    main()
