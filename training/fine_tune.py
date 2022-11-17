import sys
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from models.core import TransferModel
from training.utils import shuffle_and_split


def train_model(args):
    learning_rate = args.learning_rate
    max_epoch = args.maxEpoch
    batch_size = args.batch_size
    fine_tune_weights = args.fine_tune_weights
    X_path = args.x_path
    y_path = args.y_path
    log_dir = args.log_dir

    X = np.load(X_path)
    Y = np.load(y_path)

    # =====load fine-tune dataset and shuffle
    X_train, Y_train, X_val, Y_val = shuffle_and_split(X, Y, training_split=0.7)

    # ====load weights
    base_model = TransferModel()
    checkpoint = tf.train.Checkpoint(base_model)
    checkpoint.restore(fine_tune_weights).expect_partial()

    # =====define new model
    new_model = tf.keras.Sequential(
        [*base_model.layers[:-1], tf.keras.layers.Dense(units=8, activation="softmax")]
    )

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    new_model.compile(
        optimizer=opt, loss="categorical_crossentropy", metrics=["categorical_accuracy"]
    )
    new_model.build(input_shape=(None, 28, 28, 3))
    new_model.summary()

    # ====freeze first 2 conv and traineable last
    for layer in new_model.layers[:2]:
        layer.trainable = False

    for layer in new_model.layers:
        print(layer.name, layer.trainable)

    model_save_callback = tf.keras.callbacks.ModelCheckpoint(
        log_dir + "/weights/weights.{epoch:02d}", period=1, save_weights_only=False
    )
    train_log_callback = tf.keras.callbacks.CSVLogger(
        log_dir + "training.csv", separator=","
    )
    # ====add tensorboard log
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    train_model = new_model.fit(
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
    parser.add_argument("--maxEpoch", type=int, default=15, help="Maximum epochs")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Set the batch size for training"
    )
    parser.add_argument(
        "--x_path",
        type=str,
        default="/data1/cs330/project/project/fine_tune_x.npy",
        help="x",
    )
    parser.add_argument(
        "--y_path",
        type=str,
        default="/data1/cs330/project/project/fine_tune_y.npy",
        help="y",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/data1/cs330/project/fine_tune/model1_tune",
        help="log",
    )
    parser.add_argument(
        "--fine_tune_weights",
        type=str,
        default="/data1/cs330/project/train/model1",
        help="weight",
    )
    args = parser.parse_args()
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    train_model(args)


if __name__ == "__main__":
    main()
