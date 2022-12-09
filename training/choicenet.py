import sys
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from embedding.td_pre_process_weight import get_weight_matrix_input
from models.core import ChoiceNetv1
from training import LOG_DIR


def train_model(args):
    learning_rate = args.learning_rate
    max_epoch = args.maxEpoch
    batch_size = args.batch_size
    fine_tune_weights_list = args.fine_tune_weights_list
    log_dir = args.log_dir

    # =======get weight matri THIS PART IS NOT FINAL !!!!!
    # TODO
    X_weight = get_weight_matrix_input(fine_tune_weights_list, learning_rate)
    X_train = X_weight[:2]
    X_val = X_weight[2:3]

    Y_weight = np.array([0.9270833134651184, 0.9320833086967468, 0.9287499785423279])
    Y_train = Y_weight[:2]
    Y_val = Y_weight[2:3]

    model = ChoiceNetv1()

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt, loss="mse", metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    model.build(input_shape=(None, 147456))
    model.summary()

    model_save_callback = tf.keras.callbacks.ModelCheckpoint(
        log_dir + "/weights/weights.{epoch:02d}", period=1, save_weights_only=False
    )
    train_log_callback = tf.keras.callbacks.CSVLogger(
        log_dir + "training.csv", separator=","
    )

    model.fit(
        X_train,
        Y_train,
        epochs=max_epoch,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[model_save_callback, train_log_callback],
        verbose=1,
        shuffle=True,
    )


def main():
    parser = ArgumentParser(description="Train parameters ")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Set learning rate"
    )
    parser.add_argument("--maxEpoch", type=int, default=30, help="Maximum epochs")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Set the batch size for training"
    )
    parser.add_argument(
        "--log_dir", type=str, default=LOG_DIR, help="log base directory"
    )
    parser.add_argument(
        "--fine_tune_weights_list",
        type=str,
        default="/data1/cs330/project/train/model2",
        help="weight list",
    )
    args = parser.parse_args()
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    train_model(args)


if __name__ == "__main__":
    main()
