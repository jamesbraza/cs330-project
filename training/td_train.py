import sys
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from models.core import ChoiceNetSimple, TransferModel

learning_rate = 0.001


def get_weight_matrix_input(fine_tune_weights_list):
    weigth_matrix = []
    for fine_tune_weights in fine_tune_weights_list:
        print(fine_tune_weights)

        # ======model weight matrxi embedding:
        base_model = TransferModel()
        checkpoint = tf.train.Checkpoint(base_model)
        checkpoint.restore(fine_tune_weights).expect_partial()

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        base_model.compile(
            optimizer=opt,
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )
        base_model.build(input_shape=(None, 28, 28, 3))

        # ======just use last conv layer
        for layer in base_model.layers[2:3]:
            weights = layer.get_weights()[0]
            weigth_np = np.array(weights, dtype=object)
            weigth_np_flat = weigth_np.flatten()
            weigth_matrix.append(weigth_np_flat)
    weigth_matrix_np = np.array(weigth_matrix, dtype=object).astype("float32")

    return weigth_matrix_np


def train_model(args):
    learning_rate = args.learning_rate
    max_epoch = args.maxEpoch
    batch_size = args.batch_size
    fine_tune_weights_list = args.fine_tune_weights_list
    log_dir = args.log_dir

    # =======get weight matri THIS PART IS NOT FINAL !!!!!
    # TODO
    X_weight = get_weight_matrix_input(fine_tune_weights_list)
    X_train = X_weight[:2]
    X_val = X_weight[2:3]

    Y_weight = np.array([0.9270833134651184, 0.9320833086967468, 0.9287499785423279])
    Y_train = Y_weight[:2]
    Y_val = Y_weight[2:3]

    model = ChoiceNetSimple()

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt, loss="mse", metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    model.build(input_shape=(None, 147456))
    model.summary()

    model_save_callback = tf.keras.callbacks.ModelCheckpoint(
        log_dir + "/weights/" + "weights.{epoch:02d}", period=1, save_weights_only=False
    )
    train_log_callback = tf.keras.callbacks.CSVLogger(
        log_dir + "training.csv", separator=","
    )

    train_model = model.fit(
        X_train,
        Y_train,
        epochs=max_epoch,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[
            model_save_callback,
            train_log_callback,
        ],
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
        "--log_dir", type=str, default="/data1/cs330/project/train/model2", help="log"
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
