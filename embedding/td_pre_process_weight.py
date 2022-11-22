import os

import numpy as np
import tensorflow as tf

from embedding.embed import embed_model
from models.core import TransferModel


def get_weight_matrix_input(
    fine_tune_weights_list: list[str], learning_rate: float = 0.001
) -> np.ndarray:
    weigth_matrix = []
    for fine_tune_weights in fine_tune_weights_list:
        # ======model weight matrix embedding:
        base_model = TransferModel()
        checkpoint = tf.train.Checkpoint(base_model)
        checkpoint.restore(fine_tune_weights).expect_partial()

        base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )
        base_model.build(input_shape=(None, 28, 28, 3))

        weigth_np_r = embed_model(base_model)
        weigth_matrix.append(weigth_np_r)

    return np.array(weigth_matrix, dtype="float32")


if __name__ == "__main__":
    dir_list = ["/data1/cs330/project/weight_matrix/model8"]
    path = "/data1/cs330/project/data/x_feature_test"
    weight_matrix = get_weight_matrix_input(dir_list)
    np.save(os.path.join(path, "weight_training_matrix_model8.npy"), weight_matrix)
