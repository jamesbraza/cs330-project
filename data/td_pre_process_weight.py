import os

import numpy as np
import tensorflow as tf

from models.core import TransferModel

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
        base_model.build(input_shape=(1, 28, 28, 3))
        # base_model.summary()

        # ======just use last conv layer
        for layer in base_model.layers[2:3]:
            # print(layer.name)
            weights = layer.get_weights()[0]
            weigth_np = np.array(weights, dtype=object)
            # print(weigth_np.shape)
            weigth_np_b = np.swapaxes(weigth_np, 3, 0)
            # print(weigth_np_b.shape)
            weigth_np_r = np.reshape(
                weigth_np_b,
                (
                    weigth_np_b.shape[0],
                    weigth_np_b.shape[1] * weigth_np_b.shape[2] * weigth_np_b.shape[3],
                ),
            )
            # print(weigth_np_r.shape)
            # weigth_np_flat=weigth_np.flatten()
            # print(weigth_np_flat.shape)

        weigth_matrix.append(weigth_np_r)
    weigth_matrix_np = np.array(weigth_matrix, dtype=object).astype("float32")
    print(">>>>>final shape", weigth_matrix_np.shape)
    return weigth_matrix_np


if __name__ == "__main__":
    # path="/data1/cs330/project/weight_matrix"
    # folder_name=os.listdir(path)
    # dir_list=[os.path.join(path,f) for f in folder_name]
    # dir_list=['/data1/cs330/project/weight_matrix/model2', '/data1/cs330/project/weight_matrix/model3', '/data1/cs330/project/weight_matrix/model4', '/data1/cs330/project/weight_matrix/model5', '/data1/cs330/project/weight_matrix/model6', '/data1/cs330/project/weight_matrix/model7', '/data1/cs330/project/weight_matrix/model8']
    dir_list = ["/data1/cs330/project/weight_matrix/model8"]
    path = "/data1/cs330/project/data/x_feature_test"
    weigth_matrix_np = get_weight_matrix_input(dir_list)
    np.save(os.path.join(path, "weight_training_matrix_model8.npy"), weigth_matrix_np)
