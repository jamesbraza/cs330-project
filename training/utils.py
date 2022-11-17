import numpy as np
import tensorflow as tf


def shuffle_and_split(X, Y, training_split: float = 0.7):
    max_len = Y.shape[0]
    r = np.random.permutation(max_len)
    X_shuffle = X[r, :]
    Y_shuffle = Y[r]
    Y_enc = tf.keras.utils.to_categorical(Y_shuffle)

    training_num = round(training_split * max_len)
    num_validation = round(max_len - training_num)

    X_train = X_shuffle[:training_num]
    Y_train = Y_enc[:training_num]

    X_val = X_shuffle[training_num : training_num + num_validation]
    Y_val = Y_enc[training_num : training_num + num_validation]
    return X_train, Y_train, X_val, Y_val
