import os

import numpy as np
from sklearn.decomposition import PCA


def pca_dataset(data_path, X_shape: int = 7):
    data_matrix = np.load(data_path)

    # ====fill in nan
    batch = data_matrix.shape[0]
    x1 = data_matrix.reshape(batch, 28 * 28 * 3)

    pca = PCA(n_components=256, svd_solver="full")
    pca.fit(x1)
    x_reduce = pca.transform(x1)
    x_average = np.average(x_reduce, axis=0)
    print(x_average.shape)
    x_average_r = x_average[np.newaxis, :]
    print(x_average_r.shape)
    all_data_np = np.repeat(x_average_r, repeats=X_shape, axis=0)
    print(">>>>>your reduced feature input", all_data_np.shape)
    return all_data_np


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    all_data_np = pca_dataset(data_path=os.path.join(dirname, "X_test.npy"), X_shape=2)
    np.save(os.path.join(dirname, "x_test_new.npy"), all_data_np)
