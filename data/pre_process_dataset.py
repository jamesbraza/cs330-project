import glob
import os.path

import numpy as np
from sklearn.decomposition import PCA


def pca_dataset(path_dir) -> None:
    data_path = glob.glob(os.path.join(path_dir, "*.npy"))

    print(data_path)
    all_data_list = []
    for i in data_path:
        data_path = i
        data_matrix = np.load(data_path)

        # ====fill in nan
        batch = data_matrix.shape[0]
        x1 = data_matrix.reshape(batch, 28 * 28 * 3)

        pca = PCA(n_components=256, svd_solver="full")
        pca.fit(x1)
        x_reduce = pca.transform(x1)
        x_average = np.average(x_reduce, axis=0)

        all_data_list.append(x_average)
    all_data_np = np.array(all_data_list)
    np.save(path_dir, all_data_np)
    print(">>>>>your reduced feature input", all_data_np.shape)


if __name__ == "__main__":
    pca_dataset(path_dir="/Users/annaning/Desktop/cs330/project/data/x_feature")
