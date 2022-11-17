import glob
import numpy as np
import os.path
import cv2
from sklearn import preprocessing
from argparse import ArgumentParser, SUPPRESS
from sklearn.decomposition import PCA
import glob


def pca_dataset(path_dir):
    data_path=glob.glob(os.path.join(path_dir,"*.npy"))

    print(data_path)
    all_data_list=[]
    for i in data_path:
        data_path=i
        data_matrix=np.load(data_path)

        #====fill in nan
        batch=data_matrix.shape[0]
        x1=data_matrix.reshape(batch,28*28*3)

        pca=PCA(n_components=256, svd_solver='full')
        pca.fit(x1)
        x_reduce=pca.transform(x1)
        x_average=np.average(x_reduce,axis=0)

        all_data_list.append(x_average)
    all_data_np=np.array(all_data_list)
    np.save(path_dir,all_data_np)
    print(">>>>>your reduced feature input",all_data_np.shape)

if __name__ == "__main__":
    path_dir="/Users/annaning/Desktop/cs330/project/data/x_feature"
    pca_dataset(path_dir)