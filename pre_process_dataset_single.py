import glob
import numpy as np
import os
import cv2
from sklearn import preprocessing
from argparse import ArgumentParser, SUPPRESS
from sklearn.decomposition import PCA
import glob


def pca_dataset(data_path,X_shape=7):

    all_data_list=[]

    data_matrix=np.load(data_path)

    #====fill in nan
    print(data_matrix.shape)
    batch=data_matrix.shape[0]
    x1=data_matrix.reshape(batch,28*28*3)

    pca=PCA(n_components=256, svd_solver='full')
    pca.fit(x1)
    x_reduce=pca.transform(x1)
    x_average=np.average(x_reduce,axis=0)
    print(x_average.shape)
    x_average_r=x_average[np.newaxis,:]
    print(x_average_r.shape)
    all_data_np=np.repeat(x_average_r,repeats=X_shape,axis=0)

    print(">>>>>your reduced feature input",all_data_np.shape)
    return all_data_np

if __name__ == "__main__":
    path_dir="/data1/cs330/project/data/code/X_test.npy"
    all_data_np=pca_dataset(path_dir,X_shape=2)
    np.save("/data1/cs330/project/data/x_feature_test/x_test_new.npy",all_data_np)