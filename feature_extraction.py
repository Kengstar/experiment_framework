import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fftpack
from sklearn.cluster import DBSCAN
from sklearn.cluster.k_means_ import KMeans
from sklearn.decomposition import PCA
import time
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
from skimage.filters.rank import median



'''this module should contain the used feature transformations/ extraction/ selection'''



def get_dct_features_2dinput(dataset):
    dct_features = np.concatenate\
        ([fftpack.dct(fftpack.dct(im, axis=0), axis=1) for im in dataset]).reshape(-1,75*75)
    return dct_features


def transform_linear_scale(dataset):
    dataset = 2 ** (dataset/6)
    dataset =( dataset - np.amin(dataset)) /  (np.amax(dataset) -np.amin(dataset))
    return dataset


def get_dct_features_ulc(dataset,n):
    """"n: selected coefficients"""
    dct_features = np.concatenate\
        ([fftpack.dct(fftpack.dct(im, axis=0), axis=1)[:n,:n] for im in dataset])
    dct_features = dct_features.reshape(-1,n*n)
    return dct_features




def calc_PCA(dataset,percentage):
    pca = PCA(n_components=percentage)
    pca.fit(dataset)
    print(pca.components_.shape)
    return pca


def get_PCA_features (pca_estimator,dataset):
    return pca_estimator.transform(dataset)





def preprocessing_binary_not_finished_smarter(dataset1,dataset2):
    clusterer = DBSCAN(eps=1.5, min_samples=50, n_jobs= -1)
    ll = []
    a = np.arange(0, 75, dtype=np.int8)
    b = np.arange(0, 75, dtype=np.int8)
    ii, jj = np.meshgrid(a, b, indexing="ij")
    ii = ii.reshape((75 * 75))
    jj = jj.reshape((75 * 75))
    temp = np.zeros((75 * 75, 3), dtype=np.float64)
    temp[:, 0] = ii  # ii.reshape((75 * 75))
    temp[:, 1] = jj  # jj.reshape((75 * 75))
    temp2 = temp
    for idx in range(dataset1.shape[0]):
        start_time = time.time()
        temp[:, 2] = dataset1[idx].reshape((75 * 75))
        temp2[:, 2] = dataset2[idx].reshape((75 * 75))
        # normalize
        x = temp[:, 0]
        y = temp[:, 1]
        c = temp[:, 2]
        x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))
        img = np.column_stack(((x, y, c)))
        start_time = time.time()
        clusterer.fit(img)
        print("--- %s seconds ---" % (time.time() - start_time))



        x = temp2[:, 0]
        y = temp2[:, 1]
        c = temp2[:, 2]
        x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        y = (y - np.amin(y)) / (np.amax(y) - np.amin(y))
        img = np.column_stack(((x, y, c)))
        clusterer.fit(img)

        labels = clusterer.labels_
        cluster_labels = np.unique(labels)
        cluster_labels = cluster_labels[cluster_labels != -1]
        # Number of clusters in labels, ignoring noise if present.


        new_img = np.ones((75, 75))
        for cluster_label in cluster_labels:
            idx_of_relevant_clusters = labels == cluster_label
            choosen = img[idx_of_relevant_clusters]
            x = choosen[:, 0] * (np.amax(temp[:, 0]) - np.amin(temp[:, 0])) + np.amin(temp[:, 0])
            y = choosen[:, 1] * (np.amax(temp[:, 1]) - np.amin(temp[:, 1])) + np.amin(temp[:, 1])
            x = np.round(x).astype(int)
            y = np.round(y).astype(int)
            new_img[x, y] = 0
            # for xi, yi, in zip(x,y) :
            #for xi, yi in zip(x, y):
            #    new_img[int(np.round(xi)), int(np.round(yi))] = 0  ###carfeul very hacky
        ll.append(new_img)
        print(len(ll),"/", dataset1.shape[0])
       # print("--- %s seconds ---" % (time.time() - start_time))
    ll = np.array(ll)
    return ll
