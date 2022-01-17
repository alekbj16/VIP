from sklearn.cluster import KMeans
import numpy as np
import cv2
from matplotlib import pyplot as plt


def lloyd(img_data, k):
    flat_img_data = img_data.reshape((-1,1))

    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=300)
    kmeans.fit(flat_img_data)

    centroids = kmeans.cluster_centers_
    segmented = kmeans.predict(flat_img_data)

    for i in range(segmented.shape[0]):
        segmented[i] = centroids[segmented[i]]

    return segmented.reshape(img_data.shape)
