import random
from sklearn.cluster import KMeans
import numpy as np
import cv2
from matplotlib import pyplot as plt


class OurKMeans:
    # We assume this will only be used
    # for 1-dimensinel data
    def __init__(self, n_clusters=2, random_state=0, max_iter=300):
        self.k = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None
    
    def fit(self, data):
        np.random.seed(self.random_state)
        min_val = np.min(data)
        max_val = np.max(data)

        centroids = np.random.randint(min_val,max_val, size=self.k)        
        for b in range(self.max_iter):
            dists = []
            for i in range(self.k):
                dists.append(np.abs(data-centroids[i]))
            dists = np.array(dists)
            labels = np.argmin(dists, axis=0)
            assert(labels.shape==data.shape)

            cen_dists = np.zeros(centroids.shape)
            for i in range(data.shape[0]):
                cen_dists[labels[i]] += data[i]
            
            for i in range(self.k):
                n = (np.count_nonzero(labels == i))
                if n > 0:
                    cen_dists[i] = cen_dists[i]/(np.count_nonzero(labels == i))
                # print(cen_dists[i])
                # print((np.count_nonzero(labels == i)))
            
            if np.count_nonzero(abs(centroids - cen_dists) < 0.1) == self.k:
                centroids = cen_dists
                break
            centroids = cen_dists
        
        self.cluster_centers_ = centroids
        self.labels_ = labels
        return 
    
    def predict(self, data):
        dists = []
        for i in range(self.k):
            dists.append(np.abs(data-self.cluster_centers_[i]))
        dists = np.array(dists)
        labels = np.argmin(dists, axis=0)
        return labels

        
        
        # for i in range(data.shape[0]):
        #     for j in range(centroids.shape[0]):
        #         dist = np.abs(data[i]-centroids[j])


def lloyd(img_data, k):
    flat_img_data = img_data.reshape((-1,1))

    # kmeans = KMeans(n_clusters=k, random_state=0, max_iter=300)
    # kmeans.fit(flat_img_data)
    kmeans = OurKMeans(n_clusters=k, random_state=0, max_iter=300)
    kmeans.fit(flat_img_data)

    centroids = kmeans.cluster_centers_
    segmented = kmeans.predict(flat_img_data)

    for i in range(segmented.shape[0]):
        segmented[i] = centroids[segmented[i]]

    return segmented.reshape(img_data.shape)


def lloyd_orig(img_data, k):
    flat_img_data = img_data.reshape((-1,1))

    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=300)
    kmeans.fit(flat_img_data)

    centroids = kmeans.cluster_centers_
    segmented = kmeans.predict(flat_img_data)

    for i in range(segmented.shape[0]):
        segmented[i] = centroids[segmented[i]]

    return segmented.reshape(img_data.shape)


def test():
    k=2
    image_path = "Images_Data/camera.png"
    img_data = cv2.imread(image_path, 0) 
    flat_img_data = img_data.reshape((-1,1))
    ourkmeans = OurKMeans(n_clusters=k, random_state=0, max_iter=50)
    ourkmeans.fit(flat_img_data)
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=50)
    kmeans.fit(flat_img_data)

    ls = kmeans.predict(flat_img_data)
    ourls = ourkmeans.predict(flat_img_data)
    if (ls.reshape((-1,1)) == ourls).all():
        print("yaa")
    else:
        print(ourls)
        print(ls)

if __name__ == "__main__":
    test()



