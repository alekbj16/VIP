from sklearn.cluster import KMeans
import numpy as np
import cv2
from matplotlib import pyplot as plt

k = 5
image_path = "./Images_Data/coins.png"

img_data = cv2.imread(image_path,0)

flat_img_data = img_data.reshape((-1,1))


kmeans = KMeans(n_clusters=k, random_state=0, max_iter=300)
kmeans.fit(flat_img_data)

centroids = kmeans.cluster_centers_
segmented = kmeans.predict(flat_img_data)

for i in range(segmented.shape[0]):
    segmented[i] = centroids[segmented[i]]

segmented = segmented.reshape(img_data.shape)

cv2.imwrite("bob.png", segmented)
#plt.imshow(segmented)
