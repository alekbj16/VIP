from email.mime import image
from multiprocessing import set_forkserver_preload
from sklearn.cluster import KMeans
import numpy as np
import cv2
from matplotlib import pyplot as plt

k = 5
image_path = "./Images_Data/coins.png"

img_data = cv2.imread(image_path,0)

flat_img_data = img_data.reshape((-1,1))

#print(flat_img_data)


kmeans = KMeans(n_clusters=k, random_state=0, max_iter=300)
kmeans.fit(flat_img_data)

centroids = kmeans.cluster_centers_
print(centroids)
segmented = kmeans.predict(flat_img_data)

print(segmented)
for i in range(segmented.shape[0]):
    segmented[i] = centroids[segmented[i]]

segmented = segmented.reshape(img_data.shape)
print(segmented)

cv2.imwrite("bob.png", segmented)
#plt.imshow(segmented)
