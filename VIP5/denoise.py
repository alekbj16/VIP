import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.stats import mode

# for z in range(k):
#     for i in range(1,img_data.shape[0]-1):
#         for j in range(1, img_data.shape[1]-1):
#             arr = [img_data[i,j-1],img_data[i-1,j],img_data[i+1,j],img_data[i,j+1]]
#             label, count = mode(arr)
#             if count[0] >= 3:
#                 new_img_data[i,j] = label[0]
#     img_data = np.copy(new_img_data)

def denoise(ida, n_neighbours, n_iter, t=3):
    def get_arr(dat, i, j):
        return [dat[i,j-1],dat[i-1,j],dat[i+1,j],dat[i,j+1],dat[i+1,j+1],dat[i-1,j+1],dat[i-1,j-1],dat[i+1,j-1]][:n_neighbours]

    img_data = np.copy(ida)
    new_img_data = np.copy(img_data)

    for z in range(n_iter):
        for i in range(1,img_data.shape[0]-1):
            for j in range(1, img_data.shape[1]-1):
                arr = get_arr(img_data,i,j)
                label, count = mode(arr)
                if count[0] >= t:
                    new_img_data[i,j] = label[0]
        img_data = np.copy(new_img_data)
    return img_data
