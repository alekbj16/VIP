from skimage.filters import threshold_otsu

import numpy as np
import cv2
from matplotlib import pyplot as plt

def otsu(img_data):
    t = threshold_otsu(img_data)

    segmented = np.zeros(img_data.shape)
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            if img_data[i,j] >= t:
                segmented[i,j] = 255
    return segmented