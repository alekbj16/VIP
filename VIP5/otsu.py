from skimage.filters import threshold_otsu

import numpy as np
import cv2
from matplotlib import pyplot as plt

def otsu_thres(data):
    data = data.reshape((-1,1))
    histogram = np.histogram(data, bins=range(257))[0]
    thres = 0
    maximum = 0
    ts = np.arange(0,256,1)
    # We do not look at t=0
    for t in range(1,256):
        w0 = np.sum(histogram[0:t])
        w1 = np.sum(histogram[t:256])
        mu0 = 0
        mu1 = 0
        if w0 != 0:
            mu0 = np.sum(ts[0:t]*histogram[0:t])/w0
        if w1 != 0:
            mu1 = np.sum(ts[t:256]*histogram[t:256])/w1
        val = w0*w1*((mu0-mu1)**2)
        if val > maximum:
            thres = t
            maximum = val
    return thres-1

def otsu(img_data):
    # t = threshold_otsu(img_data)
    t = otsu_thres(img_data)
    segmented = np.zeros(img_data.shape)
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            if img_data[i,j] >= t:
                segmented[i,j] = 255
    return segmented

def test():
    image_path = "Images_Data/page.png"
    img_data = cv2.imread(image_path, 0) 
    real = threshold_otsu(img_data)
    our = otsu_thres(img_data)
    print(f"real={real}, our={our}")

if __name__ == "__main__":
    test()