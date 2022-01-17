from email.mime import image
import cv2
from kmean import lloyd
from denoise import denoise
from otsu import otsu


if __name__ == "__main__":
    image_path = "Images_Data/camera.png"
    img_data = cv2.imread(image_path, 0) 
    seg_kmean = lloyd(img_data,5)
    seg_otsu = otsu(img_data)
    seg_kmean_dn = denoise(seg_kmean,8,3)
    seg_otsu_dn = denoise(seg_otsu,8,3)

    cv2.imwrite("camera_lloyd.png", seg_kmean)
    cv2.imwrite("camera_lloyd_dn.png", seg_kmean_dn)
    cv2.imwrite("camera_otsu.png", seg_otsu)
    cv2.imwrite("camera_otsu_dn.png", seg_otsu_dn)

