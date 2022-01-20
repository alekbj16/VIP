import cv2
from kmean import lloyd
from denoise import denoise
from otsu import otsu


if __name__ == "__main__":
    # names = ["camera","Checkerboard", "page", "rocksample","smka0b", "square"]
    names = ["Checkerboard", "page", "rocksample","smka0b", "square"]
    for name in names:
        typ = "png"
        k = 2
        image_path = f"Images_Data/{name}.{typ}"
        img_data = cv2.imread(image_path, 0) 
        # seg_kmean = lloyd(img_data,k)
        seg_otsu = otsu(img_data)
        # seg_kmean_dn = denoise(seg_kmean,8,3)
        seg_otsu_dn = denoise(seg_otsu,8,3)

        # cv2.imwrite(f"results/{name}_lloyd_k{k}.{typ}", seg_kmean)
        # cv2.imwrite(f"results/{name}_lloyd_k{k}_dn.{typ}", seg_kmean_dn)
        cv2.imwrite(f"results/{name}_otsu.{typ}", seg_otsu)
        cv2.imwrite(f"results/{name}_otsu_dn.{typ}", seg_otsu_dn)
        print(name+" done")

