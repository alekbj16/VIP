import cv2
from kmean import lloyd
from denoise import denoise
from otsu import otsu


if __name__ == "__main__":
    names = ["page", "rocksample", "coins", "camera"]
    typ = "png"
    for name in names:
        k = 2
        image_path = f"Images_Data/{name}.{typ}"
        img_data = cv2.imread(image_path, 0) 

        seg_otsu = otsu(img_data)

        seg_otsu_dn8 = denoise(seg_otsu,8,3)
        seg_otsu_dn4 = denoise(seg_otsu,4,3)

        seg_lloyd2 = lloyd(img_data, 2)
        seg_lloyd5 = lloyd(img_data, 5)

        seg_lloyd_dn2 = denoise(seg_lloyd2, 8, 3)
        seg_lloyd_dn5 = denoise(seg_lloyd5, 4, 3)


        cv2.imwrite(f"results/{name}_otsu.{typ}", seg_otsu)
        cv2.imwrite(f"results/{name}_otsu_dn8.{typ}", seg_otsu_dn8)
        cv2.imwrite(f"results/{name}_otsu_dn4.{typ}", seg_otsu_dn4)
        cv2.imwrite(f"results/{name}_lloyd2.{typ}", seg_lloyd2)
        cv2.imwrite(f"results/{name}_lloyd5.{typ}", seg_lloyd5)
        cv2.imwrite(f"results/{name}_lloyd_dn2.{typ}", seg_lloyd_dn2)
        cv2.imwrite(f"results/{name}_lloyd_dn5.{typ}", seg_lloyd_dn5)
        print(name+" done")

