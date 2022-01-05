### Initial work on assignment 4
import os 
import cv2
import numpy as np
from sklearn.cluster import KMeans


def retreive_imgs(n_categories):
    """
    Specify how many categories.
    Selects the n first categories in the folder "101_ObjectCategories"
    (Google backgrounds have been deleted)
    """
    categories = []
    images = []
    f_names = []

    cwd = os.getcwd()
    path_to_categories = os.path.join(cwd,"101_ObjectCategories") #Path to all categories
    print(path_to_categories)
    list_of_categories = os.listdir(path_to_categories)
    for i in range(n_categories):
        img_in_cat = []
        f_names_cat = []
        path_to_category = os.path.join(path_to_categories,list_of_categories[i])
        images_in_category = os.listdir(path_to_category)
        categories.append(str(list_of_categories[i]))
        for img in images_in_category:
            img_path = os.path.join(path_to_category,img)
            img = cv2.imread(img_path)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_in_cat.append(grayscale)
            f_names_cat.append(img)
        images.append(img_in_cat)
        f_names.append(f_names_cat)
    
    return categories, images, f_names

if __name__ == "__main__":
    cats, imgs, f_names = retreive_imgs(5)
    sift = cv2.SIFT_create()

    train = []
    test = []
    train_des = []
    test_des = []
    for i in range(len(imgs[1])//2):
        kp, des = sift.detectAndCompute(imgs[1][i], None)
        train_des.append(des)
        for d in des:
            train.append(d)
    for i in range(len(imgs[1])//2, len(imgs[1])):
        kp, des = sift.detectAndCompute(imgs[1][i], None)
        for d in des:
            test.append(d)

    train = np.array(train)
    test = np.array(test)
    kmeans = KMeans(n_clusters=200, random_state=0)
    kmeans.fit(train)

    train_pred = []
    for des in train_des:
        pred = kmeans.predict(des)
        train_pred.append(pred)
    
