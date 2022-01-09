### Initial work on assignment 4
import os 
import cv2
import numpy as np
import joblib # csv save library
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
    cats, imgs, f_names = retreive_imgs(2)
    #  imgs = [imgs[1]]
    #  cats = [cats[1]]
    #  f_names = [f_names[1]]
    sift = cv2.SIFT_create()

    train = [] # The individual descriptors for each image, will result in: [descriptors]
    test = []  # Same as above but for testing
    train_des = [] # One big list of each descriptor for each image by each category, will result in: category x [descriptors]
    test_des = [] # Same as above but for testing
    print("Computing descriptors for training and test")
    for i in range(len(imgs)):
        train_des.append([])
        test_des.append([])
        # Computing descriptors for training
        for j in range(len(imgs[i])//2):
            kp, des = sift.detectAndCompute(imgs[i][j], None)
            train_des[i].append(des)
            for d in des:
                train.append(d)
        # Computing descriptors for testing
        for j in range(len(imgs[i])//2, len(imgs[i])):
            kp, des = sift.detectAndCompute(imgs[i][j], None)
            test_des[i].append(des)
            for d in des:
                test.append(d)

    # The kmeans traning
    print("Running kmeans")
    train = np.array(train)
    test = np.array(test)
    kmeans = KMeans(n_clusters=5, random_state=0, max_iter=10)
    kmeans.fit(train)

    print("Computing word histograms for training")
    # Initialize the word histograms for training
    cluster_histo_train = []
    for cat in range(len(cats)):
        cluster_histo_train.append([])
        for img in range(len(train_des[cat])):
            cluster_histo_train[cat].append([0]*200)

    # Computing the word histograms
    for cat in range(len(cats)):
        for img in range(len(train_des[cat])):
            preds = kmeans.predict(train_des[cat][img])
            for cluster in preds:
                cluster_histo_train[cat][img][cluster] += 1

    print("Computing word histograms for testing")
    # Initialize the word histograms for testing
    cluster_histo_test = []
    for cat in range(len(cats)):
        cluster_histo_test.append([])
        for img in range(len(test_des[cat])):
            cluster_histo_test[cat].append([0]*200)

    # Computing the word histograms for testing
    for cat in range(len(cats)):
        for img in range(len(test_des[cat])):
            preds = kmeans.predict(test_des[cat][img])
            for cluster in preds:
                cluster_histo_test[cat][img][cluster] += 1

    # Saving the data
    data = []
    for cat in range(len(cats)):
        for img in range(len(train_des[cat])):
            for cluster in cluster_histo_train:
                data.append((cats[cat], f_names[cat][img], cluster))

    joblib.dump(data, "train.txt")
    lul = joblib.load("train.txt")
    print(lul[0])
