### Initial work on assignment 4
import os 
import cv2


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
    path_to_categories = cwd + "\\101_ObjectCategories" #Path to all categories
    list_of_categories = os.listdir(path_to_categories)
    for i in range(n_categories):
        path_to_category = path_to_categories + "\\" + str(list_of_categories[i]) #Path to a specific category
        images_in_category = os.listdir(path_to_category)
        categories.append(str(list_of_categories[i]))
        for img in images_in_category:
            img_path = str(path_to_category + "\\" + img)
            print(img_path)
            img = cv2.imread(img_path)
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imshow("image",grayscale)
            cv2.waitKey(5000)
            images.append(img)

    print(images)
if __name__ == "__main__":
    retreive_imgs(1)