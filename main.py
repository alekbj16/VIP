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
    retreive_imgs(1)