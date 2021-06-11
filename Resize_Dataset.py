import cv2 as cv
import os

"""
Directory Structure

|-----------Folder
|             |-------Subfolders
|                           |
|                           S1
|                           |---- 1.jpg
|                           |---- ...
|                           |---- 10.jpg
|                           S2
|                           |---- 1.jpg
|                           |---- ...
|                           |---- 10.jpg 

 """


def resize(image, window_width=600):
    # aspect_ratio = float(image.shape[1])/float(image.shape[0])
    # window_height = window_width/aspect_ratio
    image = cv.resize(image, (int(window_width), int(600)))
    return image


os.mkdir("Resized_Dataset")

DIR = r"Dataset\Training"
subfolders = os.listdir(DIR)

for subfolder in subfolders:
    subfolder_path = os.path.join(DIR, subfolder)
    new_subfolder_path = "Resized_Dataset/%s" % subfolder
    os.mkdir(new_subfolder_path)
    files = os.listdir(subfolder_path)
    new_image_name = 1
    file_ext = ".jpg"

    for file in files:
        new_image_name_tostring = str(new_image_name)
        new_image_finalname = new_image_name_tostring + file_ext

        file_path = os.path.join(subfolder_path, file)
        img = cv.imread(file_path)
        if img is None:
            continue
        resized_img = resize(img)
        new_img_path = os.path.join(new_subfolder_path, new_image_finalname)
        cv.imwrite(new_img_path, resized_img)
        new_image_name += 1

print("Resizing Done!")
