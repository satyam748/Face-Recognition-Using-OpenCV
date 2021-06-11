import cv2 as cv
import sys
import os

if not os.path.exists('Result'):
    os.makedirs('Result')

DIR = "Result"

def store_result(image, image_number):
    file_ext = ".jpg"
    image_name = str(image_number) + file_ext
    image_path = os.path.join(DIR,image_name)
    cv.imwrite(image_path, image)

