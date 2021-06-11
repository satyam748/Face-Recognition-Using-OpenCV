import cv2 as cv
import numpy as np
import os
import Select_File
import Result
import tkinter as tk
from tkinter import filedialog

haar_cascade_face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

people = []
for i in os.listdir(r"Dataset\Training"):
    people.append(i)

face_recognizer = cv.face.LBPHFaceRecognizer_create(
    radius=2, neighbors=8, grid_x=8, grid_y=8
)
face_recognizer.read("face_trained.yml")

q = 1


def recognize(img_path):
    global q
    img = cv.imread(img_path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_rec = haar_cascade_face.detectMultiScale(gray_img, 1.3, 7)
    for (x, y, w, h) in face_rec:
        faces_region_of_interest = gray_img[x : x + w, y : y + h]
        label, confidence = face_recognizer.predict(faces_region_of_interest)
        print(f"Label = {people[label]} with a confidence of : {confidence}")
        if confidence > 100:
            cv.putText(
                img,
                str("Face not recognized!"),
                (40, 40),
                cv.FONT_HERSHEY_COMPLEX,
                1.0,
                (0, 255, 0),
                thickness=2,
            )
            cv.rectangle(img, (x, y), (x + w, y + h), (65, 185, 225), thickness=2)
            Result.store_result(img, q)
        else:
            cv.putText(
                img,
                str(people[label]),
                (40, 40),
                cv.FONT_HERSHEY_COMPLEX,
                1.0,
                (0, 255, 0),
                thickness=2,
            )
            cv.rectangle(img, (x, y), (x + w, y + h), (65, 185, 225), thickness=2)
            Result.store_result(img, q)
        q += 1
    return img


user_input = input(
    "Do you want to recognize single image, Press 'y' for Yes and 'n' for No\n"
)
if user_input == "y":
    img_path = Select_File.select_file()
    image = recognize(img_path)
    cv.imshow("Output", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

elif user_input == "n":
    DIR = r"Dataset\Validation"
    people = []
    for i in os.listdir(DIR):
        people.append(i)
    for person in people:
        path = os.path.join(DIR, person)
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            recognize(image_path)

else:
    print("Wrong key pressed")
