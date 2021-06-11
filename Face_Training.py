import cv2 as cv
import numpy as np
import os
from PIL import Image

haar_cascade_face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
DIR = r"Dataset\Training"

people = []
for i in os.listdir(DIR):
    people.append(i)

features = []
labels = []


def get_feature_and_label():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            image_array = cv.imread(image_path)
            if image_array is None:
                continue

            gray_image = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
            face_rec = haar_cascade_face.detectMultiScale(
                gray_image, scaleFactor=1.5, minNeighbors=5
            )
            if face_rec is None:
                continue

            for (x, y, w, h) in face_rec:
                faces_region_of_interest = gray_image[y : y + h, x : x + w]
                features.append(faces_region_of_interest)
                labels.append(label)


get_feature_and_label()

labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create(
    radius=2, neighbors=8, grid_x=8, grid_y=8
)
face_recognizer.train(features, labels)
face_recognizer.write("face_trained.yml")
print("Training Complete!")
