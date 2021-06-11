# Face-Recognition-Using-OpenCV
Face detection and recognition is technology which is used to identify a person from a image or video.

# OpenCV
OpenCV is an open source computer vision and machine learning software library. It contains more than 2500 optimized algorithms of computer vision and machine learning. It has Python, C++, Java and MATLAB programming interfaces and supports Windows, Linux, Android and Mac OS. This library comes with FaceRecognizer class which can be used for face recognition.

# Face Detection
For Face Detection, Haar cascade classifier is used. A classifier is an algorithm that decides whether a given image is positive (face) or negative (not a face). Haar cascade classifier provided by OpenCV is pre-trained classifier, trained on thousands of images with or without faces. This classifier has high detection accuracy and features used by detector are computer very quickly.

# Face Recognition
LBPH(Local Binary Pattern Histogram) is one of the popular face recognition algorithms. It is a feature based approach meaning it processes input image to identify and extract distinctive features. It is possible to get good results with this recognizer but in a controlled environment. This algorithm take use of LBP(Local Binary Pattern) and HOG(Histograms of Oriented Gradients) for improving the detection performance on some datasets.

# Implementation
## Software Requirement
- Python
- OpenCV package
- Numpy package
- Tkinter package
- Pillow package

## Dataset
- Dataset contains two directories Training and Validation.
- Training -> 4 Celebrities with 10 images each
- Validation -> 4 Celebrities with 4 images each

## Files
### Face_Training.py
For each image in Training folder, model detects face and is trained to get features of face and save it to face_trained.yml file.

### Face_Recognition.py
For each image in Validation, model predicts face present in that image and stores outpur in Result folder. 
A single image can also be used for face recognition through the help of Select_File.py which uses tkinter package.

### Resize_Dataset.py
This can be used to resize whole dataset to a given dimension.

### Result.py
This is used to store outpur after performing face recognition into the Result folder.

### Select_File.py
This is used to select a image using select dialog box of tkinter.

### haarcascade_frontalface_default.xml
This is a trained haar cascade classifier of faces provided by OpenCV.

### face_trained.yml
This stores features and labels of faces which is obtained after training the model.
