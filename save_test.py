import os
import pandas as pd
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('C:/Users/halwa/Downloads/haarcascade_frontalface_default.xml')


path = 'C:/Users/halwa/Downloads/Stimuli to Test_project ML/Stimuli to Test_project ML/morph.jpg'
path1 = "D:/Project Data_org"

image = cv2.imread(path,0)
img_res = cv2.resize(image, (128,128))
faces = face_cascade.detectMultiScale(img_res,scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
 # Draw rectangle around the faces
for (x, y, w, h) in faces:
    roi_color = img_res[y:y + h, x:x + w]
    cv2.imwrite(os.path.join(path1, 'test' + '.jpeg'), roi_color)

