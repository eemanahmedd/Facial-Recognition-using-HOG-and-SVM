import glob
import cv2
import numpy as np
import pandas as pd
import os
face_cascade = cv2.CascadeClassifier('C:/Users/halwa/Downloads/haarcascade_frontalface_default.xml')
from PIL import Image


# variable to get images from folder
students_images ={
    "Abdullah":glob.glob("D:/Project Data_org/AbdullahJpeg//*.jpeg"),
    "Affan":glob.glob("D:/Project Data_org/AffanJpeg//*.jpeg"),
    "Ali":glob.glob("D:/Project Data_org/AliJpeg//*.jpeg"),
    "Aziz":glob.glob("D:/Project Data_org/AzizJpeg//*.jpeg"),
    "Basit":glob.glob("D:/Project Data_org/BasitJpeg//*.jpeg"),
    "Eesha":glob.glob("D:/Project Data_org/EeshaJpeg//*.jpeg"),
    "Eman":glob.glob("D:/Project Data_org/EmanJpeg//*.jpeg"),
    "Faraz":glob.glob("D:/Project Data_org/FarazJpeg//*.jpeg"),
    "Fasih":glob.glob("D:/Project Data_org/FasihJpeg//*.jpeg"),
    "Hala":glob.glob("D:/Project Data_org/HalaJpeg//*.jpeg"),
    "Hamra":glob.glob("D:/Project Data_org/HamraJpeg//*.jpeg"),
    "Hasan":glob.glob("D:/Project Data_org/HassanJpeg//*.jpeg"),
   "Hira":glob.glob("D:/Project Data_org/HiraJpeg//*.jpeg"),
     "Jamali":glob.glob("D:/Project Data_org/JamaliJpeg//*.jpeg"),
    "Jawwad":glob.glob("D:/Project Data_org/JawwadJpeg//*.jpeg"),
   "Laviza":glob.glob("D:/Project Data_org/LavizaJpeg//*.jpeg"),
     "Parshant":glob.glob("D:/Project Data_org/ParshantJpeg//*.jpeg"),
    "Rehmat":glob.glob("D:/Project Data_org/RehmatJpeg//*.jpeg"),
    "Shehriyar":glob.glob("D:/Project Data_org/ShehriyarJpeg//*.jpeg"),
   "Subhan":glob.glob("D:/Project Data_org/SubhanJpeg//*.jpeg"),
    "Wardah":glob.glob("D:/Project Data_org/WardaJpeg//*.jpeg")
    
}

# defining labels
students_labels = {
    "Abdullah":0,
    "Affan":1,
    "Ali":2,
    "Aziz":3,
    "Basit":4,
    "Eesha":5,
    "Eman":6,
    "Faraz":7,
     "Fasih":8,
     "Hala":9,
     "Hamra":10,
     "Hasan":11,
     "Hira":12,
     "Jamali":13,
     "Jawwad":14,
     "Laviza":15,
     "Parshant":16,
     "Rehmat":17,
     "Shehriyar":18,
    "Subhan":19,
     "Wardah":20
}

X = []
Y = []
i = 0 
path = 'D:/Project Data_org/final' 

for name, images in students_images.items():
    for image in images:
        img = cv2.imread(image,0)
        img_res = cv2.resize(img, (128,128))
    
# Detect faces
        faces = face_cascade.detectMultiScale(img_res,scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60))
# Draw rectangle around the faces
        for (x, y, w, h) in faces:
            roi_color = img_res[y:y + h, x:x + w]
            
            cv2.imwrite(os.path.join(path, str(i) + '.jpeg'), roi_color) # store the image in the path define
            Y.append(students_labels[name])
        i+=1


filelist = sorted(glob.glob('D:/Project Data_org/final//*.jpeg'), key=os.path.getmtime)

images = []

for img in filelist:
    image = cv2.imread(img,0)
    img_res = cv2.resize(image, (128,128))
    images.append(img_res)
    
img_np = np.array(images)
img_np = img_np.reshape(11172, 128*128) # 11172 is the number of images 

# convert array into dataframe
df = pd.DataFrame(Y)
  
# save the dataframe as a csv file
df.to_csv("D:/Project Data_org/data_y.csv")





