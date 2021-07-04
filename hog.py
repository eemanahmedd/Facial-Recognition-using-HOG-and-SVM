import cv2
import numpy as np
import pandas as pd
import glob
import os
from skimage.feature import hog

#df_Y = pd.read_csv("D:/Project Data_org/data_y.csv")

hog_features = []

filelist = sorted(glob.glob('D:/Project Data_org/final//*.jpeg'), key=os.path.getmtime)

for img in filelist:
    image = cv2.imread(img,0)
    img_res = cv2.resize(image, (128,128))
    fd, hog_image = hog(img_res, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=False)
    hog_features.append(fd)
    
len(hog_features)

#numpy array

hog_features = np.array(hog_features)
hog_features.shape

# convert array into dataframe
df_X = pd.DataFrame(hog_features)
  
# save the dataframe as a csv file
df_X.to_csv("D:/Project Data_org/data_hog.csv")