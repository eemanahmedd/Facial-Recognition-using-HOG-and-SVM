import cv2
from skimage.feature import hog
import pickle 
model = pickle.load(open("D:/Project Data_org/model_hog.sav", 'rb'))

path = "D:/Project Data_org/test.jpeg" 

image_t = cv2.imread(path,0)
img_res_t = cv2.resize(image_t, (128,128))

   
fd, hog_image = hog(img_res_t, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True, multichannel=False)

fd.shape
 # Converting into numpy array    
fd = fd.reshape(1,-1)
fd.shape
y_pred = model.predict(fd)

print(y_pred[0])
