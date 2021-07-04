# Facial-Recognition-using-HOG-and-SVM
This is a Machine Learning course project where we had 21 classes and we were asked to make a model that recognize faces in real-time and on random images 

## Face Recognition
Using OpenCV inbuilt functions to recognize faces of my classmates. The code is written using OpenCV using haarcascade detector to identify facial features. 
HOG Feature Descriptor is used to recognize difference between faces.
Support Vector Machine (SVM) is used to identify different faces. 
Download OpenCv library in order to move forward using following command:

*pip install opencv-python*


## Description
***Set the paths that are defined in the code, according to your system.***

Create_df.py helps to create dataframe of labels and store haar features of images. Use haarcascade_frontalface_default.xml to extract haar features using OpenCV library.
Images are resized to 128 x 128 for better feature extraction using HOG. y-labels and these images were a little asynchronous. 
It was corrected manually by checking where the y-labels do not matches the images. 
Wherever the df_Y had a change in class, its corresponding image was displayed. 
And if the image or the label didn’t match, it was corrected accordingly. 
The y-labels are already stored as .csv file in the project folder as df_Y.

Hog.py helps to extract useful information from the images. Skimage.feature library is used in order to apply HOG. 
HOG works with something called a block which is similar to a sliding window. A block is considered as a pixel grid in which gradients are constituted from the 
magnitude and direction of change in the intensities of the pixel within the block. The hog 
features are already stored as .csv file in the project folder as data_hog.


Once the preprocessing of the dataset is done, we can now use a classifier to train it for the given 
dataset. We chose Support Vector Machine as our classifier for this project. SVM’s are 
supervised learning models with associated learning algorithms that analyze data and recognize 
patterns. We use a nonlinear support vector classification model with the kernel as radial basis 
function (rbf). We fit the classifier with the data that was transformed using HOG and the labels 
which were extracted while reading the images in the *'D :/Project Data_org/final'* directory (This 
would be a different one according to your system.) Once the classifier is trained we then take 
our testing dataset and check whether the label predicted by our classifier is same as that given in 
the test set. Finally, model is saved as *model_hog.sav* in local directory to use it later. This is 
done to avoid going through the whole pipeline next time user wants to use the model. 


Save_test.py and pred_test.py helps to predict on a random image stored on your machine.


Model_test.py helps to predict a face in real-time. For real-time and on random image, the 
preprocessing steps are same as they were before training the model. 


Finally, gui.py is the whole structure of our Machine Learning project. 
The GUI needs improvement but it still performs quite well.
Not only this, our model needs improvement in  robustness, which will help in predicting faces in different scenarios. 
