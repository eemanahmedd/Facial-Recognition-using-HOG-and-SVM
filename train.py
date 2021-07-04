from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pickle 

Y = pd.read_csv("D:/Project Data_org/data_Y.csv")
Y.head()
Y = np.array(Y)

X = pd.read_csv("D:/Project Data_org/data_hog.csv")
# Drop first column of dataframe
X = X.iloc[: , 1:]
X.head()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

X_train.shape

# Create and fit the model

svm_rbf = SVC(kernel='rbf' , class_weight='balanced' , C=1000 , gamma=0.0082)
svm_rbf.fit(X_train , y_train.ravel())

# Predict on the test features, print the results

y_pred = svm_rbf.predict(X_test)
print("HOG Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))



from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# Visualize it as a heatmap
import seaborn
import matplotlib.pyplot as plt
seaborn.heatmap(conf_mat)
plt.show()

# filename = 'D:/Project Data_org/model_hog.sav'
# pickle.dump(svm_rbf, open(filename, 'wb'))
