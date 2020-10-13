import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# We are reading our data
df = pd.read_csv("C:\\Users\\master\\Downloads\\heart.csv")

# First 5 rows of our data
df.head()

#Data contains;

#age - age in years
#sex - (1 = male; 0 = female)
#cp - chest pain type
#trestbps - resting blood pressure (in mm Hg on admission to the hospital)
#chol - serum cholestoral in mg/dl
#fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#restecg - resting electrocardiographic results
#thalach - maximum heart rate achieved
#exang - exercise induced angina (1 = yes; 0 = no)
#oldpeak - ST depression induced by exercise relative to rest
#slope - the slope of the peak exercise ST segment
#ca - number of major vessels (0-3) colored by flourosopy
#thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
#target - have disease or not (1=yes, 0=no)

#Data Exploration
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()

#Data
y = df.target.values
x_data = df.drop(['target'], axis = 1)

#Normalizing the data
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

#We split our data into 80-20 train and test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Transpose matrix
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T                                    

#Using KNN to fit how model
#Have found the best "K" for this model K = 7

knn = KNeighborsClassifier(n_neighbors = 7)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)

#Prediction
y_pred = knn.predict(x_test.T)

#Accuracy
accuracies = knn.score(x_test.T, y_test.T)*100

print("Test Accuracy of KNN Algorithm: {:.2f}%".format(accuracies))

###Create a Pickle file using serialization
import pickle
pickle_out = open("knn.pk1","wb")
pickle.dump(knn , pickle_out )
pickle_out.close()




                                    











