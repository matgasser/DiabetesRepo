import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# load data
rawData = pd.read_csv("../data/diabetes.csv")

# separating the data and labels
X = rawData.drop(columns = 'Outcome', axis=1)
Y = rawData['Outcome']

# standadardization of the data --> verbesserte Konvergenz und weniger Ausreissereinfluss
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

# Splitting data in Train - Test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=0)
print(X.shape, X_train.shape, X_test.shape)

# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)
