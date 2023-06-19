#random forrest plot


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


# load data
clean_Data = pd.read_csv("../data/clean_Data.csv")

# mean-imputation
from sklearn.impute import SimpleImputer
imputer = imputer = SimpleImputer(strategy='mean')
mean_imputed_data = pd.DataFrame(imputer.fit_transform(clean_Data), columns=clean_Data.columns)

# median-imputation
imputer = SimpleImputer(strategy='median')
median_imputed_data = pd.DataFrame(imputer.fit_transform(clean_Data), columns=clean_Data.columns)

# KNN-imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
KNN_imputed_data = pd.DataFrame(imputer.fit_transform(clean_Data), columns=clean_Data.columns)

# multiple-imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer()
multiple_imputed_data = pd.DataFrame(imputer.fit_transform(clean_Data), columns=clean_Data.columns)

###############################################################

# separating the data and labels
X_mean = mean_imputed_data.drop(columns = 'Outcome', axis=1)
Y_mean = mean_imputed_data['Outcome']

# standadardization of the data
scaler = StandardScaler()
scaler.fit(X_mean)
standardized_data = scaler.transform(X_mean)
X_mean = standardized_data

# Splitting data in Train - Test
X_mean_train, X_mean_test, Y_mean_train, Y_mean_test = train_test_split(X_mean,Y_mean, test_size = 0.2, stratify=Y_mean, random_state=0)
print(X_mean.shape, X_mean_train.shape, X_mean_test.shape)

# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_mean_train, Y_mean_train)

# accuracy score on the training data
X_mean_train_prediction = classifier.predict(X_mean_train)
mean_training_data_accuracy = accuracy_score(X_mean_train_prediction, Y_mean_train)
print('Accuracy score of the mean-imputed training data : ', mean_training_data_accuracy)

# accuracy score on the test data
X_mean_test_prediction = classifier.predict(X_mean_test)
mean_test_data_accuracy = accuracy_score(X_mean_test_prediction, Y_mean_test)
print('Accuracy score of the mean-imputed test data : ', mean_test_data_accuracy)

###############################################################

# separating the data and labels
X_median = median_imputed_data.drop(columns = 'Outcome', axis=1)
Y_median = median_imputed_data['Outcome']

# standadardization of the data
scaler = StandardScaler()
scaler.fit(X_median)
standardized_data = scaler.transform(X_median)
X_median = standardized_data

# Splitting data in Train - Test
X_median_train, X_median_test, Y_median_train, Y_median_test = train_test_split(X_median,Y_median, test_size = 0.2, stratify=Y_median, random_state=0)
print(X_median.shape, X_median_train.shape, X_median_test.shape)

# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_median_train, Y_median_train)

# accuracy score on the training data
X_median_train_prediction = classifier.predict(X_median_train)
median_training_data_accuracy = accuracy_score(X_median_train_prediction, Y_median_train)
print('Accuracy score of the median-imputed training data : ', median_training_data_accuracy)

# accuracy score on the test data
X_median_test_prediction = classifier.predict(X_median_test)
median_test_data_accuracy = accuracy_score(X_median_test_prediction, Y_median_test)
print('Accuracy score of the median-imputed test data : ', mean_test_data_accuracy)

###############################################################

# separating the data and labels
X_KNN = KNN_imputed_data.drop(columns = 'Outcome', axis=1)
Y_KNN = KNN_imputed_data['Outcome']

# standadardization of the data
scaler = StandardScaler()
scaler.fit(X_KNN)
standardized_data = scaler.transform(X_KNN)
X_KNN = standardized_data

# Splitting data in Train - Test
X_KNN_train, X_KNN_test, Y_KNN_train, Y_KNN_test = train_test_split(X_KNN,Y_KNN, test_size = 0.2, stratify=Y_KNN, random_state=0)
print(X_KNN.shape, X_KNN_train.shape, X_KNN_test.shape)

# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_KNN_train, Y_KNN_train)

# accuracy score on the training data
X_KNN_train_prediction = classifier.predict(X_KNN_train)
KNN_training_data_accuracy = accuracy_score(X_KNN_train_prediction, Y_KNN_train)
print('Accuracy score of the KNN-imputed training data : ', KNN_training_data_accuracy)

# accuracy score on the test data
X_KNN_test_prediction = classifier.predict(X_KNN_test)
KNN_test_data_accuracy = accuracy_score(X_KNN_test_prediction, Y_KNN_test)
print('Accuracy score of the KNN-imputed test data : ', KNN_test_data_accuracy)

###############################################################

# separating the data and labels
X_multiple = multiple_imputed_data.drop(columns = 'Outcome', axis=1)
Y_multiple = multiple_imputed_data['Outcome']

# standadardization of the data
scaler = StandardScaler()
scaler.fit(X_multiple)
standardized_data = scaler.transform(X_multiple)
X_multiple = standardized_data

# Splitting data in Train - Test
X_multiple_train, X_multiple_test, Y_multiple_train, Y_multiple_test = train_test_split(X_multiple,Y_multiple, test_size = 0.2, stratify=Y_multiple, random_state=0)
print(X_multiple.shape, X_multiple_train.shape, X_multiple_test.shape)

# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_multiple_train, Y_multiple_train)

# accuracy score on the training data
X_multiple_train_prediction = classifier.predict(X_multiple_train)
multiple_training_data_accuracy = accuracy_score(X_multiple_train_prediction, Y_multiple_train)
print('Accuracy score of the multiple-imputed training data : ', multiple_training_data_accuracy)

# accuracy score on the test data
X_multiple_test_prediction = classifier.predict(X_multiple_test)
multiple_test_data_accuracy = accuracy_score(X_multiple_test_prediction, Y_multiple_test)
print('Accuracy score of the multiple-imputed test data : ', multiple_test_data_accuracy)



median_imputed_data.describe()

median_imputed_data['Outcome'].value_counts()

from sklearn.model_selection import train_test_split
x = median_imputed_data.drop(['Outcome'],axis=1)
y = median_imputed_data['Outcome']



sc= StandardScaler()
x_scaled= sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=0)
x_train.shape, y_train.shape

x_test.shape, y_test.shape


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)


confmat = confusion_matrix(y_pred, y_test)
confmat



random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

confmat1 = confusion_matrix(y_pred, y_test)
confmat1

from sklearn import metrics
cm=metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_pred,y_test,labels=random_forest.classes_),
                              display_labels=random_forest.classes_)
forest_plot = cm.plot(cmap="magma")

plt.savefig("../Output/forest_plot.png")

accuracy_score(y_pred, y_test)