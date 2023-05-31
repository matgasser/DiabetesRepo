from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
import pandas as pd

# Load the dataset
cleandat = pd.read_csv(r'../data/clean_Data.csv')
cd = cleandat.dropna()

# Split the dataset into features (X) and target variable (y)
X = cd.drop('Outcome', axis=1)
y = cd['Outcome']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree classifier
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)

# Create and train the SVM classifier
clf_svm = SVC()
clf_svm.fit(X_train, y_train)

# Create and train the Random Forest classifier
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)

# Create and train the Logistic Regression classifier
clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred_dt = clf_dt.predict(X_test)
y_pred_svm = clf_svm.predict(X_test)
y_pred_rf = clf_rf.predict(X_test)
y_pred_lr = clf_lr.predict(X_test)

# Calculate the recall for each classifier
recall_dt = recall_score(y_test, y_pred_dt, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')
recall_rf = recall_score(y_test, y_pred_rf, average='macro')
recall_lr = recall_score(y_test, y_pred_lr, average='macro')

# Print the recall scores
print("Decision Tree Recall:", recall_dt)
print("SVM Recall:", recall_svm)
print("Random Forest Recall:", recall_rf)
print("Logistic Regression Recall:", recall_lr)
