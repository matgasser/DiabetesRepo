import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Load the dataset
cleandat = pd.read_csv(r'../data/clean_Data.csv')
# median-imputation
imputer = SimpleImputer(strategy='median')
median_imputed_data = pd.DataFrame(imputer.fit_transform(cleandat), columns=cleandat.columns)

# Split the dataset into features (X) and target variable (y)
X = median_imputed_data.drop('Outcome', axis=1)
y = median_imputed_data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a decision tree classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# Create and train an SVM classifier
svm_clf = SVC(probability=True)
svm_clf.fit(X_train, y_train)

# Create and train a random forest classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Create and train a logistic regression classifier
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)

# Calculate the AUC-ROC for each classifier
dt_auc_roc = roc_auc_score(y_test, dt_clf.predict_proba(X_test)[:, 1])
svm_auc_roc = roc_auc_score(y_test, svm_clf.predict_proba(X_test)[:, 1])
rf_auc_roc = roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1])
lr_auc_roc = roc_auc_score(y_test, lr_clf.predict_proba(X_test)[:, 1])

# Print the AUC-ROC scores
print("Decision Tree AUC-ROC:", dt_auc_roc)
print("SVM AUC-ROC:", svm_auc_roc)
print("Random Forest AUC-ROC:", rf_auc_roc)
print("Logistic Regression AUC-ROC:", lr_auc_roc)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
dt_precision = precision_score(y_test, dt_pred)
print("Decision Tree Precision:", dt_precision)

# Create and train the SVM classifier
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
svm_precision = precision_score(y_test, svm_pred)
print("SVM Precision:", svm_precision)

# Create and train the Random Forest classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_precision = precision_score(y_test, rf_pred)
print("Random Forest Precision:", rf_precision)

# Create and train the Logistic Regression classifier
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
lr_precision = precision_score(y_test, lr_pred)
print("Logistic Regression Precision:", lr_precision)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

svm = SVC()
svm.fit(X_train, y_train)

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred_decision_tree = decision_tree.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_random_forest = random_forest.predict(X_test)
y_pred_logistic_regression = logistic_regression.predict(X_test)

# Calculate the F1 score for each model
f1_decision_tree = f1_score(y_test, y_pred_decision_tree)
f1_svm = f1_score(y_test, y_pred_svm)
f1_random_forest = f1_score(y_test, y_pred_random_forest)
f1_logistic_regression = f1_score(y_test, y_pred_logistic_regression)

# Print the F1 scores
print("F1 Score - Decision Tree:", f1_decision_tree)
print("F1 Score - SVM:", f1_svm)
print("F1 Score - Random Forest:", f1_random_forest)
print("F1 Score - Logistic Regression:", f1_logistic_regression)