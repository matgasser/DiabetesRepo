" installs and imports with version "
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import ranksums, spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, r2_score, mean_squared_error,ConfusionMatrixDisplay,auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

import warnings
warnings.filterwarnings("ignore")

" load the data "
rawData = pd.read_csv('../data/diabetes.csv')

# data exploration
def data_exploration(data):
    print(data.head(10))
    print(data.shape)
    print(data.columns)
    print(data.info())
    print(data.describe())
    print(data.isna().sum())
    print('############################################################')

data_exploration(rawData)

def create_histplots(csv_file, output_folder):
    data = pd.read_csv(csv_file)
    output_folder = os.path.abspath(output_folder)
    file_name = os.path.basename(csv_file)
    file_name_without_extension = os.path.splitext(file_name)[0]

    fig, axes = plt.subplots(6, 3, figsize=(18, 21))
    plt.subplots_adjust(hspace=0.5)

    for i, column in enumerate(data.columns):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        ax.hist(data[column], bins='auto', edgecolor='black', linewidth=1.2)

        ax.set_title(f"Distribution of {file_name_without_extension}: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

        ax.grid(True)

    output_file = os.path.join(output_folder, f"histplots_{file_name_without_extension}.png")
    plt.savefig(output_file)

csv_file = "../data/diabetes.csv"
output_folder = "../Output"

create_histplots(csv_file, output_folder)

" data cleaning "
clean_Data = rawData.copy(deep = True)
clean_Data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = clean_Data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

# save the clean data
clean_Data.to_csv("../Data/clean_Data.csv", index = False)

data_exploration(clean_Data)

" create histplots of clean_Data "
csv_file = "../data/clean_Data.csv"
output_folder = "../Output"

create_histplots(csv_file, output_folder)

" data imputation"
from sklearn.impute import IterativeImputer
imputer = IterativeImputer()
multiple_imputed_data = imputer.fit_transform(clean_Data)
multiple_imputed_data = pd.DataFrame(multiple_imputed_data, columns=clean_Data.columns)
number_of_iterations = imputer.n_iter_
print("Number of imputation iterations:", number_of_iterations)

"feature eng. - creating new features "
#Glucose * BMI
multiple_imputed_data['Glucose_BMI'] = multiple_imputed_data['Glucose'] * multiple_imputed_data['BMI']

#Age * Pregnancies
multiple_imputed_data['Age_Pregnancies'] = multiple_imputed_data['Age'] * multiple_imputed_data['Pregnancies']

#Insulin / Glucose
multiple_imputed_data['Insulin_Glucose'] = multiple_imputed_data['Insulin'] / multiple_imputed_data['Glucose']

#SkinThickness + BMI
multiple_imputed_data['SkinThickness_BMI'] = multiple_imputed_data['SkinThickness'] + multiple_imputed_data['BMI']

#BloodPressure * DiabetesPedigreeFunction
multiple_imputed_data['BP_DiabetesPedigree'] = multiple_imputed_data['BloodPressure'] * multiple_imputed_data['DiabetesPedigreeFunction']

#Glucose - Age
multiple_imputed_data['Glucose_Age'] = multiple_imputed_data['Glucose'] - multiple_imputed_data['Age']

#BMI / Age
multiple_imputed_data['BMI_Age'] = multiple_imputed_data['BMI'] / multiple_imputed_data['Age']

#Insulin * Pregnancies
multiple_imputed_data['Insulin_Pregnancies'] = multiple_imputed_data['Insulin'] * multiple_imputed_data['Pregnancies']

#Glucose / Age
multiple_imputed_data['Glucose_Age_Ratio'] = multiple_imputed_data['Glucose'] / multiple_imputed_data['Age']

# Updated DataFrame
print(multiple_imputed_data.head())
print(multiple_imputed_data.shape)


"data normalization - 0 to 1"
normalized_data = (multiple_imputed_data - multiple_imputed_data.min()) / (multiple_imputed_data.max() - multiple_imputed_data.min())
print(normalized_data.describe())

data_exploration(normalized_data)

" plot the imputated-normalized data "
normalized_data = normalized_data.astype(float)
normalized_data['Outcome'] = normalized_data['Outcome'].astype(int)
normalized_data.to_csv("../Data/normalized_data.csv", index = False)

csv_file = "../data/normalized_data.csv"
output_folder = "../Output"

create_histplots(csv_file, output_folder)

"feature engineering --> combine two related features to create a new feature"

" train-test-split "
# normal distribution 80-20
X = normalized_data.drop(columns='Outcome', axis=1)
Y = normalized_data['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)


" ML models "
# LogReg, SVM, DecisionTree, RandomForrest
#LogReg
logistic = LogisticRegression()
logistic.fit(X_train,Y_train)
log_prediciton = logistic.predict(X_test)
c_m = confusion_matrix(Y_test, log_prediciton)
lr_auc_roc = roc_auc_score(Y_test, logistic.predict_proba(X_test)[:, 1])
print("Test Score (Acurracy) = " , logistic.score(X_test,Y_test))
print("Training Score:", logistic.score(X_train, Y_train) * 100)
print("Clas. Report:\n", classification_report(Y_test, log_prediciton))
print("Mean Squared Error:", mean_squared_error(Y_test, log_prediciton))
print("R2 score is:", r2_score(Y_test, log_prediciton))
print("Logistic Regression AUC-ROC:", lr_auc_roc)
print("Confusion Matrix:\n", c_m)
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=c_m)
disp.plot()
plt.show()


# Support Vector Machines (SVM)
classifier = svm.SVC(kernel='linear', probability=True)
classifier.fit(X_train, Y_train)

Y_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, Y_train_prediction)
print('Accuracy score of the training data with SVM : ', training_data_accuracy)

Y_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, Y_test_prediction)
print('Accuracy score of the test data with SVM: ', test_data_accuracy)
svm_auc_roc = roc_auc_score(Y_test, classifier.predict_proba(X_test)[:, 1])
print("SVM AUC-ROC:", svm_auc_roc)

# DecisionTree
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)
dt_auc_roc = roc_auc_score(Y_test, clf.predict_proba(X_test)[:, 1])
print("Decision Tree AUC-ROC:", dt_auc_roc)

# Decision tree plot
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.show()

# RandomForrest
random_forest = RandomForestClassifier(criterion = "gini",
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
y_pred_rf = random_forest.predict(X_test)
confmat1 = confusion_matrix(y_pred_rf, Y_test)
cm=metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_pred_rf,Y_test,labels=random_forest.classes_),
                              display_labels=random_forest.classes_)
forest_plot = cm.plot(cmap="magma")
accuracy_score(y_pred_rf, Y_test)
plt.savefig("../Output/forest_plot.png")

random_forest.fit(X_train, Y_train)
rf_auc_roc = roc_auc_score(Y_test, random_forest.predict_proba(X_test)[:, 1])
print("Random Forest AUC-ROC:", rf_auc_roc)

" results "
# confusion matrix, accuracy, F1.score, ROC

" outlook "
# what else could / should be done
