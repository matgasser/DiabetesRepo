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

    fig, axes = plt.subplots(3, 3, figsize=(15, 18))

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
accuracy = logistic.score(X_test,Y_test)
log_prediciton = logistic.predict(X_test)
c_m = confusion_matrix(Y_test, log_prediciton)
print("Accuracy = " , accuracy)
print("Clas. Report:\n", classification_report(Y_test, log_prediciton))
print("Training Score:", logistic.score(X_train, Y_train) * 100)
print("Mean Squared Error:", mean_squared_error(Y_test, log_prediciton))
print("R2 score is:", r2_score(Y_test, log_prediciton))
print("Confusion Matrix:\n", c_m)
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=c_m)
disp.plot()
plt.show()

# Support Vector Machines (SVM)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

Y_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, Y_train_prediction)
print('Accuracy score of the training data with SVM : ', training_data_accuracy)

Y_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, Y_test_prediction)
print('Accuracy score of the test data with SVM: ', test_data_accuracy)

# DecisionTree
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Decision tree plot
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.show()

# RandomForrest



" results "
# confusion matrix, accuracy, F1.score, ROC

" outlook "
# what else could / should be done
