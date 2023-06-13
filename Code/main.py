" installs and imports with version "
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import ranksums, spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

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

data_exploration(rawData)

def create_histplots (csv_file, output_folder):
    data = pd.read_csv(csv_file)
    output_folder = os.path.abspath(output_folder)
    fig, axes = plt.subplots(3, 3, figsize=(15, 18))

    for i, column in enumerate(data.columns):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        ax.hist(data[column], bins='auto', edgecolor='black', linewidth=1.2)

        ax.set_title(f"Distribution of raw data: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

        ax.grid(True)

        output_file = os.path.join(output_folder, "histplots_raw_Data.png")
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

def create_histplots (csv_file, output_folder):
    data = pd.read_csv(csv_file)
    output_folder = os.path.abspath(output_folder)
    fig, axes = plt.subplots(3, 3, figsize=(15, 18))

    for i, column in enumerate(data.columns):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        ax.hist(data[column], bins='auto', edgecolor='black', linewidth=1.2)

        ax.set_title(f"Distribution of clean data: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

        ax.grid(True)


        output_file = os.path.join(output_folder, "histplots_clean_Data.png")
    plt.savefig(output_file)

csv_file = "../data/clean_Data.csv"
output_folder = "../Output"

" data imputation"
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
KNN_imputed_data = pd.DataFrame(imputer.fit_transform(clean_Data), columns=clean_Data.columns)

from sklearn.impute import IterativeImputer
import pandas as pd

imputer = IterativeImputer()
multiple_imputed_data = imputer.fit_transform(clean_Data)
multiple_imputed_data = pd.DataFrame(multiple_imputed_data, columns=clean_Data.columns)
number_of_iterations = imputer.n_iter_
print("Number of imputation iterations:", number_of_iterations)

"data normalization - 0 to 1"
normalized_data = (KNN_imputed_data - KNN_imputed_data.min()) / (KNN_imputed_data.max() - KNN_imputed_data.min())
print(normalized_data.describe())

"feature engineering --> combine two related features to create a new feature"

" train-test-split "
# normal distribution 80-20

" ML models "
# LogReg, SVM, DecisionTree, RandomForrest

" results "
# confusion matrix, accuracy, F1.score, ROC

" outlook "
# what else could / should be done
