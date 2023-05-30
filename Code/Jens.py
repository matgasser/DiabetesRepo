"""Project Checklist
Installs and Imports --> CHECK

Get the Data --> CHECK

Basic Info --> CHECK

Replacing Zero Values --> CHECK
Exploratory Analysis --> CHECK
Correlation --> CHECK
Box Plot --> CHECK
Histograms --> CHECK
Target Split

Train Test Split

Data Preparation

Missing Data
Logarithm
Standard Scaler
LinearSVC

Artificial Neural Network

Selecting the threshold

Making Predictions

Final Evaluation"""

"""

FoDS - Project
Diabetes Dataset
Group G1G

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import stats
import seaborn as sns
import os


# loading the data
rawData = pd.read_csv('../data/diabetes.csv')

# some information about the data
print(rawData.columns)
print(rawData.shape)
print(rawData.head(3))
rawData.info()
print(rawData.describe())

print(rawData.isnull().sum())
# there are no missing data values


# How many have diabetes and how many don't
print(rawData.value_counts('Outcome'))

# general plot configuration
plt.figure(figsize=(8, 8))

# Some Plots on the data
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

"""
As we can see, there is no "missing-data".
But there are a lot of zero-values in Glucose, BloodPressure, SkinThickness, Insulin and BMI, that are impossible
To be able to work with this dataset, we replace the zeros with NaN
"""

# copy the data
clean_Data = rawData.copy(deep = True)
clean_Data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = clean_Data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
print(clean_Data.isnull().sum())
print(clean_Data.describe())

# save the clean data
clean_Data.to_csv("../Data/clean_Data.csv", index = False)

# new plots
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

create_histplots(csv_file, output_folder)



"""
Exploratory Analysis
    Correlation Heatmap
    Boxplot
    Histograms
    Skewness:
        A left-skewed distribution has a long left tail. Left-skewed distributions are also called negatively-skewed distributions. That’s because there is a long tail in the negative direction on the number line. The mean is also to the left of the peak.
        A right-skewed distribution has a long right tail. Right-skewed distributions are also called positive-skew distributions. That’s because there is a long tail in the positive direction on the number line. The mean is also to the right of the peak.
    Scatter matrix
    Pairplot
    Pearson Correlation Coefficient
"""

# Pairplot
pairplot=sns.pairplot(clean_Data, hue = 'Outcome')
plt.tight_layout()
plt.savefig("../Output/pairplot_clean_Data.png")

# Correlation between all the features
plt.figure(figsize=(10, 10))
correlation = sns.heatmap(clean_Data.corr(), annot=True, cmap='viridis')
plt.tight_layout()
plt.savefig("../Output/Heatmap clean_Data.png")

plt.figure(figsize=(10, 10))
sorted_heatmap = sns.heatmap(clean_Data.corr()[['Outcome']].sort_values(by='Outcome', ascending=False), vmin=-1, vmax=1, annot=True, cmap='viridis')
sorted_heatmap.set_title('Correlation with Outcome', fontdict={'fontsize':14}, pad=10)
plt.savefig("../Output/sorted_heatmap.png")

"""
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database --> Main
# https://www.kaggle.com/code/busegngr/building-ml-model-for-diabetes --> Building ML Model for Diabetes
# https://www.kaggle.com/code/emirhanozkan/diabetes-prediction-using-machine-learning --> Diabetes Prediction Using Machine Learning
# https://www.kaggle.com/code/salihagorgulu/feature-engineering-diabetes-ml --> Feature Engineering,Diabetes,ML
# https://www.kaggle.com/code/amirrezamasoumi/76-accuracy-on-diabetes-with-eda --> 76% accuracy on diabetes with EDA
# https://www.kaggle.com/code/tohidyousefi/predicting-diabetes-with-logistic-regression --> Predicting_Diabetes_with_Logistic_Regression
# https://www.kaggle.com/code/mridulsyed/diabetes-prediction --> Diabetes Prediction
# https://www.kaggle.com/code/caiofernandeslima/machine-learning-for-diabetes-prediction -->m Machine Learning for Diabetes Prediction
# https://www.kaggle.com/code/gvyshnya/clustering-pima-indians-diabetes-cases/notebook#Express-EDA-Of-Numeric-Feature-Variables
"""
