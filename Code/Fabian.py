#Clustering
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots



#load data
rawdata = pd.read_csv('../data/diabetes.csv',encoding="ISO-8859-1")
rawdata.head()


#Explore the data
print(rawdata.dtypes)



#Analyzie the data
rawdata.describe()


#Analyize outcomes
diabetes_count = rawdata['Outcome'].value_counts()
diabetes_count



#hisplot for the distribution of age
sns.histplot(data=rawdata, x='Age')
plt.title('Distribution of Age')
plt.show()



#scatterplot Glucose & BMI

sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=rawdata)
plt.title('Glucose Levels vs. BMI (Colored by Diabetes Outcome)')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.show()