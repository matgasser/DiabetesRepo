import pandas as pd
import plotly.express as pltex
from plotly.subplots import make_subplots
import plotly.graph_objects as pltgo
import plotly.io as pio


"""import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import stats
import seaborn as sns
import os"""

def outcomeConvert(x):
    if x == 1:
        return 'Yes'
    else:
        return 'No'


# loading the data
rawData = pd.read_csv('../data/clean_data.csv')
print(rawData)

#Outcome allg. pro Patient
rawData['Outcome'] = rawData['Outcome'].astype('category',copy=False)
rawData['Outcome'] = rawData['Outcome'].apply(outcomeConvert)
rawData.info()

rD = rawData['Outcome'].value_counts().reset_index()
rD.columns = ['Outcome', 'Quantity']
fig = pltex.bar(rD, x='Outcome', y='Quantity', title='n Patients by Outcome (no = no diabetes, yes = diabetes)')
fig.show()


#Alle Feature im Vergleich zum Outcome
fig = make_subplots(rows=2, cols=4, subplot_titles=('Outcome vs. Pregnancies',
                                                    'Outcome vs. Glucose',
                                                    'Outcome vs. Blood Pressure',
                                                    'Outcome vs. Skin Thickness',
                                                    'Outcome vs. Insulin',
                                                    'Outcome vs. BMI',
                                                    'Outcome vs. Diabetes Pedigree Func.',
                                                    'Outcome vs. Age'
                                                    ))

fig.add_trace(pltgo.Box(y=rawData['Pregnancies'], x=rawData['Outcome']), row=1, col=1)
fig.add_trace(pltgo.Box(y=rawData['Glucose'], x=rawData['Outcome']), row=1, col=2)
fig.add_trace(pltgo.Box(y=rawData['BloodPressure'], x=rawData['Outcome']), row=1, col=3)
fig.add_trace(pltgo.Box(y=rawData['SkinThickness'], x=rawData['Outcome']), row=1, col=4)
fig.add_trace(pltgo.Box(y=rawData['Insulin'], x=rawData['Outcome']), row=2, col=1)
fig.add_trace(pltgo.Box(y=rawData['BMI'], x=rawData['Outcome']), row=2, col=2)
fig.add_trace(pltgo.Box(y=rawData['DiabetesPedigreeFunction'], x=rawData['Outcome']), row=2, col=3)
fig.add_trace(pltgo.Box(y=rawData['Age'], x=rawData['Outcome']), row=2, col=4)

# Update visual layout
fig.update_layout(
    showlegend=False,
    width=1000,
    height=400,
    autosize=False,
    margin=dict(t=15, b=0, l=5, r=5),
    template="grey",
)

# update font size, axes
fig.update_coloraxes(colorbar_tickfont_size=10)
# Update font in the titles
fig.update_annotations(font_size=12)
# Reduce opacity
fig.update_traces(opacity=1)
fig.show()