import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import stats
import seaborn as sns
import os


# loading the data :)
rawData = pd.read_csv('../data/diabetes.csv')

