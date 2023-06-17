#Log Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
# Load the dataset
cleandat = pd.read_csv(r'../data/clean_Data.csv')
# median-imputation
imputer = SimpleImputer(strategy='median')
median_imputed_data = pd.DataFrame(imputer.fit_transform(cleandat), columns=cleandat.columns)

sc_X = StandardScaler()
# Split the dataset into features (X) and target variable (y)
X = median_imputed_data.drop('Outcome', axis=1)
y = median_imputed_data['Outcome']

"""X = pd.DataFrame(sc_X.fit_transform(cd.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = cd.Outcome"""

#Data splitting
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#logReg
logistic = LogisticRegression()
logistic.fit(X_train,y_train)
accuracy = logistic.score(X_test,y_test)
log_prediciton = logistic.predict(X_test)
print("Accuracy = " , accuracy)
print("Clas. Report:\n", classification_report(y_test, log_prediciton))

c_m = confusion_matrix(y_test, log_prediciton)
print("Confusion Matrix:\n", c_m)
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=c_m)
disp.plot()
plt.show()


print("Training Score:", logistic.score(X_train, y_train) * 100)
print("Mean Squared Error:", mean_squared_error(y_test, log_prediciton))
print("R2 score is:", r2_score(y_test, log_prediciton))


"""Glucose * BMI: Multiplying the 'Glucose' and 'BMI' features to capture the potential interaction between blood glucose levels and body mass index.
Age * Pregnancies: Calculating the product of 'Age' and 'Pregnancies' to capture the combined effect of age and the number of pregnancies.
Insulin / Glucose: Dividing the 'Insulin' feature by 'Glucose' to capture the ratio of serum insulin level to blood glucose concentration.
SkinThickness + BMI: Adding the 'SkinThickness' and 'BMI' features to capture the combined effect of subcutaneous fat thickness and body mass index.
BloodPressure * DiabetesPedigreeFunction: Multiplying the 'BloodPressure' and 'DiabetesPedigreeFunction' features to capture the potential interaction between blood pressure and the genetic influence of diabetes.
Glucose - Age: Calculating the difference between 'Glucose' and 'Age' to capture the deviation of blood glucose level from the average age.
BMI / Age: Dividing the 'BMI' feature by 'Age' to capture the ratio of body mass index to age.
Insulin * Pregnancies: Multiplying the 'Insulin' and 'Pregnancies' features to capture the combined effect of serum insulin level and the number of pregnancies."""


