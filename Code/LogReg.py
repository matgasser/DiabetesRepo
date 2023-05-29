#Lineare Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
# loading the data
rawData = pd.read_csv(r'../data/diabetes.csv') #unsbearbeitete Daten

sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(rawData.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
y = rawData.Outcome

#Data splitting
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

logistic = LogisticRegression()
logistic.fit(X_train,y_train)

accuracy = logistic.score(X_test,y_test)
print("Accurecy = " , accuracy * 100, "%")

log_prediciton = logistic.predict(X_test)

print("Classification Report is:\n", classification_report(y_test, log_prediciton))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_prediciton))
print("Training Score:\n", logistic.score(X_train, y_train) * 100)
print("Mean Squared Error:\n", mean_squared_error(y_test, log_prediciton))
print("R2 score is:\n", r2_score(y_test, log_prediciton))