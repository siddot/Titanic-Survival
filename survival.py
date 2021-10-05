import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# load data
titanic = pd.read_csv('titanic.csv')
titanic.head()


titanic.dropna(inplace=True)

titanic = titanic.replace({'Sex':{'male':0, 'female':1}})
sc = StandardScaler()
X_wo_Sex = titanic.loc[:, ['Pclass', 'Age', 'Fare']]
X_wo_Sex = sc.fit_transform(X_wo_Sex)

X = titanic.loc[:, ['Pclass', 'Sex', 'Age', 'Fare']]
y = titanic.loc[:, ['Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print('predictions: ', predictions)

acc = classification_report(y_test, predictions)
print(acc)

titanic.isnull().sum()
titanic.head()
