import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

titanic = pd.read_csv('../titanic.csv')
print(titanic.isnull().sum())
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)
titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)
gender_num = {'male': 0, 'female': 1}
titanic['Sex'] = titanic['Sex'].map(gender_num)
titanic.drop(['Cabin', 'Embarked', 'Name', 'Ticket', 'PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)

features = titanic.drop('Survived', axis=1)
labels = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

X_train.to_csv('../train_features.csv', index=False)
X_val.to_csv('../val_features.csv', index=False)
X_test.to_csv('../test_features.csv', index=False)

y_train.to_csv('../train_labels.csv', index=False)
y_val.to_csv('../val_labels.csv', index=False)
y_test.to_csv('../test_labels.csv', index=False)