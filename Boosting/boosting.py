import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

titanic = pd.read_csv('../titanic.csv')
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)
gender_num = {'male': 0, 'female': 1}
titanic['Sex'] = titanic['Sex'].map(gender_num)
titanic.drop(['Cabin', 'Embarked', 'Name', 'Ticket', 'PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)
features = titanic.drop('Survived', axis=1)
labels = titanic['Survived']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#Hyperparameter tuning
def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

gb = GradientBoostingClassifier()
parameters = {
    'n_estimators': [5, 50, 250, 500],
    'max_depth': [1, 3, 5, 7, 9],
    'learning_rate': [0.01, 0.1, 1, 10, 100]
}
cv = GridSearchCV(gb, parameters, cv=5)
cv.fit(features, labels.values.ravel())

print_results(cv)
print(cv.best_estimator_)

pred = cv.predict(X_val)
Testpred  = cv.predict(X_test)

# Evaluate Matrix
accuracy = np.mean(pred == y_val)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
cm = confusion_matrix(y_val, pred)
print("Confusion Matrix:")
print(cm)

# Evaluate Matrix
accuracy = np.mean(Testpred == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
cm = confusion_matrix(y_test, Testpred)
print("Confusion Matrix:")
print(cm)