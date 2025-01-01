import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

RandomForestClassifier().get_params()

tr_features = pd.read_csv('../train_features.csv')
tr_labels = pd.read_csv('../train_labels.csv')

X_train, X_test, y_train, y_test = train_test_split(tr_features, tr_labels, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


rf = RandomForestClassifier()
parameters = {
    'n_estimators': [5, 50, 250, 500],
    'max_depth': [4, 8, 16, 32, None]
}
cv = GridSearchCV(rf, parameters, cv=5)
cv.fit(X_train, y_train.values.ravel())

print_results(cv)

pred = cv.predict(X_val)

print(pred)

# Confusion matrix and classification report for validation set
print("\nConfusion Matrix:\n", confusion_matrix(y_val, pred))
print("\nClassification Report:\n", classification_report(y_val, pred))

