import pandas as pd
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

tr_features = pd.read_csv('../train_features.csv')
tr_labels = pd.read_csv('../train_labels.csv')

def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

estimators = [('rf', RandomForestClassifier()),
              ('gb', GradientBoostingClassifier())]

sc = StackingClassifier(estimators=estimators)
sc.get_params()
parameters = {
    'gb__n_estimators': [50, 100],
    'rf__n_estimators': [50, 100],
    'final_estimator': [LogisticRegression(C=0.1),
                        LogisticRegression(C=1),
                        LogisticRegression(C=10)],
    'passthrough': [True, False]
}
cv = GridSearchCV(sc, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)
print(cv.best_estimator_)