import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import preprocessing, metrics
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline


df_QT, df_non = pd.read_csv('qt_Modl2.txt', sep="\t"), pd.read_csv('non_QT_Mold2.txt', sep="\t")


positive = df_QT
positive['tag'] = 1
negative = df_non
negative['tag'] = 0
drug_all = pd.concat([positive, negative], axis=0)
X = drug_all.iloc[:, 2:779]
y = drug_all.iloc[:, 779].values


feature_link = ['D718', 'D756', 'D661', 'D759', 'D627', 'D130', 'D647', 'D626', 'D757', 'D598']
X = X[feature_link]


def SVM_grid_search_result(X, y, pre=0):
    'GridSearch'
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    C = np.arange(0.5, 10, 0.5)
    gamma = [1, 0.1, 0.01, 0.001, 0.0001]
    if pre == 0:
        clf = Pipeline([('ss', StandardScaler()), ('svc', SVC())])
        param_grid = {'svc__gamma': gamma, 'svc__C': C}
    else:
        clf = SVC()
        param_grid = {'gamma': gamma, 'C': C}
    grid_search = GridSearchCV(clf, param_grid, scoring='balanced_accuracy', n_jobs=-1, cv=kfold)

    grid_result = grid_search.fit(X, y)
    print("SVM_Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))

    'result'
    df_predict_svm = pd.DataFrame(y, columns=[0])
    for r in range(1000):
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        if pre == 0:
            model = Pipeline([('ss', StandardScaler()), ('svc',
                                                         SVC(kernel='rbf', gamma=grid_search.best_params_['svc__gamma'],
                                                             C=grid_search.best_params_['svc__C']))])
        else:
            model = SVC(kernel='rbf', gamma=grid_search.best_params_['gamma'], C=grid_search.best_params_['C'])
        model.fit(X, y)
        y_predict = cross_val_predict(model, X, y, cv=kfold)
        df_predict_temp = pd.DataFrame(y_predict, columns=[r])
        df_predict_svm = pd.concat([df_predict_svm, df_predict_temp], axis=1)

    df_predict_svm.to_csv('df_predict_svm_10.csv')
    joblib.dump(model, "svm_10.pkl")





SVM_grid_search_result(X, y, pre=0)
