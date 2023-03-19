import numpy as np
import shap
import pandas as pd
import matplotlib
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
import joblib
from sklearn.svm import SVC

df_QT, df_non = pd.read_csv('QT_Modl2.txt', sep="\t"), pd.read_csv('non_QT_Mold2.txt', sep="\t")


positive = df_QT
positive['tag'] = 1
negative = df_non
negative['tag'] = 0
drug_all = pd.concat([positive, negative], axis=0)

X = drug_all.iloc[:, 2:779].values
y = drug_all.iloc[:, 779].values

minmax = preprocessing.MinMaxScaler()
X_minmax = minmax.fit_transform(X)

sec_scr = []
for l in range(0, 776):
    if X_minmax[:, l].var() <= 0.001:
        sec_scr.append(l)

X = drug_all.iloc[:, 2:779]
m = list(X.columns)
x = []
for i in sec_scr:
    temp = m[i]
    x.append(temp)
x.append("D164")
X = X.drop(x, axis=1)
list = list(X)

transfer = StandardScaler()
X = transfer.fit_transform(X)
X = pd.DataFrame(X)
X.columns = list
model = joblib.load('svm.pkl')
model.fit(X, y)

for i in range(100):
    explainer = shap.KernelExplainer(model=model.predict, data=X)
    shap_values = explainer.shap_values(X)
    np.save(f'shap_values{i}', shap_values)
    # shap_values = np.load('shap_values.npy')
