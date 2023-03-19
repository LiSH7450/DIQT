import pandas as pd
import joblib
from matplotlib.ticker import MultipleLocator
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import numpy as np

#读取数据
X_fears = pd.read_csv('fears_Mold2.txt', sep='\t')
qt = pd.read_csv('qt_Modl2.txt',sep='\t')
non_qt = pd.read_csv('non_QT_Mold2.txt', '\t')


X_ = pd.concat([X_fears.head(95), non_qt], axis=0)

y_true = [1]*95+[0]*95
# 特征选取
positive = qt
positive['tag'] = 1
negative = non_qt
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


feature_all = list(X)
feature_10 = ['D718', 'D756', 'D661', 'D759', 'D627', 'D130', 'D647', 'D626', 'D757', 'D598']
X_full = X_[feature_all]
X_10 = X_[feature_10]
# 载入模型
model_svm = joblib.load('svm.pkl')
model_svm_10 = joblib.load('svm_10.pkl')
y_predict = model_svm.predict(X_full)
y_predict10 = model_svm_10.predict(X_10)
acc = [metrics.accuracy_score(y_true, y_predict), metrics.accuracy_score(y_true, y_predict10)]
recall = [metrics.recall_score(y_true, y_predict), metrics.recall_score(y_true, y_predict10)]

def recall_line(model, X_fears):
    recall = []
    for i in range(1,1+int(len(X_fears)/50)):
        X = X_fears.head(i*50)
        y_predict = model.predict(X)
        recall.append(metrics.recall_score([1]*i*50, y_predict))
    return recall

svm = recall_line(model_svm, X_fears[feature_all])
sxm_10 = recall_line(model_svm_10, X_fears[feature_10])
x = range(50, (len(svm)+1)*50, 50)
y1 = svm
y2 = sxm_10
fig, ax = plt.subplots()
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_minor_locator(MultipleLocator(0.01))
ax.set_xlabel('The number of drugs')
ax.set_ylabel('Recall rates')
plt.plot(x, y1, '-o', label='full-feature model')
plt.plot(x, y2, '-s', label='ten-feature model')
plt.legend()
plt.show()





