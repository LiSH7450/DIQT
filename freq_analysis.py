# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

df_XGB_result = pd.read_csv('df_predict_XGB.csv')
df_RF_result = pd.read_csv('df_predict_RF.csv')
df_svm_result = pd.read_csv('df_predict_svm.csv')
df_LR_result = pd.read_csv('df_predict_LR.csv')
df_predict_permutation_SVM_Y = pd.read_csv('df_predict_SVM_permu_Y.csv')
df_permutation_SVM_Y = pd.read_csv('df_permu_Y_SVM.csv')
df_predict_permutation_RF_X = pd.read_csv('df_predict_SVM_permu_X.csv')


def statistic_print(method, data, score_name):
    print(score_name, end=':')
    print(eval('np.' + method + '(data)'))


def freq_analysis(result_df, name, permutation='0'):
    y = result_df.iloc[:, 1].values
    recall_score = []
    precision_score = []
    acc = []
    MCC = []
    f1_score = []
    BACC = []
    auc = []
    AP = []
    SE = []
    SP = []
    score_dict = {}
    for i in range(2, 1002):
        if len(permutation) != 1:
            y = permutation.iloc[:, i].values
        else:
            y = result_df.iloc[:, 1].values
        y_predict = result_df.iloc[:, i].values
        acc.append(metrics.accuracy_score(y, y_predict))
        precision_score.append(metrics.precision_score(y, y_predict))
        recall_score.append(metrics.recall_score(y, y_predict))
        MCC.append(metrics.matthews_corrcoef(y, y_predict))
        f1_score.append(metrics.f1_score(y, y_predict))
        BACC.append(metrics.balanced_accuracy_score(y, y_predict))
        auc.append(metrics.roc_auc_score(y, y_predict))
        AP.append(metrics.average_precision_score(y, y_predict))
        tn, fp, fn, tp = metrics.confusion_matrix(y, y_predict, labels=[0, 1]).ravel()
        se = tp / float(tp + fn)
        SE.append(se)
        sp = tn / float(tn + fp)
        SP.append(sp)

    score_dict["acc"] = acc;
    score_dict["recall_score"] = recall_score;
    score_dict["precision_score"] = precision_score;
    score_dict["MCC"] = MCC;
    score_dict["BACC"] = BACC;
    score_dict["f1_score"] = f1_score;
    score_dict["AUC"] = auc;
    score_dict["AP"] = AP;
    score_dict["SE"] = SE;
    score_dict["SP"] = SP

    for method in ['mean', 'var', 'median', 'std']:
        print(method)
        for key, value in score_dict.items():
            statistic_print(method, value, key)
    return score_dict


'result'
XGB_score_dict = freq_analysis(df_XGB_result, 'XGB')

RF_score_dict = freq_analysis(df_RF_result, 'RF')

LR_score_dict = freq_analysis(df_LR_result, 'LR')

SVM_score_dict = freq_analysis(df_svm_result, 'SVM')


# RF_permu_X_score_dict = freq_analysis(df_predict_permutation_RF_X, 'RF_permu_X')

SVM_permu_Y_score_dict = freq_analysis(df_predict_permutation_SVM_Y, 'SVM_permu_Y', df_permutation_SVM_Y)

list1 = []
for key in SVM_score_dict:
    list1.append(key)
XGB = []
RF = []
LR = []
SVM = []
perm_y = []

for i in list1:
    XGB.append(np.mean(XGB_score_dict[i]))
    RF.append(np.mean(RF_score_dict[i]))
    LR.append(np.mean(LR_score_dict[i]))
    SVM.append(np.mean(SVM_score_dict[i]))
    perm_y.append(np.mean(SVM_permu_Y_score_dict[i]))
    # FEARS.append(np.mean(fears[i]))

figire_data = pd.DataFrame(list(zip(SVM, XGB, RF, LR, perm_y)))
figire_data.index = list1
figire_data.columns = ['SVM', 'XGB', 'RF', 'LR', 'perm_y']
figire_data.plot.bar()
print(figire_data)
plt.show()