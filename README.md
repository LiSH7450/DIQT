# DIQT
This repository is to provide the dataset and code to the article 'Unravelling structural alerts in marketed drugs for improving adverse outcome pathway framework of drug-induced QT prolongation'. Please note that this code is ONLY provided for academic use.

file list:

    1.ModelBuilding.py : The code for model building with four methods.
  
    2.freq_analysis.py : The code for method comparison.
      
    3.shap_.py : The code for feature selection using SHAP.
    
    4.svm_10Building.py : The code for svm model based on filtered features.
    
    5.recall_line.py : The code for comparing the results of two SVM models based on an external test set before and after feature selection.
    
folder list:

    1.Data : The data  we used in model building.
  
Note：

    1.The code for ModelBuilding.py yield some result files neventually. These files need to be further analyzed using freq_analysis.py .
  
    2.Data ,code need to be placed under the same path.
  
Python model requirement:

    1.xgboost
    
    2.sklearn
    
    3.pandas
    
    4.numpy
    
    5.joblib
    
    6.seaborn
    
    7.matplotlib
    
    8.shap
