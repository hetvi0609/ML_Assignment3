import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from metrics import *
from logisticregression import LogisticRegressor
from sklearn.datasets import load_breast_cancer
warnings. filterwarnings("ignore")
np.random.seed(42)
           
           
data = load_breast_cancer()
X,y=pd.DataFrame(data.data),pd.Series(data.target)
#Splitting dataset into 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test=X_test.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)
X_train=X_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)           


logistic_reg_l1=LogisticRegressor(type="L1")
logistic_reg_l1.fit_autograd(X_train,y_train)
logistic_reg_l1.plot()
y_hat=logistic_reg_l1.predict(X_test)
y_hat=pd.DataFrame(y_hat)
y_hat=y_hat.squeeze()
print("This is my L1-regularized logistic regression")
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y_test.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))



logistic_reg_l2=LogisticRegressor(type="L2")
logistic_reg_l2.fit_autograd(X_train,y_train)
logistic_reg_l2.plot()
y_hat=logistic_reg_l2.predict(X_test)
y_hat=pd.DataFrame(y_hat)
y_hat=y_hat.squeeze()
print("This is my L2-regularized logistic regression")
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y_test.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))



print("3 fold cross validation")
#5 fold cross validation
def five_fold_cross_validation(dataset,i,k_fold):
    n=len(dataset)//k_fold
    #selecting part of dataset
    test=dataset[n*i:n*(i+1)]
    test=test.reset_index(drop=True)
    #when first and last part of dataset are test set
    if(i==0):
        train=dataset[n*i+1:]
    elif(i==(k_fold-1)):
        train=dataset[:n*i]
    else:
        train_left=dataset[:n*i]
        train_right=dataset[n*(i+1):]
        train=pd.concat([train_left,train_right],axis=0)
    train=train.reset_index(drop=True)
    return train,test

df=pd.concat([X,y],ignore_index=True,axis=1)
k_fold=3#number of folds
print("Nested cross validation")
for i in range(k_fold):
    #cross validation for splitting val+train and test set
    trainplusval,test=five_fold_cross_validation(df,i,k_fold)
    X_test=test[test.columns[:-1]]
    y_test=test[test.columns[-1]]
    max_accuracy=0
    best_regressor=None
    optinum_lambda=0
    #iterating for 1-6 depth size
    for l in range(1,20,3):
        avg=0
        for j in range(k_fold):
             #cross validation for splitting train and validation set
            train,val_data=five_fold_cross_validation(trainplusval,j,k_fold)
            X_train=train[train.columns[:-1]]
            y_train=train[train.columns[-1]]
            #validation set
            X_val_data=val_data[val_data.columns[:-1]]
            y_val_data=val_data[val_data.columns[-1]]
            logistic_reg=LogisticRegressor(type=None)
            logistic_reg.fit_autograd(X_train,y_train,lambda_coeff=l/10000)
            y_val_hat=logistic_reg.predict(X_val_data)
            #checking accuracy on validation set
            acc=accuracy(y_val_hat,y_val_data)
            avg=avg+acc
        average_accuaracy=avg/(k_fold-1)
        if(average_accuaracy>max_accuracy):
            max_accuracy=average_accuaracy
            optinum_lambda=l/10000
            best_regressor=logistic_reg
    #predicting on best model with optimum depth
    y_hat=best_regressor.predict(X_test)
    acc_final=accuracy(y_hat,y_test)
    print("-------")
    print("Accuracy of",i+1,"th fold:",acc_final,"Optimum depth is",optinum_lambda)