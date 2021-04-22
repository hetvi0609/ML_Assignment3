import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from metrics import *
from logisticregression import LogisticRegressor
warnings. filterwarnings("ignore")
from sklearn.datasets import load_breast_cancer
np.random.seed(42)



data = load_breast_cancer()
X,y=pd.DataFrame(data.data),pd.Series(data.target)
#Splitting dataset into 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test=X_test.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)
X_train=X_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)

logistic_unreg=LogisticRegressor(type=None)
logistic_unreg.fit(X_train,y_train)
#logistic_unreg.plot()
y_hat=logistic_unreg.predict(X_test)
y_hat=pd.DataFrame(y_hat)
y_hat=y_hat.squeeze()
print("This is my unregularized logistic regression")
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
avg=0
#initializing the Logistic Regression
logistic_unreg=LogisticRegressor(type=None) 
k_fold=3#number of folds
for i in range(k_fold):
    train,test=five_fold_cross_validation(df,i,k_fold)
    X_train=train[train.columns[:-1]]
    y_train=train[train.columns[-1]]
    X_test=test[test.columns[:-1]]
    y_test=test[test.columns[-1]]
    logistic_unreg.fit(X_train,y_train)
    y_hat=logistic_unreg.predict(X_test)
    acc=accuracy(y_hat,y_test)
    print("-----------")
    print("Accuracy of",i+1,"th fold:",acc)
    avg=avg+acc
average_accuaracy=avg/k_fold
print("This is the average accuracy",average_accuaracy)

def scale_features(X):
    mean=np.mean(X)
    sigma=np.std(X)
    X_norm=(X-mean)/sigma
    return X_norm,mean,sigma

def transform_features(X, mean, sigma):
    X_norm = (X - mean)/sigma
    return X_norm



####Run the below code for decision boundary###

# print("Decision Boundary")
# X_train=X[X.columns[:4]]
# X_train=np.array(X_train)
# X_train,mean,sigma=scale_features(X_train)


# column=np.ones((len(X_train),1))
# X_train=np.append(column,X_train,axis=1)
# X_train=pd.DataFrame(X_train)
# print(X_train)
# logistic_unreg=LogisticRegressor(type=None)
# logistic_unreg.fit(X_train,y)
# logistic_unreg.plot()
# logistic_unreg.Decision_boundary(X_train,y)
# y_hat=logistic_unreg.predict(X_train)
# y_hat=pd.DataFrame(y_hat)
# y_hat=y_hat.squeeze()
# print("This is my unregularized logistic regression")
# print('Accuracy: ', accuracy(y_hat, y))
# for cls in y_test.unique():
#     print('Precision: ', precision(y_hat, y, cls))
#     print('Recall: ', recall(y_hat, y, cls))
