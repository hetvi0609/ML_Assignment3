import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from metrics import *
from sklearn.model_selection import train_test_split
from neuralnetworks import Neural_Network

X,y= load_boston(return_X_y=True)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

def scale_features(X):
    mean=np.mean(X)
    sigma=np.std(X)
    X_norm=(X-mean)/sigma
    return X_norm,mean,sigma

def transform_features(X, mean, sigma):
    X_norm = (X - mean)/sigma
    return X_norm





activation_list=["sigmoid","sigmoid","Relu","Relu","Relu","Relu","sigmoid"]
num_layers=6
hidden_nodes=[64,32,16,10,5,2,1]

print("3 fold cross validation")
#4 fold cross validation
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
X=pd.DataFrame(X)
y=pd.Series(y)
df=pd.concat([X,y],ignore_index=True,axis=1)
avg=0


k_fold=3#number of folds
min_rmse=100
for i in range(k_fold):
    train,test=five_fold_cross_validation(df,i,k_fold)
    X_train=train[train.columns[:-1]]
    y_train=train[train.columns[-1]]
    X_test=test[test.columns[:-1]]
    y_test=test[test.columns[-1]]
    temp = []
    for j in range(len(y_train)):
        temp.append([y_train[j]])
    y_train = np.array(temp)
    X_train,mean,sigma=scale_features(X_train)
    X_test=transform_features(X_test,mean,sigma)
    neural_network=Neural_Network(num_layers,hidden_nodes,activation_list,X_train,100,0.3) 
    neural_network.train(X_train,y_train)
    y_hat=neural_network.predict(X_test)
    y_hat = np.max(y_hat, axis=1)

    y_hat=pd.Series(y_hat)
    y_test=pd.Series(y_test)

    acc=rmse(y_hat,y_test)
    if(acc<min_rmse):
        best_model=neural_network
        min_rmse=acc
        X_train_best=X_train
        y_train_best=y_train
        X_test_best=X_test
        y_test_best=y_test
    print("-----------")
    print("rmse of",i+1,"th fold:",acc)
    avg=avg+acc
average_accuaracy=avg/k_fold
print("This is the average accuracy",average_accuaracy)
