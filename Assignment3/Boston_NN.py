import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from metrics import *
from sklearn.model_selection import train_test_split
from NN import Neural_Network
from sklearn.preprocessing import MinMaxScaler
np.random.seed(42)

X,y= load_boston(return_X_y=True)
X=pd.DataFrame(X)
y=pd.DataFrame(y)
df=pd.concat([X,y],axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
multiplied_by = scaler.scale_[12]
added = scaler.min_[12]

X=df[:,:-1]
y=df[:,-1]
activation_list=["Relu","sigmoid","Relu","sigmoid","Relu","sigmoid","sigmoid"]
num_layers=6
hidden_nodes=[50,100,50,30,20,2,1]

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

    X_train=np.array(X_train)    
    X_test=np.array(X_test)
    neural_network=Neural_Network(num_layers,hidden_nodes,activation_list,X_train,30,1) 
    neural_network.train(X_train,y_train)
    y_pred=neural_network.predict(X_test)
    #as the data was scaled it needs to be scaled
    y_hat = []
    for j in y_pred:
        y_hat.append((np.max(j)-added)/multiplied_by)
    y_test=(y_test-added)/multiplied_by
    y_test=pd.Series(y_test)
    y_hat=pd.Series(y_hat)

    acc=rmse(y_hat,y_test)
    if(acc<min_rmse):
        best_model=neural_network
        min_rmse=acc
        X_train_best=X_train
        y_train_best=y_train
        X_test_best=X_test
        y_test_best=y_test
    print("-----------")
    print("Rmse of",i+1,"th fold:",acc)
    avg=avg+acc
average_accuaracy=avg/k_fold
print("This is the average Rmse",average_accuaracy)