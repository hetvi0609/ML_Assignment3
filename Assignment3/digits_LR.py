import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from MulticlassLR import LogisticRegressor
from sklearn import metrics
from metrics import *
import seaborn as sns
from sklearn.decomposition import PCA

digits = load_digits()
X=digits.data
y=digits.target


print("4 fold cross validation")
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


k_fold=4#number of folds
max_acc=0
for i in range(k_fold):
    train,test=five_fold_cross_validation(df,i,k_fold)
    X_train=train[train.columns[:-1]]
    y_train=train[train.columns[-1]]
    X_test=test[test.columns[:-1]]
    y_test=test[test.columns[-1]]
    logistic_unreg=LogisticRegressor(type=None) 
    logistic_unreg.fit(X_train,y_train)
    y_hat=logistic_unreg.predict(X_test)
    acc=accuracy(y_hat,y_test)
    if(acc>max_acc):
        best_model=logistic_unreg
        max_acc=acc
        X_train_best=X_train
        y_train_best=y_train
        X_test_best=X_test
        y_test_best=y_test
    print("-----------")
    print("Accuracy of",i+1,"th fold:",acc)
    avg=avg+acc
average_accuaracy=avg/k_fold
print("This is the average accuracy",average_accuaracy)


best_model.fit(X_train_best,y_train_best)
y_hat=best_model.predict(X_test_best)
print("This is best accurray of unregularized multiclass logisticregression")
print('Accuracy: ', accuracy(y_hat, y_test_best))
for cls in y_test.unique():
    print('Precision: ', precision(y_hat, y_test_best, cls))
    print('Recall: ', recall(y_hat, y_test_best, cls))

cm=metrics.confusion_matrix(y_test_best,y_hat)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy(y_hat, y_test_best))
plt.title(all_sample_title, size = 15)
plt.show()

#PCA
pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)


plt.figure()
plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Accent', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()
