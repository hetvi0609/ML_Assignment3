import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from metrics import *
from MulticlassLR import LogisticRegressor
warnings. filterwarnings("ignore")

np.random.seed(42)

N = 30
M=5
P = 3
X = pd.DataFrame(np.random.randn(N, M))
y = pd.Series(np.random.randint(P, size = N))            

def scale_features(X):
    mean=np.mean(X)
    sigma=np.std(X)
    X_norm=(X-mean)/sigma
    return X_norm,mean,sigma

def transform_features(X, mean, sigma):
    X_norm = (X - mean)/sigma
    return X_norm


X_train,mean,sigma=scale_features(X)
X_test=transform_features(X,mean,sigma)

logistic_unreg=LogisticRegressor(type=None)
logistic_unreg.fit(X_train,y)
logistic_unreg.plot()

y_hat=logistic_unreg.predict(X_test)

print("This is my unregularized logistic regression")
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))


logistic_unreg=LogisticRegressor(type=None)
logistic_unreg.fit_autograd(X_train,y)
logistic_unreg.plot()

y_hat=logistic_unreg.predict(X_test)

print("This is my unregularized logistic regression with autograd")
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))


