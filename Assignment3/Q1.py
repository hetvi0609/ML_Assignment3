import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from metrics import *
from logisticregression import LogisticRegressor
warnings. filterwarnings("ignore")

np.random.seed(42)

N = 30
M=5
P = 2
X = pd.DataFrame(np.random.randn(N, M))
y = pd.Series(np.random.randint(P, size = N))            


logistic_unreg=LogisticRegressor(type=None)
logistic_unreg.fit(X,y)
logistic_unreg.plot()
y_hat=logistic_unreg.predict(X)
print("This is my unregularized logistic regression")
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

logistic_unreg=LogisticRegressor(type=None)
logistic_unreg.fit_autograd(X,y)
logistic_unreg.plot()
y_hat=logistic_unreg.predict(X)
y_hat=pd.DataFrame(y_hat)
y_hat=y_hat.squeeze()
#y=y.squeeze() 
print("This is my unregularized logistic regression with autograd ")
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

