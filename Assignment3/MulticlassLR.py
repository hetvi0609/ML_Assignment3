import pandas as pd
import random
import matplotlib.pyplot as plt
import math
from autograd import elementwise_grad
import autograd.numpy as np

class LogisticRegressor():
    def __init__(self,type=None): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''

        self.type=None

    def softmax(self,thetas,sample,class_id):
        theta_k=thetas[class_id]
        sample_biased=sample[:-1]
        # print(theta_k.shape)
        # print(sample_biased.shape)
        neu=np.exp(np.dot(theta_k.T, sample_biased))
        deno=0
        for theta in thetas:
            deno+=np.exp(np.dot(theta.T,sample_biased))
        return neu/deno
    def calc_cross_entropy(self,thetas,sample):
        cross_entropy=0
        #num_classes=len(thetas)
        num_classes=len(self.diff_classes)
        for j in range(num_classes):
            #label=sample[-1]
            label=sample[-1]
            if(label==self.diff_classes[j]):
                softmax=self.softmax(thetas,sample,j)
                cross_entropy+=np.log(softmax)
        return -1*cross_entropy

    def calc_total_loss(self,thetas,data):
        total_cross_entropy=0
        for i in range(len(data)):
            sample=data[i]
            cross_entropy=self.calc_cross_entropy(thetas,sample)
            total_cross_entropy+=cross_entropy
        return total_cross_entropy/(len(data))


    def delta_cross_entropy(self,thetas,data,class_id):
        m=len(data)
        delta_J=np.array([0.0]*len(thetas[0]))
        for sample in data:
            softmax=self.softmax(thetas,sample,class_id)
            sample_biased=sample[:-1]
            label=sample[-1]
            if(label==self.diff_classes[class_id]):
                I=1
            else:
                I=0
            product=np.array(sample_biased)*(I-softmax)
            delta_J+=product
        return delta_J/m

    def total_delta(self,thetas,data):
        num_classes=len(thetas)
        n=len(data[0])-1
        delta_cost=np.zeros((len(self.diff_classes),n))
        for i in range(num_classes):
            delta_k=self.delta_cross_entropy(thetas,data,i)
            delta_cost[i,:]=(delta_k)
        return delta_cost

    def fit(self, X, y,alpha=0.001,iterations=10):
        df=pd.concat([X,y],axis=1, ignore_index=True)
        data=np.array(df)
        #data=data.tolist()
        y=np.array(y)
        y=y.tolist()
        self.diff_classes=list(set(y))
        self.max_iter=iterations
        self.alpha=alpha
        n=len(data[0])-1
        thetas=np.zeros((len(self.diff_classes),n))
        #thetas=theta.tolist()
        J=[]
        for i in range(self.max_iter):
            #print(i)
            J.append(self.calc_total_loss(thetas,data))
            delta_J=self.total_delta(thetas,data)
            thetas=thetas+(alpha*delta_J)
        self.theta=thetas
        self.J=J 
    

    def fit_autograd(self, X, y,alpha=0.001,iterations=10):
        df=pd.concat([X,y],axis=1, ignore_index=True)
        data=np.array(df)
        #data=data.tolist()
        y=np.array(y)
        y=y.tolist()
        self.diff_classes=list(set(y))
        self.max_iter=iterations
        self.alpha=alpha
        n=len(data[0])-1
        thetas=np.zeros((len(self.diff_classes),n))
        #thetas=theta.tolist()
        gradient = elementwise_grad(self.calc_total_loss)
        J=[]
        for i in range(self.max_iter):
            J.append(self.calc_total_loss(thetas,data))
            delta_J=gradient(thetas,data)
            thetas=thetas-(alpha*delta_J)
        self.theta=thetas
        self.J=J 
    
    def plot(self):
        #plotting cost
        num_iter=[]
        for i in range(0,self.max_iter):
            num_iter.append(i)
        plt.plot(num_iter,self.J)
        plt.show()

    def softmax_predict(self,thetas,sample,class_id):
        theta_k=thetas[class_id]

        sample_biased=sample
        neu=np.exp(np.dot(theta_k.T, sample_biased))
        deno=0
        for theta in thetas:
            deno+=np.exp(np.dot(theta.T,sample_biased))
        return neu/deno

    def predict(self,X):
        X=np.array(X)
        #X=X.tolist()
        y_hat=[]
        for sample in X:
            max=0
            predict=0
            for i in range(len(self.diff_classes)):
                val=self.softmax_predict(self.theta,sample, i)
                if(val>max):
                    max=val
                    predict=self.diff_classes[i]
            y_hat.append(predict)
        return pd.Series(y_hat)




        

