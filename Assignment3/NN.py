
#import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
from autograd import elementwise_grad
import autograd.numpy as np


class Neural_Network():
    def __init__(self,num_layers,hidden_nodes,activation_list,data,iterations,alpha): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.num_layers=num_layers
        self.weights=[]
        self.bias=[]
        self.weights_gradient=[]
        self.bias_gradient=[]
        self.activation_list=activation_list
        self.hidden_nodes=hidden_nodes
        self.iterations=iterations
        self.a=[]
        self.alpha=alpha
        self.Z=[]
        for i in range(self.num_layers+1):
            if(i==0):
                self.weights.append(np.random.randn(data.shape[1],self.hidden_nodes[i])-0.5)
                self.bias.append(np.random.randn(1,self.hidden_nodes[i])-0.5)  
                 
            else:
                self.weights.append(np.random.randn(self.hidden_nodes[i-1],self.hidden_nodes[i])-0.5)
                self.bias.append(np.random.randn(1,self.hidden_nodes[i])-0.5)

            

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))



    def mse(self,y_hat,y):
        print(y)
        print(np.mean(np.power(y-y_hat,2)))
        return np.mean(np.power(y-y_hat,2))

    def forward(self,data):
        for i in range(self.num_layers+1):
            if(i==0):
                z=np.dot(data,self.weights[i])+self.bias[i]
                self.Z.append(z)
            else:
                z=np.dot(gz,self.weights[i])+self.bias[i]
                self.Z.append(z)

            if(self.activation_list[i]=='Relu'):        
                gz=self.Relu(z)
                self.a.append(gz)
            elif(self.activation_list[i]=="identity"):
                gz=self.identity(z)
                self.a.append(gz)
            elif(self.activation_list[i]=="sigmoid"):
                gz=self.sigmoid(z)
                self.a.append(gz)
            elif(self.activation_list[i]=="softmax"):
                gz=self.softmax(z)
                self.a.append(gz)

        return gz

    def train(self,X,y):
        print("Training")
        loss=[]
        for k in range(self.iterations):
            err=0
            for j in range(len(X)):
                y_hat=self.forward(X[j])
                err+=self.mse(y_hat[0],y[j])
                #print(err)
                grad= elementwise_grad(self.mse)
                dcda=grad(y_hat,y[j])

                for i in range(self.num_layers,-1,-1):
                    if(self.activation_list[i]=="sigmoid"):
                        gradient_act=elementwise_grad(self.sigmoid)
                        dadz=gradient_act(self.Z[i])   
                        dcdz=(dadz*dcda)
                        dcdw=np.dot(self.a[i].T,dcdz)
                        dcdb=dcdz        
                    dcda=np.dot(dcdz,self.weights[i].T)
                    self.weights[i]=self.weights[i]-((self.alpha*dcdw)/len(y))
                    self.bias[i]=self.bias[i]-(self.alpha*dcdb/len(y))
            loss.append(err)
        print(loss)

    def predict(self,X):
        print(self.weights)
        y_hat=[]
        for j in range(len(X)):
            y_hat.append(self.forward(X[j]))
        print(y_hat)

                
                


                
            


                

                




