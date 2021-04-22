
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
        self.a=[data]
        self.alpha=alpha
        self.Z=[]
        for i in range(self.num_layers+1):
            if(i==0):
                self.weights.append(np.random.randn(self.hidden_nodes[i],data.shape[1]))
                self.bias.append(np.random.randn(1,self.hidden_nodes[i]))  
                
            else:
                self.weights.append(np.random.randn(self.hidden_nodes[i],self.hidden_nodes[i-1]))
                self.bias.append(np.random.randn(1,self.hidden_nodes[i]))  

    def Relu(self,x):
        return np.maximum(0,x)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def identity(self,x):
        return x

    def softmax(self,x):
        e_x = np.exp(x )
        return e_x / np.array(e_x.sum(axis=0)).T

    def tanh(self,x):
        return np.tanh(x)


    def mse(self,y_hat,y):
        return np.mean(np.power(y-y_hat,2))
    
    def cross_entropy_loss(self,y_hat,y):
        loss=-np.sum(np.multiply(y, np.log(y_hat)) +  np.multiply(1-y, np.log(1-y_hat)))/len(y)
        loss=np.squeeze(loss)
        return loss

    def forward(self,data):
        for i in range(self.num_layers+1):
            if(i==0):
                z=np.dot(data,self.weights[i].T)+self.bias[i]
                self.Z.append(z)
            else:
                z=np.dot(gz,self.weights[i].T)+self.bias[i]
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
            
            elif(self.activation_list[i]=="tanh"):
                gz=self.tanh(z)
                self.a.append(gz)

        return gz

    def train(self,X,y):

        print("Training")
        loss=[]
        for i in range(self.iterations):
            gradient=elementwise_grad(self.mse)
            y_hat=self.forward(X)
            dcda=gradient(y_hat,y)
            loss.append(self.mse(y_hat,y))
            for j in range(self.num_layers,-1,-1):
                #print(j)
                if(self.activation_list[j]=="sigmoid"):
                    gradient_act=elementwise_grad(self.sigmoid)
                    dadz=gradient_act(self.Z[j]) 
                    dcdz=(dcda*dadz)
                    dcdw=np.dot((dcdz.T),self.a[j])
                    dcdb=np.sum(dcdz,axis=0, keepdims=True)

                elif(self.activation_list[j]=="Relu"):
                    gradient_act=elementwise_grad(self.Relu)
                    dadz=gradient_act(self.Z[j]) 
                    dcdz=(dcda*dadz)
                    dcdw=np.dot((dcdz.T),self.a[j])
                    dcdb=np.sum(dcdz,axis=0, keepdims=True)

                elif(self.activation_list[j]=="softmax"):
                    gradient_act=elementwise_grad(self.softmax)
                    dadz=gradient_act(self.Z[j]) 
                    dcdz=(dcda*dadz)
                    dcdw=np.dot((dcdz.T),self.a[j])
                    dcdb=np.sum(dcdz,axis=0, keepdims=True)
                
                elif(self.activation_list[j]=="identity"):
                    gradient_act=elementwise_grad(self.identity)
                    dadz=gradient_act(self.Z[j]) 
                    dcdz=(dcda*dadz)
                    dcdw=np.dot((dcdz.T),self.a[j])
                    dcdb=np.sum(dcdz,axis=0, keepdims=True)
                
                elif(self.activation_list[j]=="tanh"):
                    gradient_act=elementwise_grad(self.tanh)
                    dadz=gradient_act(self.Z[j]) 
                    dcdz=(dcda*dadz)
                    dcdw=np.dot((dcdz.T),self.a[j])
                    dcdb=np.sum(dcdz,axis=0,keepdims=True)
                

                dcda=np.dot(dcdz,self.weights[j])
                self.weights[j]=self.weights[j]-self.alpha*dcdw
                self.bias[j]=self.bias[j]-self.alpha*dcdb

    def predict(self,X):
        # print(self.weights)
        y_hat=self.forward(X)
        return y_hat

                
                


                
            


                

                




