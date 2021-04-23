import pandas as pd
import random
from autograd import elementwise_grad
import autograd.numpy as np
np.random.seed(42)

class Neural_Network():
    def __init__(self,num_layers,hidden_nodes,activation_list,data,iterations,alpha): # Optional Arguments: Type of estimator 
        self.num_layers=num_layers
        self.weights=[]
        self.bias=[]
        self.activation_list=activation_list
        self.hidden_nodes=hidden_nodes
        self.iterations=iterations
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

    def relu(self,x):
        return np.maximum(0,x)

    def softmax(self,x):
        nue=np.exp(-x)
        deno=np.sum(nue)
        return nue/deno

    def identity(self,x):
        return x

    def mse(self,y_pred, y_true):
        return np.mean(np.power(y_true-y_pred, 2))

    def predict(self, X):
        num_examples=X.shape[0]
        y_pred = []
        for i in range(num_examples):
            A = X[i]
            for layer in range(self.num_layers+1):
                input=A
                weight_layer=self.weights[layer]
                bias_layer=self.bias[layer]
                activation_layer=self.activation_list[layer]
                A=self.forward(input,weight_layer,bias_layer,activation_layer)
            y_pred.append(A)
        return y_pred

    def forward(self,X,weight_layer,bias_layer,activation_layer):
        z=np.dot(X,weight_layer) + bias_layer
        self.Z.append(z)
        if(activation_layer=="sigmoid"):
            A=self.sigmoid(z)
        if(activation_layer=="Relu"):
            A=self.relu(z)
        if(activation_layer=="identity"):
            A=self.identity(z)
        if(activation_layer=="softmax"):
            A=self.softmax(z)
        self.a.append(A)
        return A
    
    def backward(self,dcda,weight_layer,bias_layer,activation_layer,layer):
        if(activation_layer=="sigmoid"):
            grad = elementwise_grad(self.sigmoid)  ##
        elif(activation_layer=="Relu"):
            grad = elementwise_grad(self.relu)
        elif(activation_layer=="identity"):
            grad = elementwise_grad(self.identity)
        dadz = grad(self.Z[layer])
        dcdz=dadz*dcda
        dcdw=np.dot(self.a[layer].T,dcdz)
        dcdb=dcdz
        dcda= np.dot(dcdz, weight_layer.T) 
        weight_layer-=self.alpha*dcdw
        bias_layer-=self.alpha*dcdb
        return dcda


    def train(self, x_train, y_train):
        self.m = len(x_train)
        loss=[]
        for k in range(self.iterations):
            e = 0
            for j in range(self.m):
                sample=x_train[j]
                sample=sample.reshape((1,sample.shape[0]))
                self.a=[sample]
                # forward propagation
                output = x_train[j]
                for layer in range(self.num_layers+1):
                    input=output
                    output = self.forward(input,self.weights[layer],self.bias[layer],self.activation_list[layer])
                e += self.mse(output, y_train[j])
                # backward propagation
                grad = elementwise_grad(self.mse)
                error = grad(output,y_train[j]) #dcda
                for layer in range(self.num_layers,-1,-1):
                    error=self.backward(error,self.weights[layer],self.bias[layer],self.activation_list[layer],layer)
            e = e/self.m
            loss.append(e)

     