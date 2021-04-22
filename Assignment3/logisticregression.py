
#import numpy as np
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


    def scale_features(self,X):
        self.mean=np.mean(X)
        self.sigma=np.std(X)
        X_norm=(X-self.mean)/self.sigma
        return X_norm

    def sigmoid(self,z):
        sigmoid=1/(1+np.exp(-z))
        return sigmoid

    def cost(self,theta,X,y):
        X=np.array(X)
        y=np.array(y)
        h=self.sigmoid(np.dot(X,theta))
        a=-1*np.dot((y.T),np.log(h))
        b=-1*np.dot((1-y).T,np.log(1-h))
        if(self.type=="L2"):
            lambda_reg=(self.l)*np.dot(theta,theta.T)/(2*len(y))
        elif(self.type=="L1"):
            lambda_reg=(self.l)*theta/(len(y))
        else:
            lambda_reg=0
        J=((a+b)/len(y))+lambda_reg
        return J

    def delta_cost(self,theta,X,y):
        h=self.sigmoid(np.dot(X,theta))        
        delta_J=(1/len(y))*np.dot(X.T,(h-y))
        return delta_J

    def fit(self, X, y,alpha=0.01,iterations=1000,lambda_coeff=0.0001):
        y=pd.DataFrame(y)
        self.max_iter=iterations
        self.l=lambda_coeff
        #X_train=self.scale_features(X)
        X_train=X
        m=len(y)
        n=X_train.shape[1]
        theta=np.zeros((n,1))
        J=np.zeros((self.max_iter,1))
        for i in range(self.max_iter):
            J[i]=self.cost(theta,X_train,y)
            delta_J=self.delta_cost(theta,X_train,y)
            theta=theta-(alpha*delta_J)
        self.theta=theta
        self.J=J
        
    def fit_autograd(self, X, y,alpha=0.001,iterations=3000,lambda_coeff=0.0001):
        y=pd.DataFrame(y)
        self.max_iter=iterations
        X_train=X

        m=len(y)
        n=X.shape[1]
        #theta=np.zeros((n,1))
        theta=np.random.randn(n,1)
        J=np.zeros((self.max_iter,1))
        gradient = elementwise_grad(self.cost)
        for i in range(self.max_iter):
            J[i]=self.cost(theta,X_train,y)
            delta_J=gradient(theta,X_train,y)
            theta=theta-(alpha*delta_J)

        #print(theta)
        self.theta=theta
        self.J=J     
        


        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        

    def predict(self, X):
        #X_train=self.scale_features(X)
        X_train=X
        prob=(self.sigmoid(np.dot(X_train,self.theta)))
        predict=np.zeros((prob.shape[0],1))
        class_0=(prob<0.5)
        class_1=(prob>=0.5)
        predict[class_0]= 0
        predict[class_1]=1
        predict=pd.DataFrame(predict)
        predict=predict.squeeze()
        return predict

        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """



        

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.
        Figure 2 should also create a decision surface by combining the individual estimators
        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
        This function should return [fig1, fig2]
        """
        num_iter=[]
        for i in range(0,self.max_iter):
            num_iter.append(i)
        plt.plot(num_iter,self.J)
        plt.show()
    

    def Decision_boundary(self,X,y):
        df=X
        X=np.array(X)
        # Plot the data with the 
        theta=self.theta
        print(theta)
        # Logistic Regression Decision Boundary

        # Create a plot set
        plot_x = np.asarray([X.T[1].min()+1, X.T[1].max()-3])

        # Calculate the Decision Boundary
        plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])

        for i in range(len(y)):
            if y[i]==0.0:
                c = 'y'
                m = u'o'
            if y[i]==1.0:
                c = 'black'
                m = u'+'
            plt.scatter(X.T[1][i], X.T[2][i], color=c, marker=m) 
            
        # Plot the Decision Boundary (red line)
        plt.plot(plot_x, plot_y, color='red')

        # Put labels
        plt.xlabel(df[df.columns[1]])
        plt.ylabel(df[df.columns[2]])
        plt.show()
