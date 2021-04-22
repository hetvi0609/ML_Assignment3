import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from metrics import *
from neuralnetworks import Neural_Network
warnings. filterwarnings("ignore")
np.random.seed(42)

X=np.array([[1,0],[0,1],[1,1],[0,0]])

y=np.array([[1],[1],[0],[0]])

activation_list=["tanh","sigmoid"]
num_layers=1
hidden_nodes=[2,1]
neural_network=Neural_Network(num_layers,hidden_nodes,activation_list,X,100,0.3) 
neural_network.train(X,y)

print(neural_network.predict(X))
