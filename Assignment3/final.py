from autograd import numpy as np, elementwise_grad
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
        


    
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            output = self.forwardPass(output)
            result.append(output)
        print(result)
        return result 

    def activation(self,x):
        return 1/(1+np.exp(-x))


    def forwardPass(self,inputdata):
        self.input=inputdata
        for i in range(self.num_layers+1):
            self.output = np.dot(self.input,self.weights[i])+self.bias[i]
            self.Z.append(self.output)
            self.input = self.activation(self.output)
            self.a.append(self.input)
        return self.input
            
    def mse(self,y_pred, y_true):
        return np.mean(np.power(y_true-y_pred, 2))


    def train(self, x_train, y_train):
        # sample dimension first
        samples = len(x_train)
        # training loop
        for k in range(self.iterations):
            #print(i)
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                output = self.forwardPass(output)

                # compute loss (for display purpose only)
                err += self.mse(output, y_train[j])
                # backward propagation
                agrad = elementwise_grad(self.mse)
                error = agrad(output,y_train[j]) #output = z
                
                for i in range(self.num_layers,-1,-1):
                    agrad = elementwise_grad(self.activation)
                    output_error=agrad(self.Z[i])*error
                    error = np.dot(output_error,self.weights[i].T) 
                    weightsError = np.dot(self.a[i].T,output_error) 

                    self.weights[i] -= self.alpha*weightsError 
                    self.bias[i] -= self.alpha*output_error

            # calculate average error on all samples
            err /= samples
            
            print('epoch %d/%d   error=%f' % (k+1, self.iterations, err))

     