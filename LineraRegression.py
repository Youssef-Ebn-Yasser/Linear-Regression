import numpy as np


class LineraRegression:
    def __init__(self,learning_rate :float = .02,n_iters:int = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters

        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples , n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters) :
            #prdict
            prediction = np.dot(X,self.weights)

            #compute - gradiant
            dw = (1/n_samples) * np.dot(X.T,(prediction - y))
            db = (1/n_samples) * np.sum(prediction - y)

            #step
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self,X):
        return  np.dot(X,self.weights) + self.bias