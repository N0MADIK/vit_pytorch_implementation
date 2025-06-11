import numpy as np


class LinearRegression: 

    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None 
        self.bias = None
    
    def fit(self, X: np.array, y: np.array) -> None:
        n_samples, n_feats = X.shape

        #initialize parameters
        self.weights = np.zeros(n_feats)
        self.bias = 0

        #Gradient descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            #compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            #update params 
            self.weights -= self.lr * dw 
            self.bias -= self.lr * db

    def predict(self, X) -> np.array: 
        return np.dot(X, self.weights) + self.bias
        