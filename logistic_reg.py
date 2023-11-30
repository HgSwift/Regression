import numpy as np
import matplotlib.pyplot as plt
import scipy.io



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_descent(X, y, w, lr=0.1, num_iter=100):
    m = len(y)
    print(m)
    loss = np.empty((num_iter,1))
    accuracy = np.empty((num_iter,1))
    for i in range(num_iter):
        print(i)
        w = w - (lr/m) * (X.T @ (sigmoid(X @ w) - y)) 
        loss[i] = training_loss(X, y, w)
        y_pred = np.round(sigmoid(X @ w))
        accuracy[i] = float(sum(y_pred == y))/ float(len(y))
    return (loss, w)

def training_loss(X, Y, w):
    m = len(Y)
    h = sigmoid(X @ w)
    e = 1e-7 # I swear I saw this somewhere but I don't know where
    loss = (1/m)*(((-Y).T @ np.log(h + e))-((1-Y).T @ np.log(1-h + e)))
    print(loss)
    print(log_likelihood(X, Y, w))
    return loss

def log_likelihood(X, Y, w):
    scores = np.dot(X, w)
    ll = np.sum( Y*scores - np.log(1 + np.exp(scores)) )
    return ll

# def predict(X, params):
#     return np.round(sigmoid(X @ params))

def get_data(filename):
    mat = scipy.io.loadmat(filename)
    x = mat['X']
    y = mat['Y']
    return(x,y)

X, Y = get_data('test_data.mat')

Y = Y[:,np.newaxis]



m = len(Y)

X = np.hstack((np.ones((m,1)),X))
n = np.size(X,1)
params = np.zeros((n,1))

# iterations = 1500
# learning_rate = 0.03

#initial_cost = compute_cost(X, y, params)

#print("Initial Cost is: {} \n".format(initial_cost))

params_optimal = gradient_descent(X, Y, params)
y_pred = np.round(sigmoid(X @ params))
score = float(sum(y_pred == Y))/ float(len(Y))

print("Optimal Parameters are: \n", params_optimal, "\n")