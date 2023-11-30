import torch
import hw1_utils as utils
import matplotlib.pyplot as plt
from torch.nn import functional as F

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else (otherwise, your w will be ordered differently than the
    reference solution's in the autograder)!!!
'''

# Problem Linear Regression
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (N x d FloatTensor): the feature matrix
        Y (N x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    
    NOTE: Prepend a column of ones to X. (different from slides!!!)
    '''
    X = torch.cat((torch.ones(X.size()[0], 1), X), 1)
    w = torch.zeros(X.size()[1], 1)
    for i in range(num_iter):
        prediction = torch.matmul(X, w)
        w = w - (1/X.size()[0])*lrate*torch.matmul(torch.transpose(X, 0, 1), (prediction-Y))
    return(w)
    pass

def linear_normal(X, Y):
    '''
    Arguments:
        X (N x d FloatTensor): the feature matrix
        Y (N x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    
    NOTE: Prepend a column of ones to X. (different from slides!!!)
    '''
    X = torch.cat((torch.ones(X.size()[0], 1), X), 1)
    X_inv = torch.pinverse(X)
    w = torch.matmul(X_inv, Y)
    return(w)
    pass


def plot_linear(X, Y):
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    #X, Y = utils.load_reg_data()
    #w = linear_normal(X,Y)
    w = linear_gd(X, Y, 0.005, 1000)
    print(w[0])
    out = torch.mul(X, w[1])
    out = torch.add(out, w[0])
    plt.scatter(X[:,0].numpy(), Y.numpy())
    plt.scatter(X.numpy(), out.numpy(), c='#ff0000', marker='.')
    plt.show()
    return(plt)
    pass

def log_grad(X, Y, p):

    test = torch.matmul((-1*Y),(torch.transpose(X, 0 , 1)))
    #top = Y*p*torch.exp(torch.matmul((-Y, )))
    #bottom = torch.add(1, torch.exp())
    #out = torch.divide(top,bottom)
# Problem Logistic Regression
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (N x d FloatTensor): the feature matrix
        Y (N x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    
    NOTE: Prepend a column of ones to X. (different from slides) 
    '''
    #Prepends a column of ones to X
    X = torch.cat((torch.ones(X.size()[0], 1), X), 1)
    w = torch.zeros(X.size()[1], 1)
    for i in range(num_iter):
        prediction = (torch.matmul(X, w))
        #print(prediction)
        w = w - (1/X.size()[0])*lrate*torch.matmul(torch.transpose(X, 0, 1), (prediction-Y))
        #w = w - (1/X.size()[0])*lrate*log_grad(X,Y,prediction)
        #print(w)
    return(w)
    pass


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_reg_data()
    w = linear_normal(X,Y)
    p = logistic(X, Y, 0.01, 1000)
    #print(w[0])
    out1 = torch.mul(X, w[1])
    out1 = torch.add(out1, w[0])
    out2 = torch.mul(X, p[1])
    out2 = torch.add(out2, p[0])
    plt.scatter(X[:,0].numpy(), Y.numpy())
    plt.scatter(X.numpy(), out1.numpy(), c='#ff0000', marker='.')
    plt.scatter(X.numpy(), out2.numpy(), c='#00ff00', marker='.')
    plt.show()
    return(plt)
    pass
X, Y = utils.load_reg_data()
print(X.shape)
print(Y.shape)
X = torch.tensor([[1.8], [2.0], [2.4], [2.5], [3.0], [3.5], [4.3], [5.0], [5.3], [6.0], [6.6], [6.8], [7.8], [7.9], [8.4], [8.9], [9.4]])
print(X.shape)
Y = torch.tensor([[8.9], [5.1], [11.9], [13.0], [4.8], [6.3], [7.5], [8.0], [6.5], [12.5], [8.0], [8.7], [10.0], [8.0], [7.5], [9.0], [8.2]])
Y.reshape(17, 1)
print(Y.shape)
plot_linear(X, Y)
#logistic_vs_ols()
#X = torch.rand(100, 6)
#print(linear_gd(X, Y, 0.01, 20))
#print(logistic(X, Y, 0.01, 20))
#print(linear_normal(X,Y))
#print(torch.cat((torch.ones(Y.size()), Y), 1))
#print(X.size())
#print(Y.size())
#print(torch.cat((X,Y), 1))
#plt.scatter(X[:,0].numpy(), Y.numpy())
#plt.show()
#print(torch.zeros(2).requires_grad_())
