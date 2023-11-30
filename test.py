import torch
import numpy as np
from torch.utils.data import TensorDataset
import hw1_utils as utils
import matplotlib.pyplot as plt


X, Y = utils.load_reg_data()
def f(x, params):
    m, c= params
    return m*x + c
    
def mse(preds, targets): 
    return ((preds-targets)**2).mean()

params = torch.zeros(2, 2).requires_grad_()
print(params)
lr = 0.01
w = torch.zeros(1, 2)
print(w)
def apply_step(params, w):
  preds = f(X, params)
  loss = mse(preds, Y)
  loss.backward()
  params.data -= lr * params.grad.data
  params.grad = None
  w = torch.cat((w ,params.data),0)
  return w
for i in range(10):
    preds = f(X, params)
    loss = mse(preds, Y)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    w = torch.cat((w ,params.data),0)
print(w)
#print(torch.cat((X, apply_step(params)),1))
#plt.scatter(X.numpy(), apply_step(params).detach().numpy(), c='#ff0000', marker=".")
#plt.scatter(X[:,0].numpy(), Y.numpy())
#plt.show()
'''
for i in range(500):
    preds = f(X, params)
    loss = mse(preds, Y)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    #print(params)
    print(f"Epoch {i}/50: Loss: {loss}")
    #plt.scatter(X[:,0].numpy(), Y.numpy())
    #plt.show()

'''