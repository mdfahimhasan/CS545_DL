#!/usr/bin/env python
# coding: utf-8

# # A1: Three-Layer Neural Network

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-2">Example Results</a></span></li><li><span><a href="#Discussion" data-toc-modified-id="Discussion-3">Discussion</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will start with code from lecture notes 04 and add code to do the following. You will implement and apply a neural network as in lecture notes 04 but now with an additional hidden layer.  The resulting three-layer network will consist of three weight matrices, `U`, `V` and `W`.
# 
# First, implement the forward pass to calculate outputs of each layer:
# 
# * Define functions `add_ones` and `rmse` by copying it from the lecture notes.
# * Define function `forward_layer1` with two arguments, the input `X` and the first layer's weights `U`. It calculates and returns the output, `Zu`, of the first layer, using the `tanh` activation function.
# * Define function `forward_layer2` with two arguments, the input `Zu` and the second layer's weights `V`. It calculates and returns the output, `Zv`, of the second layer, using the `tanh` activation function.
# * Define function `forward_layer3` with two arguments, the input `Zv` and the third layer's weights `W`. It calculates and returns the output, `Y`, of the third layer as just the weighted sum of the inputs, without an activation function.
# * Define function `forward` with four arguments, the input `X` to the network and the weight matrices, `U`, `V` and `W` of the three layers. It calls the above three functions and returns the outputs of all layers, `Zu`, `Zv`, `Y`.
# 
# Now implement the backward pass that calculates `delta` values for each layer:
# 
# * Define function `backward_layer3` that accepts as arguments the target values `T` and the predicted values `Y` calculated by function `forward`. It calculates and returns `delta_layer3` for layer 3, which is just `T - Y`.
# * Define function `backward_layer2` that accepts as arguments `delta_layer3`, `W` and `Zv` and calculates and returns `delta` for layer 2 by back-propagating `delta_layer3` through `W`.
# * Define function `backward_layer1` that accepts as arguments `delta_layer2`, `V` and `ZU` and calculates and returns `delta` for layer 1 by back-propagating `delta_layer2` through `V`.
# * Define function `gradients` that accepts as arguments `X`, `T`, `Zu`, `Zv`, `Y`, `U`, `V`, and `W`, and calls the above three functions and uses the results to calculate the gradient of the mean squared error between `T` and `Y` with respect to `U`, `V` and `W` and returns those three gradients.
# 
# Now you can use `forward` and `gradients` to define the function `train` to train a three-layer neural network.
#           
# * Define function `train` that returns the resulting values of `U`, `V`, and `W` and the `X` and `T` standardization parameters.  Arguments are unstandardized `X` and `T`, the number of units in each of the two hidden layers, the number of epochs and the learning rate. This function standardizes `X` and `T`, initializes `U`, `V` and `W` to uniformly distributed random values between -0.1 and 0.1, and updates `U`, `V` and `W` by the learning rate times their gradients for `n_epochs` times as shown in lecture notes 04.  This function must call `forward`, `gradients` and `add_ones`.  It must also collect in a list called `rmses` the root-mean-square errors for each epoch between `T` and `Y`.
# 
#       def train(X, T, n_units_U, n_units_V, n_epochs, rho):
#           .
#           .
#           .
#           return rmses, U, V, W, X_means, X_stds, T_means, T_stds
# 
# Then we need a function `use` that calculates an output `Y` for new samples.  
# 
# * Define function `use` that accepts unstandardized `X`, standardization parameters, and weight matrices `U`, `V`, and `W` and returns the unstandardized output.
# 
#       def use(X, X_means, X_stds, T_means, T_stds, U, V, W):
#           .
#           .
#           .
#           Y = ....
#           return Y

# ## Example Results

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[74]:


def add_ones(S):
    return np.insert(S, 0, 1, axis=1)

def rmse(model_output, observed, observed_std):
    error = (observed - model_output) * observed_std
    return np.sqrt(np.mean(error ** 2))

def forward_layer1(X, U):
    Zu = np.tanh(add_ones(X) @ U)
    return Zu
    
def forward_layer2(Zu, V):
    Zv = np.tanh(add_ones(Zu) @ V)
    return Zv

def forward_layer3(Zv, W):
    Y = add_ones(Zv) @ W
    return Y

def forward(X, U, V, W):  
    Zu = forward_layer1(X, U) 
    Zv = forward_layer2(Zu, V)
    Y = forward_layer3(Zv, W)
    return Zu, Zv, Y

def backward_layer3(T, Y):  # to update W
    Dw = T-Y
    return Dw

def backward_layer2(Dw, W, Zv):  # to update V
    Dv = Dw @ W[1:, :].T * (1 - Zv **2)   
    return Dv

def backward_layer1(Dv, V, Zu):  # to update U
    Du = Dv @ V[1:, :].T * (1 - Zu **2)  
    return Du

def gradients(X, T, Zu, Zv, Y, U, V, W): 
    Dw = backward_layer3(T, Y)
    Dv = backward_layer2(Dw, W, Zv)
    Du = backward_layer1(Dv, V, Zu)
    
    grad_wrt_W = -add_ones(Zv).T @ Dw # have to work with bias added hidden layers
    grad_wrt_V = -add_ones(Zu).T @ Dv
    grad_wrt_U = -add_ones(X).T @ Du
    
    return grad_wrt_U, grad_wrt_V, grad_wrt_W

def train(X, T, n_units_U, n_units_V, n_epochs, rho):
    X_means = X.mean(axis=0)
    X_stds = X.std(axis=0)
    T_means = T.mean(axis=0)
    T_stds = T.std(axis=0)
    
    # standardizing
    XtrainS = (X - X_means) / X_stds
    TtrainS = (T - T_means) / T_stds
    
    # No. of samples and output
    n_samples, n_output = TtrainS.shape
     
    # weight initializing
    np.random.seed(0)
    U = np.random.uniform(-0.1, 0.1, size=(1 + XtrainS.shape[1], n_units_U)) / np.sqrt(add_ones(XtrainS).shape[1]) 
    V = np.random.uniform(-0.1, 0.1, size=(n_units_U + 1, n_units_V)) / np.sqrt(n_units_U + 1)
    W = np.random.uniform(-0.1, 0.1, size=(1 + n_units_V, n_output)) / np.sqrt(1 + n_units_V)
    
    # rmse list
    rmses = []
    
    for epoch in range(n_epochs):
        # forward propagation
        Zu, Zv, Y = forward(XtrainS, U, V, W)  # Y is the model predicted output here

        # backward propagation
        grad_wrt_U, grad_wrt_V, grad_wrt_W = gradients(XtrainS, TtrainS, Zu, Zv, Y, U, V, W)
        
        # updating weights
        W = W - rho * grad_wrt_W
        V = V - rho * grad_wrt_V
        U = U - rho * grad_wrt_U
        
        rmses.append(rmse(Y, TtrainS, T_stds))
        
    return rmses, U, V, W, X_means, X_stds, T_means, T_stds
    
def use(X, X_means, X_stds, T_means, T_stds, U, V, W):   
    # standardizing test data using X_means and X_stds of training data
    Xs = (X - X_means) / X_stds
    
    # forward pass
    Zu, Zv, Ys = forward(Xs, U, V, W)
    # no need of backward pass as we only need the output 
    # using the already trained U, V, W weigths from train() function
    
    # unstandardizing output generated by the model
    Y = Ys * T_stds + T_means
    
    return Y


# Add code cells here to define the functions above.  Once these are correctly defined, the following cells should run and produce the same results as those here.

# In[75]:


Xtrain = np.arange(4).reshape(-1, 1)
Ttrain = Xtrain ** 2

Xtest = Xtrain + 0.5
Ttest = Xtest ** 2

Xtrain.shape, Ttrain.shape, Xtest.shape, Ttest.shape


# In[76]:


U = np.array([[1, 2, 3], [4, 5, 6]])  # 2 x 3 matrix, for 2 inputs (include constant 1) and 3 units
V = np.array([[-1, 3], [1, 3], [-2, 1], [2, -4]]) # 2 x 3 matrix, for 3 inputs (include constant 1) and 2 units
W = np.array([[-1], [2], [3]])  # 3 x 1 matrix, for 3 inputs (include constant 1) and 1 output unit
U.shape, V.shape, W.shape


# In[77]:


X_means = np.mean(Xtrain, axis=0)
X_stds = np.std(Xtrain, axis=0)
Xtrain_st = (Xtrain - X_means) / X_stds
Xtrain_st


# In[78]:


T_means = np.mean(Ttrain, axis=0)
T_stds = np.std(Ttrain, axis=0)
Ttrain_st = (Ttrain - T_means) / T_stds
Ttrain_st


# In[79]:


Zu = forward_layer1(Xtrain_st, U)
Zu


# In[80]:


Zv = forward_layer2(Zu, V)
Zv


# In[81]:


Y = forward_layer3(Zv, W)
Y


# In[82]:


Zu, Zv, Y = forward(Xtrain_st, U, V, W)
print(f'{Zu=}')
print(f'{Zv=}')
print(f'{Y=}')


# In[83]:


delta_layer3 = backward_layer3(Ttrain_st, Y)
delta_layer3


# In[84]:


delta_layer2 = backward_layer2(delta_layer3, W, Zv)
delta_layer2


# In[62]:


delta_layer1 = backward_layer1(delta_layer2, V, Zu)
delta_layer1


# In[48]:


grad_U, grad_V, grad_W = gradients(Xtrain_st, Ttrain_st, Zu, Zv, Y, U, V, W)
print(f'{grad_U=}')
print(f'{grad_V=}')
print(f'{grad_W=}')


# In[49]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
Y


# In[50]:


error, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 10, 10, 1000, 0.05)


# In[17]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)
np.hstack((Ttrain, Y))


# In[18]:


plt.plot(error)
plt.xlabel('Epoch')
plt.ylabel('RMSE')


# Here is another example with a little more interesting data.

# In[19]:


np.random.seed(1)
n = 30
Xtrain = np.linspace(0., 20.0, n).reshape((n, 1)) - 10
Ttrain = 0.2 + 0.05 * (Xtrain + 10) + 0.4 * np.sin(Xtrain + 10) + 0.2 * np.random.normal(size=(n, 1))

Xtest = Xtrain + 0.1 * np.random.normal(size=(n, 1))
Ttest = 0.2 + 0.05 * (Xtest + 10) + 0.4 * np.sin(Xtest + 10) + 0.2 * np.random.normal(size=(n, 1))


# In[20]:


error, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 10, 10, 10000, 0.01)


# In[21]:


plt.plot(error)
plt.xlabel('Epoch')
plt.ylabel('RMSE')


# In[22]:


Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)


# In[23]:


plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.legend();


# In[24]:


error, U, V, W, X_means, X_stds, T_means, T_stds = train(Xtrain, Ttrain, 10, 10, 10000, 0.01)
Y = use(Xtrain, X_means, X_stds, T_means, T_stds, U, V, W)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(error)
plt.xlabel('Epoch')
plt.ylabel('RMSE')

plt.subplot(1, 2, 2)
plt.plot(Xtrain, Ttrain, label='Train')
plt.plot(Xtrain, Y, label='Test')
plt.xlabel('Input')
plt.ylabel('Target and Output')
plt.legend()
plt.tight_layout();


# Your plots will probably differ from these results, because you start with different random weight values.

# ## Discussion

# I had difficulties understaing the dimensions of the weights. Later, found that I have to add the bias-dimension to the weights. Also, the instructions says to use add_ones() function in the train function. If I do that, then the forward layers generate wrong dimension. Therefore, I had to use the add_ones() in the forward layer 1, 2, 3 functions.
# 
# The plots generated from the new data doesn't seem to match at all. The result between train and test data varies a lot, thought rhe rmse plot shows that the error was minimized with epochs.
# 
# **Easy parts:**
# - understanding overall concept of Neural network
# - standardizing
# - add_ones() and rmse_func() functions
# 
# **Hard parts:**
# - how matrix operation works in numpy and figuring out the dimensions of U, V, W
# - Understanding how the concept of calculus works in python matrix calculation
# - Understaning the backpropagation part
# 

# # Grading
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A1grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A1grader.tar) <font color="red">(updated August 28th)</font> and extract `A1grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 10 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  A perfect execution score from this grading script does not guarantee that you will receive a perfect execution score from the final grading script.
# 
# For the grading script to run correctly, you must first name this notebook as 'Lastname-A1.ipynb' with 'Lastname' being your last name, and then save this notebook.

# In[25]:


get_ipython().run_line_magic('run', '-i A1grader.py')


# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.
