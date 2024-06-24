#!/usr/bin/env python
# coding: utf-8

# # A2: NeuralNetwork Class

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Code-for-NeuralNetwork-Class" data-toc-modified-id="Code-for-NeuralNetwork-Class-2">Code for <code>NeuralNetwork</code> Class</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-3">Example Results</a></span></li><li><span><a href="#Application-to-Boston-Housing-Data" data-toc-modified-id="Application-to-Boston-Housing-Data-4">Application to Boston Housing Data</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will complete the implementation of the `NeuralNetwork` class, starting with the code included in the `04b` lecture notes.  Your implementation must 
# 
# 1. Allow any number of hidden layers, including no hidden layers specified by an empty list as `[]`. <font color='red'>Don't forget this case.</font>
# 2. Define `_forward(self, X)` and `_gradients(self, X, T` functions. `_forward` must return the output of the network, `Y`, in standardized form and create `self.Zs` as a list consisting of the input `X` and the outputs of all hidden layers. `_gradients` must return the gradients of the mean square error with respect to the weights in each layer. 
# 2. Your `train` function must standardize `X` and `T` and save the standardization parameters (means and stds) in member variables. It must append to `self.rmse_trace` the RMSE value for each epoch.  Initialize this list to be `[]` in the constructor to allow multiple calls to `train` to continue to append to the same `rmse_trace` list.
# 2. Your `use` function must standardize `X` and unstandardize the output.
# 
# See the following examples for more details.
# 
# Then apply your `NeuralNetwork` class to the problem of predicting the value of houses in Boston as described below.

# ## Code for `NeuralNetwork` Class

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.display as ipd  # for display and clear_output
import time


# In[8]:


# insert your NeuralNetwork class definition here.
class NeuralNetwork:
    
    pass


# In this next code cell, I add a new method to your class that replaces the weights created in your constructor with non-random values to allow you to compare your results with mine, and to allow our grading scripts to work well.

# In[10]:


def set_weights_for_testing(self):
    for W in self.Ws[:-1]:   # leave output layer weights at zero
        n_weights = W.shape[0] * W.shape[1]
        W[:] = np.linspace(-0.01, 0.01, n_weights).reshape(W.shape)
        for u in range(W.shape[1]):
            W[:, u] += (u - W.shape[1]/2) * 0.2
    # Set output layer weights to zero
    self.Ws[-1][:] = 0
    print('Weights set for testing by calling set_weights_for_testing()')

setattr(NeuralNetwork, 'set_weights_for_testing', set_weights_for_testing)


# ## Example Results

# Here we test the `NeuralNetwork` class with some simple data.  
# 

# In[11]:


X = np.arange(0, 10).reshape(-1, 1)
T = np.sin(X) + 0.01 * (X ** 2)
X, T


# In[12]:


plt.plot(X, T, '.-')


# In[13]:


n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [3, 2], n_outputs)
nnet


# In[14]:


nnet.n_inputs, nnet.n_hiddens_each_layer, nnet.n_outputs


# In[15]:


nnet.rmse_trace


# In[16]:


nnet.Ws


# In[17]:


nnet.set_weights_for_testing()


# In[18]:


nnet.Ws


# In[19]:


nnet.train(X, T, n_epochs=1, learning_rate=0.1)


# In[20]:


nnet.Zs


# In[21]:


print(nnet)


# In[22]:


nnet.X_means, nnet.X_stds


# In[23]:


nnet.T_means, nnet.T_stds


# In[24]:


[Z.shape for Z in nnet.Zs]


# In[25]:


nnet.Ws


# In[26]:


dir(nnet)


# In[27]:


def plot_data_and_model(nnet, X, T):
    plt.clf()        
    plt.subplot(2, 1, 1)
    plt.plot(nnet.rmse_trace)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')

    plt.subplot(2, 1, 2)
    Y = nnet.use(X)

    plt.plot(X, Y, 'o-', label='Y')
    plt.plot(X, T, 'o', label='T', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('T or Y')
    plt.legend();


# In[28]:


X = np.arange(0, 10).reshape(-1, 1)
# X = np.arange(0, 0.5, 0.05).reshape(-1, 1)
T = np.sin(X) + 0.01 * (X ** 2)

n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [10, 5], n_outputs)
nnet.set_weights_for_testing()

n_epochs = 5000
n_epochs_per_plot = 500

fig = plt.figure()
for reps in range(n_epochs // n_epochs_per_plot):
    plt.clf()
    nnet.train(X, T, n_epochs=n_epochs_per_plot, learning_rate=0.1)
    plot_data_and_model(nnet, X, T)
    ipd.clear_output(wait=True)
    ipd.display(fig)
    time.sleep(0.2)  # 0.2 seconds
ipd.clear_output(wait=True)


# In[29]:


X = np.arange(-2, 2, 0.05).reshape(-1, 1)
T = np.sin(X) * np.sin(X * 10)

n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [50, 10, 5], n_outputs)
nnet.set_weights_for_testing()

n_epochs = 50000
n_epochs_per_plot = 500

fig = plt.figure()
for reps in range(n_epochs // n_epochs_per_plot):
    plt.clf()
    nnet.train(X, T, n_epochs=n_epochs_per_plot, learning_rate=0.1)
    plot_data_and_model(nnet, X, T)
    ipd.clear_output(wait=True)
    ipd.display(fig)
    # time.sleep(0.01)  # 0.01 seconds
ipd.clear_output(wait=True)


# Your results will not be the same, but your code should complete and make plots somewhat similar to these.

# ## Application to Boston Housing Data

# Download data from [Boston House Data at Kaggle](https://www.kaggle.com/fedesoriano/the-boston-houseprice-data). Read it into python using the `pandas.read_csv` function.  Assign the first 13 columns as inputs to `X` and the final column as target values to `T`.  Make sure `T` is two-dimensional.

# Before training your neural networks, partition the data into training and testing partitions, as shown here.

# In[30]:


def partition(X, T, train_fraction):
    n_samples = X.shape[0]
    rows = np.arange(n_samples)
    np.random.shuffle(rows)

    n_train = round(n_samples * train_fraction)

    Xtrain = X[rows[:n_train], :]
    Ttrain = T[rows[:n_train], :]
    Xtest = X[rows[n_train:], :]
    Ttest = T[rows[n_train:], :]

    return Xtrain, Ttrain, Xtest, Ttest


# In[31]:


X = np.arange(20).reshape(-1, 1)
T = X * 2
np.hstack((X, T))  # np.hstack just to print X and T together in one array


# In[32]:


Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8)  


# In[33]:


np.hstack((Xtrain, Ttrain))


# In[34]:


np.hstack((Xtest, Ttest))


# Write and run code using your `NeuralNetwork` class to model the Boston housing data. Experiment with a variety of neural network structures (numbers of hidden layer and units) including no hidden layers, learning rates, and numbers of epochs. Show results for at least three different network structures, learning rates, and numbers of epochs for a total of at least 27 results.  Show your results in a `pandas` DataFrame with columns `('Structure', 'Epochs', 'Learning Rate', 'Train RMSE', 'Test RMSE')`.
# 
# Try to find good values for the RMSE on testing data.  Discuss your results, including how good you think the RMSE values are by considering the range of house values given in the data. 

# # Grading
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A2grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A2grader.tar) and extract `A2grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 20 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  
# 
# For the grading script to run correctly, you must first name this notebook as `Lastname-A2.ipynb` with `Lastname` being your last name, and then save this notebook.

# In[7]:


get_ipython().run_line_magic('run', '-i A2grader.py')


# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.
#!/usr/bin/env python
# coding: utf-8

# # A2: NeuralNetwork Class

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Code-for-NeuralNetwork-Class" data-toc-modified-id="Code-for-NeuralNetwork-Class-2">Code for <code>NeuralNetwork</code> Class</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-3">Example Results</a></span></li><li><span><a href="#Application-to-Boston-Housing-Data" data-toc-modified-id="Application-to-Boston-Housing-Data-4">Application to Boston Housing Data</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will complete the implementation of the `NeuralNetwork` class, starting with the code included in the `04b` lecture notes.  Your implementation must 
# 
# 1. Allow any number of hidden layers, including no hidden layers specified by an empty list as `[]`. <font color='red'>Don't forget this case.</font>
# 2. Define `_forward(self, X)` and `_gradients(self, X, T` functions. `_forward` must return the output of the network, `Y`, in standardized form and create `self.Zs` as a list consisting of the input `X` and the outputs of all hidden layers. `_gradients` must return the gradients of the mean square error with respect to the weights in each layer. 
# 2. Your `train` function must standardize `X` and `T` and save the standardization parameters (means and stds) in member variables. It must append to `self.rmse_trace` the RMSE value for each epoch.  Initialize this list to be `[]` in the constructor to allow multiple calls to `train` to continue to append to the same `rmse_trace` list.
# 2. Your `use` function must standardize `X` and unstandardize the output.
# 
# See the following examples for more details.
# 
# Then apply your `NeuralNetwork` class to the problem of predicting the value of houses in Boston as described below.

# ## Code for `NeuralNetwork` Class

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.display as ipd  # for display and clear_output
import time


# In[4]:


# insert your NeuralNetwork class definition here.
class NeuralNetwork:
    
    def __init__(self, n_inputs, n_hiddens_each_layer, n_outputs):
        
        self.n_inputs = n_inputs
        self.n_hiddens_each_layer = n_hiddens_each_layer
        self.n_outputs = n_outputs
        
        self.n_epochs = 0
        self.rmse = None
        self.rmse_trace = []
        
        self.Ws = []
        
        # Hidden Layers
        nh = 0
        ni = self.n_inputs
        for nh in self.n_hiddens_each_layer:
            self.Ws.append(np.random.uniform(-1, 1, size=(1 + ni, nh)) / np.sqrt(1 + ni))
            ni = nh

        # Output Layer
        ni = nh if nh > 0 else ni
        nu = self.n_outputs
        self.Ws.append(np.zeros((1 + ni, nu)))

        
    # def set_weights_for_testing(self):
    #     for W in self.Ws[:-1]:   # leave output layer weights at zero
    #         n_weights = W.shape[0] * W.shape[1]
    #         W[:] = np.linspace(-0.01, 0.01, n_weights).reshape(W.shape)
    #         for u in range(W.shape[1]):
    #             W[:, u] += (u - W.shape[1]/2) * 0.2
    #     print('Weights set for testing by calling set_weights_for_testing()')
            
    def __repr__(self):
        return 'NeuralNetwork({}, {}, {})'.format(self.n_inputs, self.n_hiddens_each_layer, self.n_outputs)
    
    def __str__(self):
        return self.__repr__() + ', trained for {} epochs with a final RMSE of {}'.format(self.n_epochs, self.rmse)
    
    def calc_rmse(self, Y, T):
        return np.sqrt(np.mean(((T - Y) * self.T_stds) ** 2))
    
    def train(self, X, T, n_epochs, learning_rate):

        n_samples, n_inputs = X.shape
        _, n_outputs = T.shape

        learning_rate = learning_rate / (n_samples * n_outputs)

        self.X_means = np.mean(X, axis=0)
        self.X_stds = np.std(X, axis=0)
        # X_stds[X_stds == 0] = 1
        self.T_means = np.mean(T, axis=0)
        self.T_stds = np.std(T, axis=0)

        X = (X - self.X_means) / self.X_stds
        T = (T - self.T_means) / self.T_stds

        # dW = [np.zeros_like(W) for W in self.Ws]
        # momentum = 0.0
        
        for epoch in range(n_epochs):

            Y = self._forward(X)
            self.rmse_trace.append(self.calc_rmse(Y, T))

            gradients = self._gradients(X, T)

            # for W_layer, grad_layer, dW_layer in zip(self.Ws, gradients, dW):
            #     dW_layer[:] = learning_rate * grad_layer + momentum * dW_layer
            #     W_layer -= dW_layer  # learning_rate * grad_layer

            for W_layer, grad_layer in zip(self.Ws, gradients):
                W_layer -= learning_rate * grad_layer

        self.n_epochs += n_epochs

        return self
    

    def use(self, X):
        # standardize X
        X = (X - self.X_means) / self.X_stds
        # Forward pass
        Y = self._forward(X)
        # Unstandardize output
        Y = Y * self.T_stds + self.T_means
        return Y
    
    def _add_ones(self, X):
        return np.insert(X, 0, 1, axis=1)

    def _forward(self, X):
        # X already standardized
        Z_previous_layer = X
        self.Zs = [X]
        for W_layer in self.Ws[:-1]:
            Z_previous_layer = np.tanh(self._add_ones(Z_previous_layer) @ W_layer)
            self.Zs.append(Z_previous_layer)  # save for gradient calculations
        # Output Layer
        Y = self._add_ones(Z_previous_layer) @ self.Ws[-1]
        self.Zs.append(Y)
        return Y
    
    def _gradients(self, X, T):
        # X and T already standardized and self.Z and self.Y already calculated
        Y = self.Zs[-1]
        delta = -(T - Y)
        # Backprop through weights starting with last layer, but not including first layer,
        #  and zip with outputs of layers starting with second to last layer.
        gradients = []
        # print('len(self.Ws[::-1])={} len(self.Z[::-1][1:])={}'.format(len(self.Ws[::-1]), len(self.Z[::-1][1:])))
        for W, Z in zip(self.Ws[::-1], self.Zs[:-1][::-1]):
            gradients.append(self._add_ones(Z).T @ delta)
            # print('W.shape={} Z.shape={} gradient.shape={}'.format(W.shape, Z.shape, self.gradients[-1].shape))
            delta = delta @ W[1:, :].T * (1 - Z**2)  # last delta is  not used
        gradients = gradients[::-1]
        return gradients


# In this next code cell, I add a new method to your class that replaces the weights created in your constructor with non-random values to allow you to compare your results with mine, and to allow our grading scripts to work well.

# In[5]:


def set_weights_for_testing(self):
    for W in self.Ws[:-1]:   # leave output layer weights at zero
        n_weights = W.shape[0] * W.shape[1]
        W[:] = np.linspace(-0.01, 0.01, n_weights).reshape(W.shape)
        for u in range(W.shape[1]):
            W[:, u] += (u - W.shape[1]/2) * 0.2
    # Set output layer weights to zero
    self.Ws[-1][:] = 0
    print('Weights set for testing by calling set_weights_for_testing()')

setattr(NeuralNetwork, 'set_weights_for_testing', set_weights_for_testing)


# ## Example Results

# Here we test the `NeuralNetwork` class with some simple data.  
# 

# In[6]:


X = np.arange(0, 10).reshape(-1, 1)
T = np.sin(X) + 0.01 * (X ** 2)
X, T


# In[7]:


plt.plot(X, T, '.-')


# In[8]:


n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [3, 2], n_outputs)
nnet


# In[9]:


nnet.n_inputs, nnet.n_hiddens_each_layer, nnet.n_outputs


# In[10]:


nnet.rmse_trace


# In[11]:


nnet.Ws


# In[12]:


nnet.set_weights_for_testing()


# In[13]:


nnet.Ws


# In[14]:


nnet.train(X, T, n_epochs=1, learning_rate=0.1)


# In[15]:


nnet.Zs


# In[16]:


print(nnet)


# In[17]:


nnet.X_means, nnet.X_stds


# In[18]:


nnet.T_means, nnet.T_stds


# In[19]:


[Z.shape for Z in nnet.Zs]


# In[20]:


nnet.Ws


# In[21]:


dir(nnet)


# In[22]:


def plot_data_and_model(nnet, X, T):
    plt.clf()        
    plt.subplot(2, 1, 1)
    plt.plot(nnet.rmse_trace)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')

    plt.subplot(2, 1, 2)
    Y = nnet.use(X)

    plt.plot(X, Y, 'o-', label='Y')
    plt.plot(X, T, 'o', label='T', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('T or Y')
    plt.legend();


# In[23]:


X = np.arange(0, 10).reshape(-1, 1)
# X = np.arange(0, 0.5, 0.05).reshape(-1, 1)
T = np.sin(X) + 0.01 * (X ** 2)

n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [10, 5], n_outputs)
nnet.set_weights_for_testing()

n_epochs = 5000
n_epochs_per_plot = 500

fig = plt.figure()
for reps in range(n_epochs // n_epochs_per_plot):
    plt.clf()
    nnet.train(X, T, n_epochs=n_epochs_per_plot, learning_rate=0.1)
    plot_data_and_model(nnet, X, T)
    ipd.clear_output(wait=True)
    ipd.display(fig)
    time.sleep(0.2)  # 0.2 seconds
ipd.clear_output(wait=True)


# In[24]:


X = np.arange(-2, 2, 0.05).reshape(-1, 1)
T = np.sin(X) * np.sin(X * 10)

n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [50, 10, 5], n_outputs)
nnet.set_weights_for_testing()

n_epochs = 50000
n_epochs_per_plot = 500

fig = plt.figure()
for reps in range(n_epochs // n_epochs_per_plot):
    plt.clf()
    nnet.train(X, T, n_epochs=n_epochs_per_plot, learning_rate=0.1)
    plot_data_and_model(nnet, X, T)
    ipd.clear_output(wait=True)
    ipd.display(fig)
    # time.sleep(0.01)  # 0.01 seconds
ipd.clear_output(wait=True)


# Your results will not be the same, but your code should complete and make plots somewhat similar to these.

# ## Application to Boston Housing Data

# Download data from [Boston House Data at Kaggle](https://www.kaggle.com/fedesoriano/the-boston-houseprice-data). Read it into python using the `pandas.read_csv` function.  Assign the first 13 columns as inputs to `X` and the final column as target values to `T`.  Make sure `T` is two-dimensional.

# Before training your neural networks, partition the data into training and testing partitions, as shown here.

# In[25]:


def partition(X, T, train_fraction):
    n_samples = X.shape[0]
    rows = np.arange(n_samples)
    np.random.shuffle(rows)

    n_train = round(n_samples * train_fraction)

    Xtrain = X[rows[:n_train], :]
    Ttrain = T[rows[:n_train], :]
    Xtest = X[rows[n_train:], :]
    Ttest = T[rows[n_train:], :]

    return Xtrain, Ttrain, Xtest, Ttest


# Write and run code using your `NeuralNetwork` class to model the Boston housing data. Experiment with a variety of neural network structures (numbers of hidden layer and units) including no hidden layers, learning rates, and numbers of epochs. Show results for at least three different network structures, learning rates, and numbers of epochs for a total of at least 27 results.  Show your results in a `pandas` DataFrame with columns `('Structure', 'Epochs', 'Learning Rate', 'Train RMSE', 'Test RMSE')`.
# 
# Try to find good values for the RMSE on testing data.  Discuss your results, including how good you think the RMSE values are by considering the range of house values given in the data. 

# In[26]:


import pandas
boston = pandas.read_csv('boston.csv')
boston = boston.to_numpy()
X = boston[:, :13]
T = boston[:, -1:]
X.shape, T.shape


# In[28]:


Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8)

nnet = NeuralNetwork(X.shape[1], [50, 50], T.shape[1])

nnet.train(Xtrain, Ttrain, 5000, 0.01)


# In[29]:


plt.plot(nnet.rmse_trace)


# In[35]:


def rmse(Y, T):
    return np.sqrt(np.mean((T - Y)**2))

n_inputs = Xtrain.shape[1]
n_outputs = Ttrain.shape[1]

results = []
for hiddens in [ [], [5], [50], [10, 10], [20, 20, 20] ]:
    for epochs in [10, 100, 500, 1000, 5000]:
        for lr in [0.001, 0.01, 0.1, 0.2]:

            nnet = NeuralNetwork(n_inputs, hiddens, n_outputs)
            nnet.train(Xtrain, Ttrain, epochs, lr)
            rmse_train = rmse(nnet.use(Xtrain), Ttrain)
            rmse_test = rmse(nnet.use(Xtest), Ttest)
            results.append([hiddens, epochs, lr, rmse_train, rmse_test])
            
            df = pandas.DataFrame(results, 
                                  columns=('Structure', 'Epochs', 'Learning Rate', 
                                           'Train RMSE', 'Test RMSE'))
            ipd.clear_output(wait=True)

            print(df.sort_values(by='Test RMSE', ascending=True).head(20))


# In[38]:


def rmse(Y, T):
    return np.sqrt(np.mean((T - Y)**2))

n_inputs = Xtrain.shape[1]
n_outputs = Ttrain.shape[1]

results = []
for hiddens in [ [] ]:
    for epochs in [10, 100, 500, 1000, 5000]:
        for lr in [0.001, 0.01, 0.1, 0.2]:

            nnet = NeuralNetwork(n_inputs, hiddens, n_outputs)
            nnet.train(Xtrain, Ttrain, epochs, lr)
            rmse_train = rmse(nnet.use(Xtrain), Ttrain)
            rmse_test = rmse(nnet.use(Xtest), Ttest)
            results.append([hiddens, epochs, lr, rmse_train, rmse_test])
            
            df = pandas.DataFrame(results, 
                                  columns=('Structure', 'Epochs', 'Learning Rate', 
                                           'Train RMSE', 'Test RMSE'))
            ipd.clear_output(wait=True)

            print(df.sort_values(by='Test RMSE', ascending=True).head(20))


# In[36]:


T.min(), T.max()


# In[37]:


2.8 / (50 - 5)


# The best Test RMSE achieved is 2.8.  This is only about 6% of the target range, so a very good result.  This was for the largest network tried ([20, 20, 20]), 1,000 epochs, and learning rate of 0.1.
# 
# I see some results showing a low Train RMSE of around 1.3, but for these cases the neural network is overfitting the training data, resulting in a higher Test RMSE.
# 
# Linear networks (with no hidden units) did not get lower than about 4.4 Test RMSE, for the highest number of epochs tested (5000) and highest learning rate tested (0.2).

# # Grading
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A2grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A2grader.tar) and extract `A2grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 20 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  
# 
# For the grading script to run correctly, you must first name this notebook as `Lastname-A2.ipynb` with `Lastname` being your last name, and then save this notebook.

# In[39]:


get_ipython().run_line_magic('run', '-i A2grader.py')


# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.
#!/usr/bin/env python
# coding: utf-8

# # A2: NeuralNetwork Class

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Code-for-NeuralNetwork-Class" data-toc-modified-id="Code-for-NeuralNetwork-Class-2">Code for <code>NeuralNetwork</code> Class</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-3">Example Results</a></span></li><li><span><a href="#Application-to-Boston-Housing-Data" data-toc-modified-id="Application-to-Boston-Housing-Data-4">Application to Boston Housing Data</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will complete the implementation of the `NeuralNetwork` class, starting with the code included in the `04b` lecture notes.  Your implementation must 
# 
# 1. Allow any number of hidden layers, including no hidden layers specified by an empty list as `[]`. <font color='red'>Don't forget this case.</font>
# 2. Define `_forward(self, X)` and `_gradients(self, X, T` functions. `_forward` must return the output of the network, `Y`, in standardized form and create `self.Zs` as a list consisting of the input `X` and the outputs of all hidden layers. `_gradients` must return the gradients of the mean square error with respect to the weights in each layer. 
# 2. Your `train` function must standardize `X` and `T` and save the standardization parameters (means and stds) in member variables. It must append to `self.rmse_trace` the RMSE value for each epoch.  Initialize this list to be `[]` in the constructor to allow multiple calls to `train` to continue to append to the same `rmse_trace` list.
# 2. Your `use` function must standardize `X` and unstandardize the output.
# 
# See the following examples for more details.
# 
# Then apply your `NeuralNetwork` class to the problem of predicting the value of houses in Boston as described below.

# ## Code for `NeuralNetwork` Class

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import IPython.display as ipd  # for display and clear_output
import time


# In[2]:


class NeuralNetwork():
    """
    A Neural Network class consisting methods to create a neural network. Written for non-linear regression problems.
    
    :param n_inputs: int
                     Number of variable columns in the dataset (excluding output varibales). 
                     Represents number of predictors in the model.
    :param n_hiddens_each_layer: list of int
                                 Number of units in each hidden layers. Each element of the list represents 
                                 to one hidden layer.  
    :param n_outputs: int
                      Number of output variables to predict.
    :param n_epochs: int|
                     Number of epochs to run. Default is None. Updates from the train() function.
    :param rho: float
                Learning rate of the neural network. Default is None. Updates from the train() function.
                
    :param rmse_trace: list of floats
                       An auto-updated list generated while training the model. Holds all rmse values while training.
    :param X_means : 1D numpy array
                     Mean value of input variables.
    :param X_stds : 1D numpy array
                    Standard deviation of input variables.
    :param T_means : 1D numpy array
                     Mean value of output variables.
    :param T_stds : 1D numpy array
                    Standard deviation of output variables.
                    
    ---------------------
    Methods:
    
    rmse_score() : Calculates RMSE score.
    _add_ones() : Adds a column of 1 in the begininning of an array. Used to add bias to data.
    _generate_weights() : Generates weights for hidden layers of the neural network based on class inputs.
    _forward() : Performs the forward propagation steps.
    _gradients() : Performs backward propagation to generate gradients. 
    train() : Trains the neural network with input parameters and updates weights.
    use() : Predicts output variable using the trained model.
    
    """

    def __init__(self, n_inputs, n_hiddens_each_layer, n_outputs):
        """
        Generates a neural network with a given structure. Holds various parameters of the 
        neural network.

        :param n_inputs: int
                         Number of variable columns in the dataset (excluding output varibales). 
                         Represents number of predictors in the model.
        :param n_hiddens_each_layer: list of int
                                     Number of units in each hidden layers. Each element of the list represents one hidden layer.  
        :param n_outputs: int
                          Number of output variables to predict.
                          
        :return: A Neural netwrk object.
        """
        self.n_inputs = n_inputs
        self.n_hiddens_each_layer = n_hiddens_each_layer
        self.n_outputs = n_outputs

        self.n_epochs = None
        self.rho = None
        self.rmse_trace = []

        self.Ws = self._generate_weights()
        self.Zs = None

        self.X_means = None
        self.X_stds = None
        self.T_means = None
        self.T_stds = None

    def rmse_score(self, Y, T, T_std):
        """
        Calculates RMSE score.
        
        ** T and Y need to be standardized in this function is used on test data.
        
        :param Y: 2D numpy array
                  Model predicted output variable. 
        :param T: 2D numpy array
                  Observed variable that model will predict.
        :param T_std: 1D numpy array
                      Standard deviations of observed output variable.
                      
        :return: RMSE score.
        """
        error = (T - Y) * T_std 
        rmse_error = np.sqrt(np.mean(error ** 2))
        return rmse_error


    def _add_ones(self, X):
        """
        Adds a column of 1 in the begininning of an array. Used to add bias to data.
        
        :param X: 2D numpy array
                  Input variables to which bias need to be added.
        
        :return: Bias added 2D numpy array.
        """
        X1 = np.insert(X, 0, 1, axis=1)
        return X1
    
    
    def _generate_weights(self):
        """
        Generates weights for hidden layers of the neural network based on class inputs.
        
        :return: A list of numpy array of representating weights of hidden layers of the network.  
        """
        hidden_layers = [i for i in
                         self.n_hiddens_each_layer]  # assigning n_hiddens_each_layer to a local variable for convenience
        layer_weights = []
        if len(hidden_layers) > 0:
            hidden_layers.insert(0, self.n_inputs)
            hidden_layers.append(self.n_outputs)
    
            for i in range(len(hidden_layers) - 1):
                W_shape = (1 + hidden_layers[i], hidden_layers[i + 1])
                if i < (len(hidden_layers) - 2):  # inserting this block to set last weight to zeros
                    W = np.random.uniform(-0.01, 0.01, size=W_shape) / np.sqrt(1 + hidden_layers[i])
                else:
                    W = np.zeros(W_shape)
                layer_weights.append(W)  # weights are in first to last layer order
    
        else:
            W_shape = (self.n_inputs + 1, self.n_outputs)
            W = np.zeros(W_shape)
            layer_weights.append(W)
    
        return layer_weights
    
    
    def _forward(self, X):
        """
        Performs the forward propagation steps.
    
        :param X: 2D numpy array
                  Input variables of the model. Generally train dataset.
                  
        :return: List of Zs (Zs will have X in the beginning), Output variable Y for the forward propagation.
        """
        self.Zs = [X]
        # Append output of each layer to list in self.Zs, then return it.
        weights = self.Ws
        Z = X
        # for layers upto n-1
        for i in range(len(weights) - 1):
            Z_w = np.tanh(self._add_ones(Z) @ weights[i])               
            Z = Z_w
            self.Zs.append(Z_w)
            
        # for output layer
        if self.n_hiddens_each_layer:
            self.Zs.append(self._add_ones(Z_w) @ weights[-1]) 
        else:
            self.Zs.append(self._add_ones(X) @ weights[-1])
            
        return self.Zs, self.Zs[-1]
    
    
    def _gradients(self, X, T):
        """
        Performs backward propagation to generate gradients. 
    
        :param X: 2D numpy array
                  Input variables of the model. Generally train dataset.
        :param T: 2D numpy array
                  Observed variable that model will predict.
                  
        :return: List of gradients generated from the backpropagation.
        """
        weights = self.Ws
        forward_prop = self.Zs  # returns all Zs including Y in the end. Zs has X in the beginning
        Zs_without_Y = forward_prop[:-1]
        Zs_Y = forward_prop[-1:][0]
    
        gradient_list = []  # initializing gradient list
    
        # first delta of the epoch calculated from the last Zs
        delta = T - Zs_Y
    
        # backpropagating delta to previous layers and calculating gradients 
        for W, Z in zip(weights[::-1], Zs_without_Y[::-1]):
            gradient = -(self._add_ones(Z)).T @ delta
            delta = delta @ W[1:, :].T * (1 - Z ** 2)
            gradient_list.append(gradient)  # in reverse order, last layers' weight are in the beginning
    
        return gradient_list
    
    
    def train(self, X, T, n_epochs, learning_rate=None, verbose=True):
        """
        Trains the neural network with input parameters and updates weights.
        
        :param X: 2D numpy array
                  Input variables of the model. Generally train dataset.
        :param T: 2D numpy array
                  Observed variable that model will predict.
        :param n_epochs: int
                         Number of epochs to run.
        :param learning_rate: float
                              Learning rate of the neural network.
        :param verbose: boolean
                        Progress is shown with print statements if True.
                        
        :return: A trained model with updated weights.
        """
        self.rho = learning_rate
    
        self.X_means = X.mean(axis=0)
        self.X_stds = X.std(axis=0)
        self.T_means = T.mean(axis=0)
        self.T_stds = T.std(axis=0)
    
        # standardizing
        XtrainS = (X - self.X_means) / self.X_stds
        TtrainS = (T - self.T_means) / self.T_stds
    
        self.n_epochs = n_epochs
        for epoch in range(self.n_epochs):
            Zs, Ys = self._forward(XtrainS)
            gradients = self._gradients(XtrainS, TtrainS)
    
            # Updating weights. weights are in first to last order, gradients are in last to first order
            # Weights will be in first to last order
            # rho is divided by X.shape[0]/T.shape[1] because of keeping weights similar to grader
            weights = self.Ws
            self.Ws = [(w - self.rho/X.shape[0]/T.shape[1] * grd) for w, grd in zip(weights, gradients[::-1])]
            self.rmse_trace.append(self.rmse_score(Ys, TtrainS, self.T_stds))
    
        return self
    
    
    def use(self, X):
        """
        Predicts output variable using the trained model.
        
        :param X: 2D numpy array
                  Input variables, generally test dataset.
        
        :return: Predicted output variables.
        """
        Xs = (X - self.X_means) / self.X_stds  # standardizing input X
        Ys = self._forward(Xs)[-1]  # extracting output from last column of Zs
        Y = Ys * self.T_stds + self.T_means
    
        return Y
    
    def __repr__(self):
        return f'NeuralNetwork ({self.n_inputs}, {self.n_hiddens_each_layer}, {self.n_outputs})'
    
    
    def __str__(self):
        return self.__repr__() + f', trained for {self.n_epochs} epochs with a final RMSE of {self.rmse_trace}'


# In this next code cell, I add a new method to your class that replaces the weights created in your constructor with non-random values to allow you to compare your results with mine, and to allow our grading scripts to work well.

# In[3]:


def set_weights_for_testing(self):
    for W in self.Ws[:-1]:   # leave output layer weights at zero
        n_weights = W.shape[0] * W.shape[1]
        W[:] = np.linspace(-0.01, 0.01, n_weights).reshape(W.shape)
        for u in range(W.shape[1]):
            W[:, u] += (u - W.shape[1]/2) * 0.2
    # Set output layer weights to zero
    self.Ws[-1][:] = 0
    print('Weights set for testing by calling set_weights_for_testing()')

setattr(NeuralNetwork, 'set_weights_for_testing', set_weights_for_testing)


# ## Example Results

# Here we test the `NeuralNetwork` class with some simple data.  
# 

# In[4]:


X = np.arange(0, 10).reshape(-1, 1)
T = np.sin(X) + 0.01 * (X ** 2)
X.shape, T.shape


# In[5]:


plt.plot(X, T, '.-')


# In[6]:


n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [3, 2], n_outputs)
nnet


# In[7]:


nnet.n_inputs, nnet.n_hiddens_each_layer, nnet.n_outputs


# In[8]:


nnet.rmse_trace


# In[9]:


nnet.Ws


# In[10]:


nnet.set_weights_for_testing()


# In[11]:


nnet.Ws


# In[12]:


nnet.train(X, T, n_epochs=1, learning_rate=0.1)


# In[13]:


nnet.Zs


# In[14]:


print(nnet)


# In[15]:


nnet.X_means, nnet.X_stds


# In[16]:


nnet.T_means, nnet.T_stds


# In[17]:


[Z.shape for Z in nnet.Zs]


# In[18]:


nnet.Ws


# In[19]:


dir(nnet)


# In[20]:


def plot_data_and_model(nnet, X, T):
    plt.clf()        
    plt.subplot(2, 1, 1)
    plt.plot(nnet.rmse_trace)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')

    plt.subplot(2, 1, 2)
    Y = nnet.use(X)

    plt.plot(X, Y, 'o-', label='Y')
    plt.plot(X, T, 'o', label='T', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('T or Y')
    plt.legend();


# In[21]:


X = np.arange(0, 10).reshape(-1, 1)
# X = np.arange(0, 0.5, 0.05).reshape(-1, 1)
T = np.sin(X) + 0.01 * (X ** 2)

n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [10, 5], n_outputs)
nnet.set_weights_for_testing()

n_epochs = 5000
n_epochs_per_plot = 500


fig = plt.figure()
for reps in range(n_epochs // n_epochs_per_plot):
    plt.clf()
    nnet.train(X, T, n_epochs=n_epochs_per_plot, learning_rate=0.05)
    plot_data_and_model(nnet, X, T)
    ipd.clear_output(wait=True)
    ipd.display(fig)
    time.sleep(0.2)  # 0.2 seconds
ipd.clear_output(wait=True)


# In[22]:


X = np.arange(-2, 2, 0.05).reshape(-1, 1)
T = np.sin(X) * np.sin(X * 10)

n_inputs = X.shape[1]
n_outputs = T.shape[1]

nnet = NeuralNetwork(n_inputs, [50, 10, 5], n_outputs)
nnet.set_weights_for_testing()

n_epochs = 50000
n_epochs_per_plot = 500

fig = plt.figure()
for reps in range(n_epochs // n_epochs_per_plot):
    plt.clf()
    nnet.train(X, T, n_epochs=n_epochs_per_plot, learning_rate=0.1)
    plot_data_and_model(nnet, X, T)
    ipd.clear_output(wait=True)
    ipd.display(fig)
    # time.sleep(0.01)  # 0.01 seconds
ipd.clear_output(wait=True)


# Your results will not be the same, but your code should complete and make plots somewhat similar to these.

# ## Application to Boston Housing Data

# Download data from [Boston House Data at Kaggle](https://www.kaggle.com/fedesoriano/the-boston-houseprice-data). Read it into python using the `pandas.read_csv` function.  Assign the first 13 columns as inputs to `X` and the final column as target values to `T`.  Make sure `T` is two-dimensional.

# Before training your neural networks, partition the data into training and testing partitions, as shown here.

# In[23]:


# Building 27 Data Structures
import itertools

def partition(X, T, train_fraction):
    """
    Creates train and test datasets.
    
    :param X: 2D numpy array.
              The input variables of the model.
    :param T: 2D numpy array
              The output variable of the model.
    :param X: float
              fraction of data in training set.
              The input variables of the model.
    
    return: Xtrain, Ttrain, Xtest, Ttest
    
    """
    n_samples = X.shape[0]
    rows = np.arange(n_samples)
    np.random.shuffle(rows)

    n_train = round(n_samples * train_fraction)
    Xtrain = X[rows[:n_train], :]
    Ttrain = T[rows[:n_train], :]
    Xtest = X[rows[n_train:], :]
    Ttest = T[rows[n_train:], :]

    return Xtrain, Ttrain, Xtest, Ttest

def make_param_grid(structure, rho, epoch):
    """
    Makes a gridspace with all possible combinations of input parameters.
    
    :param structure: list of int
                      Number of units in each hidden layers. Each element of the list represents one hidden layer.
    :param rho: float
                Learning rate of the model.
    :param epoch: int
                  Number of epoch to run the model.
                  
    :return: A list of all parameter combinations.
    """
    param_combs = list(itertools.product(*[structure, rho, epoch]))
    return param_combs

def rmse(Y, T):
    """
    calculates rmse score
    """
    return np.sqrt(np.mean((Y-T)**2))

def configure_nnet_model(X, T, structures, rhos, epochs, train_test_ratio):
    """
    Runs the model for number of times of parameter combinations and saves model layout, train rmse, and test rmse. 
    
    :param X: 2D numpy array.
              The input variables of the model.
    :param T: 2D numpy array
              The output variable of the model.
    :param structures: A nested list of int, such as [[50, 20], [30, 10]]
                       Number of units in each hidden layers. Each element of the list represents one hidden layer.
    :param rhos: list of float
                 Learning rate of the model.
    :param epochs: list of int
                   Number of epoch to run the model.
    :param train_test_ratio: 
    
    :return: A dataframe with model layouts and respective train test rmse.
    """
    parameter_combinations = make_param_grid(structures, rhos, epochs)

    Xtrain, Ttrain, Xtest, Ttest = partition(X, T, train_test_ratio)

    n_inputs, n_outputs = Xtrain.shape[1], Ttrain.shape[1]

    result_dict = {'Structure': [], 'Epochs': [], 'Learning Rate': [], 'Train RMSE': [], 
                   'Test RMSE': []}

    for params in parameter_combinations:
        hidden_layers, rho, epoch = params
        print(f'Training for {hidden_layers=}, {rho=}, {epoch=}')
        nn_model = NeuralNetwork(n_inputs, hidden_layers, n_outputs)
        nn_model.train(Xtrain, Ttrain, epoch, learning_rate=rho)
        
        # RMSE (No standardization is required)
        Ttrain_predicted = nn_model.use(Xtrain)
        rmse_train = rmse(Ttrain_predicted, Ttrain)
        
        Ttest_predicted = nn_model.use(Xtest)
        rmse_test = rmse(Ttest_predicted, Ttest)
        
        # appending results to the result dictionary
        result_dict['Structure'].append(hidden_layers)
        result_dict['Epochs'].append(epoch)
        result_dict['Learning Rate'].append(rho)
        result_dict['Train RMSE'].append(rmse_train)
        result_dict['Test RMSE'].append(rmse_test)

    result_df = pd.DataFrame(result_dict)
    result_df.sort_values(by=['Test RMSE'], axis=0, ascending=True, inplace=True)

    return result_df


# In[24]:


# Reading Boston Data
boston_df = pd.read_csv('boston.csv')
boston_df.columns


# In[25]:


X_df = boston_df.iloc[:, :-1]
T_df = boston_df.iloc[:, -1:]
X_df.head()


# In[26]:


T_df.head()


# In[27]:


X = X_df.to_numpy()
T = T_df.to_numpy()


# In[28]:


# Train Test partition using partition() function
Xtrain, Ttrain, Xtest, Ttest = partition(X, T, train_fraction=0.7)
np.hstack((Xtest, Ttest))


# Write and run code using your `NeuralNetwork` class to model the Boston housing data. Experiment with a variety of neural network structures (numbers of hidden layer and units) including no hidden layers, learning rates, and numbers of epochs. Show results for at least three different **network structures, learning rates, and numbers of epochs** for a total of at least 27 results.  Show your results in a `pandas` DataFrame with columns `('Structure', 'Epochs', 'Learning Rate', 'Train RMSE', 'Test RMSE')`.
# 
# Try to find good values for the RMSE on testing data.  Discuss your results, including how good you think the RMSE values are by considering the range of house values given in the data. 

# In[ ]:


# Searching for best model layout for Boston data
nnet_structure = [[30, 20, 10, 5], [10, 5], []]
nnet_rho = [0.001, 0.005, 0.01]
nnet_epoch = [2000, 5000, 10000]

results = configure_nnet_model(X, T, nnet_structure, nnet_rho, nnet_epoch, train_test_ratio=0.7)
results


# <h3> Discussion </h3>
# 
# <b>Three</b> types of model structures were tested for the designed Neural Network class using the <b>Boston</b> dataset.
# 
# 1. More hidden layers: [20, 10, 10, 5]
# 2. Less hidden layers: [10, 5]
# 3. No hidden layers: []
# 
# Surprisingly, <b> No hidden layers</b> NN models performs the best in terms of RMSE values. Test rmse is higher than train rmse, which means there is no overfitting. The best possible reason for such good performance can be -the variables in boston dataset are very linearly connected to the output variable 'median value'. So, even model with no hidden layers work that well. With this fewer training sample and 13 number of variables, a model with no hidden layer might be the best model.
# 
# <b> Less hidden layers </b> NN models and <b> More hidden layers </b> NN models work almost same with the data. The created models are overfitting the data with lower test rmse than train rmse. It seems like non-linear models with hidden layers might not be the best fit for a dataset with such low number of samples. If there were more samples, hidden layers might have been been able to find out underlying patterns and generate better results.
# 
# The output variable's maximum and minimum range in between **50 to 5**. Test rmse of 5.28 (lowest found during the experiment) indicates that the model with no hidden layer is going to perform relatively better for samples with output value near the maximum. As discussed above, the models with hidden layers will work relatively poorer. Possibly, more samples in the training dataset, higher number of epochs, and more complex neural networks would be able to perform better on the dataset with lower rmse values.

# # Grading
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A2grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A2grader.tar) and extract `A2grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 20 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  
# 
# For the grading script to run correctly, you must first name this notebook as `Lastname-A2.ipynb` with `Lastname` being your last name, and then save this notebook.

# In[ ]:


get_ipython().run_line_magic('run', '-i A2grader.py')


# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.
