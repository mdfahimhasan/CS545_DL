#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Neural-Network-Classifier" data-toc-modified-id="Neural-Network-Classifier-1">Neural Network Classifier</a></span></li><li><span><a href="#Apply-NeuralNetworkClassifier-to-Handwritten-Digits" data-toc-modified-id="Apply-NeuralNetworkClassifier-to-Handwritten-Digits-2">Apply <code>NeuralNetworkClassifier</code> to Handwritten Digits</a></span></li><li><span><a href="#Experiments" data-toc-modified-id="Experiments-3">Experiments</a></span><ul class="toc-item"><li><span><a href="#Check-In" data-toc-modified-id="Check-In-3.1">Check-In</a></span></li></ul></li><li><span><a href="#Grading" data-toc-modified-id="Grading-4">Grading</a></span></li><li><span><a href="#Extra-Credit" data-toc-modified-id="Extra-Credit-5">Extra Credit</a></span></li></ul></div>

# # Neural Network Classifier
# 
# For this assignment, you will be adding code to the python script file `neuralnetworksA4.py` that you can download from [here](https://www.cs.colostate.edu/~anderson/cs545/notebooks/neuralnetworksA4.tar). This file currently contains the implementation of the `NeuralNetwork` class that is a solution to A3. It also contains an incomplete implementation of the subclass `NeuralNetworkClassifier` that extends `NeuralNetwork` as discussed in class.  You must complete this implementation. Your `NeuralNetworkClassifier` implementation should rely on inheriting functions from `NeuralNetwork` as much as possible. Your `neuralnetworksA4.py` file (notice it is plural) will now contain two classes, `NeuralNetwork` and `NeuralNetworkClassifier`.
# 
# In `NeuralNetworkClassifier` you will replace the `_error_f` function with one called `_neg_log_likelihood_f`. You will also have to define a new version of the `_gradient_f` function for `NeuralNetworkClassifier`.

# Here are some example tests.

# In[5]:


get_ipython().run_cell_magic('writefile', 'neuralnetworksA4.py', '\nimport numpy as np\nimport optimizers as opt\nimport sys  # for sys.float_info.epsilon\n\n######################################################################\n## class NeuralNetwork()\n######################################################################\n\nclass NeuralNetwork():\n\n    """\n    A class that represents a neural network for nonlinear regression.\n\n    Attributes\n    ----------\n    n_inputs : int\n        The number of values in each sample\n    n_hidden_units_by_layers : list of ints, or empty\n        The number of units in each hidden layer.\n        Its length specifies the number of hidden layers.\n    n_outputs : int\n        The number of units in output layer\n    all_weights : one-dimensional numpy array\n        Contains all weights of the network as a vector\n    Ws : list of two-dimensional numpy arrays\n        Contains matrices of weights in each layer,\n        as views into all_weights\n    all_gradients : one-dimensional numpy array\n        Contains all gradients of mean square error with\n        respect to each weight in the network as a vector\n    Grads : list of two-dimensional numpy arrays\n        Contains matrices of gradients weights in each layer,\n        as views into all_gradients\n    total_epochs : int\n        Total number of epochs trained so far\n    performance_trace : list of floats\n        Mean square error (unstandardized) after each epoch\n    n_epochs : int\n        Number of epochs trained so far\n    X_means : one-dimensional numpy array\n        Means of the components, or features, across samples\n    X_stds : one-dimensional numpy array\n        Standard deviations of the components, or features, across samples\n    T_means : one-dimensional numpy array\n        Means of the components of the targets, across samples\n    T_stds : one-dimensional numpy array\n        Standard deviations of the components of the targets, across samples\n    debug : boolean\n        If True, print information to help with debugging\n        \n    Methods\n    -------\n    make_weights_and_views(shapes)\n        Creates all initial weights and views for each layer\n\n    train(X, T, n_epochs, method=\'sgd\', learning_rate=None, verbose=True)\n        Trains the network using input and target samples by rows in X and T\n\n    use(X)\n        Applies network to inputs X and returns network\'s output\n    """\n\n    def __init__(self, n_inputs, n_hiddens_each_layer, n_outputs):\n        """Creates a neural network with the given structure\n\n        Parameters\n        ----------\n        n_inputs : int\n            The number of values in each sample\n        n_hidden_units_by_layers : list of ints, or empty\n            The number of units in each hidden layer.\n            Its length specifies the number of hidden layers.\n        n_outputs : int\n            The number of units in output layer\n\n        Returns\n        -------\n        NeuralNetwork object\n        """\n\n        self.n_inputs = n_inputs\n        self.n_hiddens_each_layer = n_hiddens_each_layer\n        self.n_outputs = n_outputs\n\n        # Create one-dimensional numpy array of all weights with random initial values\n\n        shapes = []\n        n_in = n_inputs\n        for nu in self.n_hiddens_each_layer + [n_outputs]:\n            shapes.append((n_in + 1, nu))\n            n_in = nu\n\n        # Build list of views (pairs of number of rows and number of columns)\n        # by reshaping corresponding elements from vector of all weights \n        # into correct shape for each layer.        \n\n        self.all_weights, self.Ws = self.make_weights_and_views(shapes)\n        self.all_gradients, self.Grads = self.make_weights_and_views(shapes)\n\n        self.X_means = None\n        self.X_stds = None\n        self.T_means = None\n        self.T_stds = None\n\n        self.total_epochs = 0\n        self.performance = None\n        self.performance_trace = []\n        self.debug = False\n        \n    def __repr__(self):\n        return \'{}({}, {}, {})\'.format(type(self).__name__, self.n_inputs, self.n_hiddens_each_layer, self.n_outputs)\n\n    def __str__(self):\n        s = self.__repr__()\n        if self.total_epochs > 0:\n            s += \'\\n Trained for {} epochs.\'.format(self.total_epochs)\n            s += \'\\n Final standardized RMSE {:.4g}.\'.format(self.performance_trace[-1])\n        return s\n \n    def make_weights_and_views(self, shapes):\n        """Creates vector of all weights and views for each layer\n\n        Parameters\n        ----------\n        shapes : list of pairs of ints\n            Each pair is number of rows and columns of weights in each layer.\n            Number of rows is number of inputs to layer (including constant 1).\n            Number of columns is number of units, or outputs, in layer.\n\n        Returns\n        -------\n        Vector of all weights, and list of views into this vector for each layer\n        """\n\n        # Make vector of all weights by stacking vectors of weights one layer at a time\n        # Divide each layer\'s weights by square root of number of inputs\n        all_weights = np.hstack([np.random.uniform(-1, 1, size=shape).flat / np.sqrt(shape[0])\n                                 for shape in shapes])\n\n        # Build weight matrices as list of views (pairs of number of rows and number \n        # of columns) by reshaping corresponding elements from vector of all weights \n        # into correct shape for each layer.  \n        # Do the same to make list of views for gradients.\n \n        views = []\n        first_element = 0\n        for shape in shapes:\n            n_elements = shape[0] * shape[1]\n            last_element = first_element + n_elements\n            views.append(all_weights[first_element:last_element].reshape(shape))\n            first_element = last_element\n\n        # Set output layer weights to zero.\n        views[-1][:] = 0\n        \n        return all_weights, views\n\n    def set_debug(self, d):\n        """Set or unset printing of debugging information.\n\n        Parameters\n        ----------\n        d : boolean\n            If True, print debugging information. \n        """\n        \n        self.debug = d\n        if self.debug:\n            print(\'Debugging information will now be printed.\')\n        else:\n            print(\'No debugging information will be printed.\')\n        \n    def train(self, X, T, n_epochs, method=\'sgd\', learning_rate=None, verbose=True):\n        """Updates the weights.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array \n            number of samples  by  number of input components\n        T : two-dimensional numpy array\n            number of samples  by  number of output components\n        n_epochs : int\n            Number of passes to take through all samples\n        method : str\n            \'sgd\', \'adam\', or \'scg\'\n        learning_rate : float\n            Controls the step size of each update, only for sgd and adam\n        verbose: boolean\n            If True, progress is shown with print statements\n\n        Returns\n        -------\n        self : NeuralNetwork instance\n        """\n\n        # Calculate and assign standardization parameters\n\n        if self.X_means is None:\n            self.X_means = X.mean(axis=0)\n            self.X_stds = X.std(axis=0)\n            self.X_stds[self.X_stds == 0] = 1\n            self.T_means = T.mean(axis=0)\n            self.T_stds = T.std(axis=0)\n\n        # Standardize X and T.  Assign back to X and T.\n\n        X = (X - self.X_means) / self.X_stds\n        T = (T - self.T_means) / self.T_stds\n\n        # Instantiate Optimizers object by giving it vector of all weights\n        \n        optimizer = opt.Optimizers(self.all_weights)\n\n        # Define function to convert mean-square error to root-mean-square error,\n        # Here we use a lambda function just to illustrate its use.  \n        # We could have also defined this function with\n        # def error_convert_f(err):\n        #     return np.sqrt(err)\n\n        error_convert_f = lambda err: np.sqrt(err)\n\n        # Call the requested optimizer method to train the weights.\n\n        if method == \'sgd\':\n\n            performance_trace = optimizer.sgd(self._error_f, self._gradient_f,\n                                              fargs=[X, T], n_epochs=n_epochs,\n                                              learning_rate=learning_rate,\n                                              error_convert_f=error_convert_f,\n                                              error_convert_name=\'RMSE\',\n                                              verbose=verbose)\n\n        elif method == \'adam\':\n\n            performance_trace = optimizer.adam(self._error_f, self._gradient_f,\n                                               fargs=[X, T], n_epochs=n_epochs,\n                                               learning_rate=learning_rate,\n                                               error_convert_f=error_convert_f,\n                                               error_convert_name=\'RMSE\',\n                                               verbose=verbose)\n\n        elif method == \'scg\':\n\n            performance_trace = optimizer.scg(self._error_f, self._gradient_f,\n                                              fargs=[X, T], n_epochs=n_epochs,\n                                              error_convert_f=error_convert_f,\n                                              error_convert_name=\'RMSE\',\n                                              verbose=verbose)\n\n        else:\n            raise Exception("method must be \'sgd\', \'adam\', or \'scg\'")\n\n        self.total_epochs += len(performance_trace)\n        self.performance_trace += performance_trace\n\n        # Return neural network object to allow applying other methods\n        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)\n\n        return self\n\n    def _add_ones(self, X):\n        return np.insert(X, 0, 1, 1)\n    \n    def _forward(self, X):\n        """Calculate outputs of each layer given inputs in X.\n        \n        Parameters\n        ----------\n        X : input samples, standardized with first column of constant 1\'s.\n\n        Returns\n        -------\n        Standardized outputs of all layers as list, include X as first element.\n        """\n\n        self.Zs = [X]\n\n        # Append output of each layer to list in self.Zs, then return it.\n\n        for W in self.Ws[:-1]:  # forward through all but last layer\n            self.Zs.append(np.tanh(self._add_ones(self.Zs[-1]) @ W))\n        last_W = self.Ws[-1]\n        self.Zs.append(self._add_ones(self.Zs[-1]) @ last_W)\n\n        return self.Zs\n\n    # Function to be minimized by optimizer method, mean squared error\n    def _error_f(self, X, T):\n        """Calculate output of net given input X and its mean squared error.\n        Function to be minimized by optimizer.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array, standardized\n            number of samples  by  number of input components\n        T : two-dimensional numpy array, standardized\n            number of samples  by  number of output components\n\n        Returns\n        -------\n        Standardized mean square error as scalar float that is the mean\n        square error over all samples and all network outputs.\n        """\n\n        if self.debug:\n            print(\'in _error_f: X[0] is {} and T[0] is {}\'.format(X[0], T[0]))\n        Zs = self._forward(X)\n        mean_sq_error = np.mean((T - Zs[-1]) ** 2)\n        if self.debug:\n            print(f\'in _error_f: mse is {mean_sq_error}\')\n        return mean_sq_error\n\n    # Gradient of function to be minimized for use by optimizer method\n    def _gradient_f(self, X, T):\n        """Returns gradient wrt all weights. Assumes _forward already called\n        so input and all layer outputs stored in self.Zs\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array, standardized\n            number of samples  x  number of input components\n        T : two-dimensional numpy array, standardized\n            number of samples  x  number of output components\n\n        Returns\n        -------\n        Vector of gradients of mean square error wrt all weights\n        """\n\n        # Assumes forward_pass just called with layer outputs saved in self.Zs.\n\n        if self.debug:\n            print(\'in _gradient_f: X[0] is {} and T[0] is {}\'.format(X[0], T[0]))\n        n_samples = X.shape[0]\n        n_outputs = T.shape[1]\n\n        # delta is delta matrix to be back propagated.\n        # Dividing by n_samples and n_outputs here replaces the scaling of\n        # the learning rate.\n\n        delta = -(T - self.Zs[-1]) / (n_samples * n_outputs)\n\n        # Step backwards through the layers to back-propagate the error (delta)\n        self._backpropagate(self, delta)\n\n        return self.all_gradients\n\n    def _backpropagate(self, delta):\n        """Backpropagate output layer delta through all previous layers,\n        setting self.Grads, the gradient of the objective function wrt weights in each layer.\n\n        Parameters\n        ----------\n        delta : two-dimensional numpy array of output layer delta values\n            number of samples  x  number of output components\n        """\n\n        n_layers = len(self.n_hiddens_each_layer) + 1\n        if self.debug:\n            print(\'in _gradient_f: first delta calculated is\\n{}\'.format(delta))\n        for layeri in range(n_layers - 1, -1, -1):\n            # gradient of all but bias weights\n            self.Grads[layeri][:] = self._add_ones(self.Zs[layeri]).T @ delta\n            # Back-propagate this layer\'s delta to previous layer\n            if layeri > 0:\n                delta = delta @ self.Ws[layeri][1:, :].T * (1 - self.Zs[layeri] ** 2)\n                if self.debug:\n                    print(\'in _gradient_f: next delta is\\n{}\'.format(delta))\n\n    def use(self, X):\n        """Return the output of the network for input samples as rows in X.\n        X assumed to not be standardized.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array\n            number of samples  by  number of input components, unstandardized\n\n        Returns\n        -------\n        Output of neural network, unstandardized, as numpy array\n        of shape  number of samples  by  number of outputs\n        """\n\n        # Standardize X\n        X = (X - self.X_means) / self.X_stds\n        Zs = self._forward(X)\n        # Unstandardize output Y before returning it\n        return Zs[-1] * self.T_stds + self.T_means\n\n    def get_performance_trace(self):\n        """Returns list of unstandardized root-mean square error for each epoch"""\n        return self.performance_trace\n\n\n######################################################################\n## class NeuralNetworkClassifier(NeuralNetwork)\n######################################################################\nclass NeuralNetworkClassifier(NeuralNetwork):\n    \n    def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):\n        super(NeuralNetworkClassifier, self).__init__(n_inputs, n_hidden_units_by_layers, n_outputs)\n    \n    \n    def __repr__(self):\n        return f\'NeuralNetworkClassifier({self.n_inputs}, \' + \\\n            f\'{self.n_hidden_units_by_layers}, {self.n_outputs})\'\n    \n    def __str__(self):\n        s = self.__repr__()  # using NeuralNetwork.__repr__()\n        if self.total_epochs > 0:\n            s += \'\\n Trained for {} epochs.\'.format(self.total_epochs)\n            s += \'\\n Final data likelihood {:.4g}.\'.format(self.performance_trace[-1])\n        return s\n \n    def _make_indicator_vars(self, T):\n        """Convert column matrix of class labels (ints or strs) into indicator variables\n\n        Parameters\n        ----------\n        T : two-dimensional array of all ints or all strings\n            number of samples by 1\n        \n        Returns\n        -------\n        Two dimensional array of indicator variables. Each row is all 0\'s except one value of 1.\n            number of samples by number of output components (number of classes)\n        """\n\n        # Make sure T is two-dimensional. Should be n_samples x 1.\n        if T.ndim == 1:\n            T = T.reshape((-1, 1))    \n        return (T == np.unique(T)).astype(float)  # to work on GPU\n\n    def _softmax(self, Y):\n        """Convert output Y to exp(Y) / (sum of exp(Y)\'s)\n\n        Parameters\n        ----------\n        Y : two-dimensional array of network output values\n            number of samples by number of output components (number of classes)\n\n        Returns\n        -------\n        Two-dimensional array of indicator variables representing Y\n            number of samples by number of output components (number of classes)\n        """\n\n        # Trick to avoid overflow\n        # maxY = max(0, self.max(Y))\n        maxY = Y.max()  #self.max(Y))        \n        expY = np.exp(Y - maxY)\n        denom = expY.sum(1).reshape((-1, 1))\n        Y_softmax = expY / (denom + sys.float_info.epsilon)\n        return Y_softmax\n\n    # Function to be minimized by optimizer method, mean squared error\n    def _neg_log_likelihood_f(self, X, T):\n        """Calculate output of net given input X and the resulting negative log likelihood.\n        Function to be minimized by optimizer.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array, standardized\n            number of samples  by  number of input components\n        T : two-dimensional numpy array of class indicator variables\n            number of samples  by  number of output components (number of classes)\n\n        Returns\n        -------\n        Negative log likelihood as scalar float.\n        """\n        Y = self._softmax(self._forward(X)[-1])\n        neg_mean_log_likelihood =  - np.mean(T * np.log(Y + sys.float_info.epsilon))  # T should be indicator variable\n\n        return neg_mean_log_likelihood\n\n    def _gradient_f(self, X, T):\n        """Returns gradient wrt all weights. Assumes _forward (from NeuralNetwork class)\n        has already called so input and all layer outputs stored in self.Zs\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array, standardized\n            number of samples  x  number of input components\n        T : two-dimensional numpy array of class indicator variables\n            number of samples  by  number of output components (number of classes)\n\n        Returns\n        -------\n        Vector of gradients of negative log likelihood wrt all weights\n        """\n\n        n_samples = X.shape[0]\n        n_outputs = T.shape[1]\n\n        # delta is delta matrix to be back propagated.\n        # Dividing by n_samples and n_outputs here replaces the scaling of\n        # the learning rate.\n\n        delta = -(T - self._softmax(self.Zs[-1])) / (n_samples * n_outputs)\n\n        # Step backwards through the layers to back-propagate the error (delta)\n        self._backpropagate(delta)\n\n        return self.all_gradients\n                    \n\n    def train(self, X, T, n_epochs, method=\'sgd\', learning_rate=None, verbose=True):\n        """Updates the weights.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array \n            number of samples  by  number of input components\n        T : two-dimensional numpy array of target classes, as ints or strings\n            number of samples  by  1\n        n_epochs : int\n            Number of passes to take through all samples\n        method : str\n            \'sgd\', \'adam\', or \'scg\'\n        learning_rate : float\n            Controls the step size of each update, only for sgd and adam\n        verbose: boolean\n            If True, progress is shown with print statements\n\n        Returns\n        -------\n        self : NeuralNetworkClassifier instance\n        """\n\n        # Calculate and assign standardization parameters\n\n        if self.X_means is None:\n            self.X_means = X.mean(axis=0)\n            self.X_stds = X.std(axis=0)\n            self.X_stds[self.X_stds == 0] = 1\n            # Not standardizing target classes.\n\n        # Standardize X and assign back to X.\n\n        X = (X - self.X_means) / self.X_stds\n\n        # Assign class labels to self.classes, and counts of each to counts.\n        # Create indicator values representation from target labels in T.\n\n        self.classes, counts = np.unique(T, return_counts=True)\n        T_ind_vars = self._make_indicator_vars(T)\n\n        # Instantiate Optimizers object by giving it vector of all weights.\n        optimizer = opt.Optimizers(self.all_weights)\n\n        # Define function to convert negative log likelihood values to likelihood values.\n\n        _error_convert_f = lambda nll: np.exp(-nll)\n\n        if method == \'sgd\':\n\n            performance_trace = optimizer.sgd(self._neg_log_likelihood_f,\n                                            self._gradient_f,\n                                            fargs=[X, T_ind_vars], n_epochs=n_epochs,\n                                            learning_rate=learning_rate,\n                                            error_convert_f=_error_convert_f,\n                                            error_convert_name=\'Likelihood\',\n                                            verbose=verbose)\n\n        elif method == \'adam\':\n\n            performance_trace = optimizer.adam(self._neg_log_likelihood_f,\n                                               self._gradient_f,\n                                               fargs=[X, T_ind_vars], n_epochs=n_epochs,\n                                               learning_rate=learning_rate,\n                                               error_convert_f=_error_convert_f,\n                                               error_convert_name=\'Likelihood\',\n                                               verbose=verbose)\n\n        elif method == \'scg\':\n\n            performance_trace = optimizer.scg(self._neg_log_likelihood_f,\n                                              self._gradient_f,\n                                              fargs=[X, T_ind_vars], n_epochs=n_epochs,\n                                              error_convert_f=_error_convert_f,\n                                              error_convert_name=\'Likelihood\',\n                                              verbose=verbose)\n\n        else:\n            raise Exception("method must be \'sgd\', \'adam\', or \'scg\'")\n\n        self.total_epochs += len(performance_trace)\n        self.performance_trace += performance_trace\n\n        # Return neural network object to allow applying other methods\n        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)\n\n        return self\n\n    def use(self, X):\n        """Return the predicted class and probabilities for input samples as rows in X.\n        X assumed to not be standardized.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array, unstandardized input samples by rows\n            number of samples  by  number of input components, unstandardized\n\n        Returns\n        -------\n        Classes (Predicted classes) : two-dimensional array of predicted classes for each sample\n            number of samples by 1  of ints or strings, depending on how target classes were specified\n        Y (Class probabilities)  : Two_dimensional array of probabilities of each class for each sample\n            number of samples by number of outputs (number of classes)\n        """\n\n        # Standardize X\n        X = (X - self.X_means) / self.X_stds\n\n        Y_softmax = self._softmax(self._forward(X)[-1])\n        Y = Y_softmax\n        \n        pred_idx = np.argmax(Y_softmax, axis=1)\n        classes = np.array([self.classes[idx] for idx in pred_idx]).reshape(-1, 1)\n     \n        return classes, Y')


# In[6]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[8]:


import neuralnetworksA4 as nn


# In[9]:


issubclass(NeuralNetworkClassifier, NeuralNetwork)


# In[ ]:


X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
T = np.array([[0], [1], [1], [0]])
X, T


# In[ ]:


np.random.seed(111)
nnet = nn.NeuralNetworkClassifier(2, [10], 2)


# In[ ]:


print(nnet)


# In[ ]:


nnet.Ws


# The `_error_f` function is replaced with `_neg_log_likelihood`.  If you add some print statements in `_neg_log_likelihood` functions, you can compare your output to the following results.

# In[ ]:


nnet.set_debug(True)


# In[ ]:


nnet.train(X, T, n_epochs=1, method='sgd', learning_rate=0.01)


# In[ ]:


nnet.set_debug(False)


# In[ ]:


print(nnet)


# Now if you comment out those print statements, you can run for more epochs without tons of output.

# In[ ]:


np.random.seed(111)
nnet = nn.NeuralNetworkClassifier(2, [10], 2)


# In[ ]:


nnet.train(X, T, 100, method='scg')


# The `use()` function returns two `numpy` arrays. The first one are the class predictions for eachs sample, containing values from the set of unique values in `T` passed into the `train()` function.
# 
# The second value are the probabilities of each class for each sample. This should a column for each unique value in `T`.

# In[ ]:


nnet.use(X)


# In[ ]:


def percent_correct(Y, T):
    return np.mean(T == Y) * 100


# In[ ]:


percent_correct(nnet.use(X)[0], T)


# Works!  The XOR problem was used early in the history of neural networks as a problem that cannot be solved with a linear model.  Let's try it.  It turns out our neural network code can do this if we use an empty list for the hidden unit structure!

# In[ ]:


nnet = nn.NeuralNetworkClassifier(2, [], 2)
nnet.train(X, T, 100, method='scg')


# In[ ]:


nnet.use(X)


# In[ ]:


percent_correct(nnet.use(X)[0], T)


# A second way to evaluate a classifier is to calculate a confusion matrix. This shows the percent accuracy for each class, and also shows which classes are predicted in error.
# 
# Here is a function you can use to show a confusion matrix.

# In[ ]:


import pandas

def confusion_matrix(Y_classes, T):
    class_names = np.unique(T)
    table = []
    for true_class in class_names:
        row = []
        for Y_class in class_names:
            row.append(100 * np.mean(Y_classes[T == true_class] == Y_class))
        table.append(row)
    conf_matrix = pandas.DataFrame(table, index=class_names, columns=class_names)
    # cf.style.background_gradient(cmap='Blues').format("{:.1f} %")
    print('Percent Correct')
    return conf_matrix.style.background_gradient(cmap='Blues').format("{:.1f}")


# In[ ]:


confusion_matrix(nnet.use(X)[0], T)


# # Apply `NeuralNetworkClassifier` to Handwritten Digits

# Apply your `NeuralNetworkClassifier` to the [MNIST digits dataset](https://www.cs.colostate.edu/~anderson/cs545/notebooks/mnist.pkl.gz).

# In[6]:


import pickle
import gzip


# In[7]:


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

Xtrain = train_set[0]
Ttrain = train_set[1].reshape(-1, 1)

Xval = valid_set[0]
Tval = valid_set[1].reshape(-1, 1)

Xtest = test_set[0]
Ttest = test_set[1].reshape(-1, 1)

print(Xtrain.shape, Ttrain.shape,  Xval.shape, Tval.shape,  Xtest.shape, Ttest.shape)


# In[8]:


28*28


# In[9]:


def draw_image(image, label, predicted_label=None):
    plt.imshow(-image.reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    title = str(label)
    color = 'black'
    if predicted_label is not None:
        title += ' as {}'.format(predicted_label)
        if predicted_label != label:
            color = 'red'
    plt.title(title, color=color)


# In[10]:


plt.figure(figsize=(7, 7))
for i in range(100):
    plt.subplot(10, 10, i+1)
    draw_image(Xtrain[i], Ttrain[i, 0])
plt.tight_layout()


# In[11]:


nnet = nn.NeuralNetworkClassifier(784, [], 10)
nnet.train(Xtrain, Ttrain, n_epochs=40, method='scg')


# In[12]:


print(nnet)


# In[13]:


[percent_correct(nnet.use(X)[0], T) for X, T in zip([Xtrain, Xval, Xtest], [Ttrain, Tval, Ttest])]


# In[ ]:


confusion_matrix(nnet.use(Xtest)[0], Ttest)


# In[ ]:


nnet = nn.NeuralNetworkClassifier(784, [20], 10)
nnet.train(Xtrain, Ttrain, n_epochs=40, method='scg')


# In[ ]:


[percent_correct(nnet.use(X)[0], T) for X, T in zip([Xtrain, Xval, Xtest],
                                                    [Ttrain, Tval, Ttest])]


# In[ ]:


confusion_matrix(nnet.use(Xtest)[0], Ttest)


# In[ ]:


plt.figure(figsize=(7, 7))
Ytest, _ = nnet.use(Xtest[:100, :])
for i in range(100):
    plt.subplot(10, 10, i + 1)
    draw_image(Xtest[i], Ttest[i, 0], Ytest[i, 0])
plt.tight_layout()


# # Experiments
# 
# For each method, try various hidden layer structures, learning rates, and numbers of epochs.  Use the validation percent accuracy to pick the best hidden layers, learning rates and numbers of epochs for each method (ignore learning rates for scg).  Report training, validation and test accuracy for your best validation results for each of the three methods.
# 
# Include plots of data likelihood versus epochs, and confusion matrices, for best results for each method.
# 
# Write at least 10 sentences about what you observe in the likelihood plots, the train, validation and test accuracies, and the confusion matrices.

# ## Check-In
# 
# Tar or zip your jupyter notebook (`<name>-A4.ipynb`) and your python script file (`neuralnetworksA4.py`) into a file named `<name>-A4.tar` or `<name>-A4.zip`.  Check in the tar or zip file in Canvas.

# # Grading
# 
# Download [A4grader.tar](https://www.cs.colostate.edu/~anderson/cs545/notebooks/A4grader.tar), extract `A4grader.py` before running the following cell.
# 
# Remember, you are expected to design and run your own tests in addition to the tests provided in `A4grader.py`.

# In[10]:


get_ipython().run_line_magic('run', '-i A4grader.py')


# # Extra Credit
# 
# Repeat the above experiments with a different data set.  Randonly partition your data into training, validaton and test parts if not already provided.  Write in markdown cells descriptions of the data and your results.
