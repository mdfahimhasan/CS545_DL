
import numpy as np
import optimizers as opt


class NeuralNetwork():
    """
    A class that represents a neural network for nonlinear regression.

    Attributes
    ----------
    n_inputs : int
        The number of values in each sample
    n_hidden_units_by_layers : list of ints, or empty
        The number of units in each hidden layer.
        Its length specifies the number of hidden layers.
    n_outputs : int
        The number of units in output layer
    all_weights : one-dimensional numpy array
        Contains all weights of the network as a vector
    Ws : list of two-dimensional numpy arrays
        Contains matrices of weights in each layer,
        as views into all_weights
    all_gradients : one-dimensional numpy array
        Contains all gradients of mean square error with
        respect to each weight in the network as a vector
    Grads : list of two-dimensional numpy arrays
        Contains matrices of gradients weights in each layer,
        as views into all_gradients
    performance_trace : list of floats
        Mean square error (unstandardized) after each epoch
    n_epochs : int
        Number of epochs trained so far
    X_means : one-dimensional numpy array
        Means of the components, or features, across samples
    X_stds : one-dimensional numpy array
        Standard deviations of the components, or features, across samples
    T_means : one-dimensional numpy array
        Means of the components of the targets, across samples
    T_stds : one-dimensional numpy array
        Standard deviations of the components of the targets, across samples
        
        
    Methods
    -------
    make_weights_and_views(shapes)
        Creates all initial weights and views for each layer

    train(X, T, n_epochs, method='sgd', learning_rate=None, verbose=True)
        Trains the network using input and target samples by rows in X and T

    use(X)
        Applies network to inputs X and returns network's output
    """

    def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):
        """Creates a neural network with the given structure

        Parameters
        ----------
        n_inputs : int
            The number of values in each sample
        n_hidden_units_by_layers : list of ints, or empty
            The number of units in each hidden layer.
            Its length specifies the number of hidden layers.
        n_outputs : int
            The number of units in output layer

        Returns
        -------
        NeuralNetwork object
        """

        # Assign attribute values. 
        # Set performance_trace to [].
        # Set self.X_means to None to indicate
        # that standardization parameters have not been calculated.
        self.n_inputs = n_inputs
        self.n_hidden_units_by_layers = n_hidden_units_by_layers
        self.n_outputs = n_outputs

        self.performance_trace = []
        
        self.n_epochs = 0
        self.rho = None
        
        self.X_means = None
        self.X_stds = None
        self.T_means = None
        self.T_stds = None
  

        # Build list of shapes for weight matrices in each layer
        if n_hidden_units_by_layers:
            # 1st layer weight shape
            shapes = [(n_inputs+1, n_hidden_units_by_layers[0])]  
            # 2nd to n-1 layer weight shape
            for i in range(len(n_hidden_units_by_layers)-1):
                shapes.append((n_hidden_units_by_layers[i]+1, n_hidden_units_by_layers[i+1])) 
            # last (n) layer weight
            shapes.append((n_hidden_units_by_layers[-1]+1, n_outputs))  
            
        else: 
            shapes = [(n_inputs+1, n_outputs)]  # shape of weight if no hidden layer
        
            
        # Call make_weights_and_views to create all_weights and Ws
        
        self.all_weights, self.Ws = self.make_weights_and_views(shapes)

        # Call make_weights_and_views to create all_gradients and Grads
        
        self.all_gradients, self.Grads = self.make_weights_and_views(shapes)


    def make_weights_and_views(self, shapes):
        """Creates vector of all weights and views for each layer

        Parameters
        ----------
        shapes : list of pairs of ints
            Each pair is number of rows and columns of weights in each layer.
            Number of rows is number of inputs to layer (including constant 1).
            Number of columns is number of units, or outputs, in layer.

        Returns
        -------
        Vector of all weights, and list of views into this vector for each layer
        """
        # Create one-dimensional numpy array of all weights with random initial values
        num_in_weights = sum([np.prod(s) for s in shapes])
        all_weights = np.random.uniform(-1, 1, size=(num_in_weights, 1)).flatten()

        # Build weight matrices as list of views (pairs of number of rows and number 
        # of columns) by reshaping corresponding elements from vector of all weights 
        # into correct shape for each layer. 
        first_idx = 0
        Ws = []
        for shp in shapes:
            nw = shp[0] * shp[1]
            last_idx = first_idx + nw
            Ws.append(all_weights[first_idx:last_idx].reshape(shp))
            first_idx = last_idx
        
        # Divide values of each weight matrix by square root of number of its inputs.
        for w in Ws:
            w[:] /= np.sqrt(w.shape[0] + 1)
        
        # Set output layer weights to zero.
        Ws[-1][:] = np.zeros(shapes[-1])
        
        return all_weights, Ws
    
    
    def __repr__(self):
        return 'NeuralNetwork({}, {}, {})'.format(self.n_inputs, self.n_hidden_units_by_layers, self.n_outputs)

    def __str__(self):
        s = self.__repr__()
        if self.n_epochs > 0:  # self.total_epochs
            s += '\n Trained for {} epochs.'.format(self.n_epochs)
            s += '\n Final standardized training error {:.4g}.'.format(self.performance_trace[-1])
        return s
 
    def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
        """Updates the weights.

        Parameters
        ----------
        X : two-dimensional numpy array 
            number of samples  by  number of input components
        T : two-dimensional numpy array
            number of samples  by  number of output components
        n_epochs : int
            Number of passes to take through all samples
        method : str
            'sgd', 'adam', or 'scg'
        learning_rate : float
            Controls the step size of each update, only for sgd and adam
        verbose: boolean
            If True, progress is shown with print statements
        """
        self.learning_rate = learning_rate
        
        # Calculate and assign standardization parameters
        self.X_means = X.mean(axis = 0)
        self.X_stds = X.std(axis = 0)
        self.T_means = T.mean(axis = 0)
        self.T_stds = T.std(axis = 0)

        # Standardize X and T.  Assign back to X and T.
        X = (X - self.X_means) / self.X_stds
        T = (T - self.T_means) / self.T_stds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = opt.Optimizers(self.all_weights)

        # Define function to convert mean-square error to root-mean-square error,
        # Here we use a lambda function just to illustrate its use.  
        # We could have also defined this function with
        # def error_convert_f(err):
        #     return np.sqrt(err)

        error_convert_f = lambda err: np.sqrt(err) 
        
        # Call the requested optimizer method to train the weights.
        if method == 'sgd':
            
            performance_trace = optimizer.sgd(self.error_f, self.gradient_f,
                                              fargs=[X, T], n_epochs=n_epochs,
                                              learning_rate=learning_rate,
                                              error_convert_f=error_convert_f, 
                                              error_convert_name='RMSE',
                                              verbose=verbose)

        elif method == 'adam':

            performance_trace = optimizer.adam(self.error_f, self.gradient_f,
                                               fargs=[X, T], n_epochs=n_epochs,
                                               learning_rate=learning_rate,
                                               error_convert_f=error_convert_f, 
                                               error_convert_name='RMSE',
                                               verbose=verbose)

        elif method == 'scg':

            performance_trace = optimizer.scg(self.error_f, self.gradient_f,
                                              fargs=[X, T], n_epochs=n_epochs,
                                              error_convert_f=error_convert_f, 
                                              error_convert_name='RMSE',
                                              verbose=verbose)

        else:
            raise Exception("method must be 'sgd', 'adam', or 'scg'")

        self.n_epochs += len(performance_trace)
        self.performance_trace += performance_trace

        # Return neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

        return self

    def _add_ones(self, X):
        return np.insert(X, 0, 1, 1)
    
    def _forward(self, X):
        """Calculate outputs of each layer given inputs in X.
        
        Parameters
        ----------
        X : input samples, standardized.

        Returns
        -------
        Standardized outputs of all layers as list, include X as first element.
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
        if self.n_hidden_units_by_layers:
            self.Zs.append(self._add_ones(Z_w) @ weights[-1]) 
        else:
            self.Zs.append(self._add_ones(X) @ weights[-1])
            
        return self.Zs
        

    # Function to be minimized by optimizer method, mean squared error
    def error_f(self, X, T):
        """Calculate output of net given input X and its mean squared error.
        Function to be minimized by optimizer.

        Parameters
        ----------
        X : two-dimensional numpy array, standardized
            number of samples  by  number of input components
        T : two-dimensional numpy array, standardized
            number of samples  by  number of output components

        Returns
        -------
        Standardized mean square error as scalar float that is the mean
        square error over all samples and all network outputs.
        """
        # Call _forward, calculate mean square error and return it.
        Y = self._forward(X)[-1]
        mse = np.mean(((T - Y)) **2)
        return mse
         

    # Gradient of function to be minimized for use by optimizer method
    def gradient_f(self, X, T):
        """Returns gradient wrt all weights. Assumes _forward already called
        so input and all layer outputs stored in self.Zs

        Parameters
        ----------
        X : two-dimensional numpy array, standardized
            number of samples  x  number of input components
        T : two-dimensional numpy array, standardized
            number of samples  x  number of output components

        Returns
        -------
        Vector of gradients of mean square error wrt all weights
        """
        # Assumes forward_pass just called with layer outputs saved in self.Zs.
        
        n_samples = X.shape[0]
        n_outputs = T.shape[1]
        n_layers = len(self.n_hidden_units_by_layers) + 1

        # delta is delta matrix to be back propagated.
        # Dividing by n_samples and n_outputs here replaces the scaling of the learning rate.
        
        delta = (T - self.Zs[-1]) / (n_samples * n_outputs)

        # Step backwards through the layers to back-propagate the error (delta)
        
        for layeri in range(n_layers - 1, -1, -1):
            
            # gradient of all but bias weights
            Z = self.Zs[layeri]
            W = self.Ws[layeri]
            self.Grads[layeri][:] = -(self._add_ones(Z).T @ delta)
            
            # Back-propagate this layer's delta to previous layer
            if layeri > 0:
                delta = delta @ W[1:, :].T * (1 - Z ** 2)
            
        return self.all_gradients

    def use(self, X):
        """Return the output of the network for input samples as rows in X.
        X assumed to not be standardized.

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  by  number of input components, unstandardized

        Returns
        -------
        Output of neural network, unstandardized, as numpy array
        of shape  number of samples  by  number of outputs
        """

        # Standardize X
        Xs = (X - self.X_means) / self.X_stds
        
        Ys = self._forward(Xs)[-1]
        # Unstandardize output Y before returning it
        Y = Ys * self.T_stds + self.T_means
        
        return Y

    def get_performance_trace(self):
        """Returns list of unstandardized root-mean square error for each epoch"""
        return self.performance_trace
