import numpy as np 
import jax 
import jax.numpy as jnp 
import pandas as pd 
import sklearn.metrics

def prepare_inputs(input_df):
    """
        Prepares the input features that will be fed into the model.

        Inputs:
            input_df: the input dataframe into the function. Should consist ONLY of input features.
        Outputs:
            Z: the input feature matrix of size NxK, where K is the number of features
    """
    # Let's identify categorical columns in a dataframe
    categorical_cols = input_df.select_dtypes(include='object').columns
    
    # Let's identify the numeric columns in the dataframe
    numeric_cols = input_df.select_dtypes(include='number').columns

    # We want to construct the input features into the model
    # We will use a numpy array that contains both numeric and categorically encoded values
    X = input_df[numeric_cols].to_numpy() # (NxK)
    
    # Now we need to z-score the numeric features so that they can lead to efficient learning
    col_means = np.mean(X, axis=0) # K
    col_stds = np.std(X, axis=0, ddof=1) # K
    
    # Z-score
    # (NxK - 
    #  1xK) 
    #  / 
    #  (1xK)
    Z = (X - col_means[None, :]) / col_stds[None, :]
    
    # Now we want to code the categorical columns using one-hot encoding
    for col in categorical_cols:
        # NxC (C is the number of unique values in the column)
        # So for origin this will be Nx3 
        dummies = pd.get_dummies( input_df[col] ).to_numpy() 
        
        # concatenate dummies matrix onto Z
        #print(Z.shape)
        #print(dummies.shape)
        Z = np.hstack((Z, dummies)) 
    
    # finally we want to add a column of ones at the start of Z
    ones_col = np.ones((Z.shape[0], 1)) # Nx1
    
    Z = np.hstack((ones_col, Z))

    return Z

def forward_fn(params, Z):
    """
        The MLP forward function.
        This is a neuron network with one hidden layer and ReLU activations.
        
        Inputs:
            params: the weight of the model
            Z: the input feature matrix, as returned by prepare_inputs (size NxK)
        Output:
            yhat: the model's predictions (size N)
    """
    # compute the inputs into the neurons
    # 1xH + Nxd @ dxH = 1xH + NxH = NxH
    hidden_inputs = params['b_hidden'][None,:] + Z @ params['W_input_hidden']
    
    # next, we apply the non-linearity to those inputs
    #hidden_activations = jnp.tanh(hidden_inputs) # NxH
    hidden_activations = jnp.maximum(0, hidden_inputs) # <-- ReLU
    
    # now we compute the output
    # 1 + NxH @ H = 1 + N = N 
    f = params['b_output'] + hidden_activations @ params['W_hidden_output']
    p = 1/(1+jnp.exp(-f)) # <-- sigmoid function (the probability of a positive)

    # so that the log doesn't blow up
    p = jnp.clip(p, 0.01, 0.99)

    return p

def predict(params, input_df):
    """
        Convienience function that prepares inputs and runs the forward function.

        Inputs:
            params: the weights of the model
            input_df: input data frame (input features only, no output column).
        Output:
            yhat: the model's predictions (size N)
    """
    Z = prepare_inputs(input_df)
    return forward_fn(params, Z)

def loss_fn(params, Z, y):
    """
        Computes the mean squared error loss function for the model.

        Inputs:
            params: the weights of the model
            Z: the input feature matrix, as returned by prepare_inputs (size NxK)
            y: actual observations (size N)
        Output:
            mse: mean squared error
    """
    p = forward_fn(params, Z) # N

    # Log Probability of each data point, given the model (shape N)
    log_probability_of_data = y * jnp.log(p) + (1-y) * jnp.log(1-p)

    # We want to maximize the log probability of the data under the model
    # but gradient descent minimizes a loss
    # So we want to minimize the negative log probability
    loss = -jnp.mean(log_probability_of_data)
    
    return loss 

def optimize(rng, input_df, y, learning_rate, epochs, n_hidden):
    """
        Input parameters:
            rng: the random key
            input_df: dataframe containing input columns
            y: a vector of outputs that we wish to predict
            learning_rate: how quickly we want gradient descent learning
            epochs: the number of steps of gradient descent
            n_hidden: the number of hidden units
        Output:
            params: fitted model parameters
    """

    # To make this work, we need to convert y to jax
    y = jnp.array(y)
    
    # the magic: the gradient function
    # Creates a function that can calculate the  gradient of the model
    grad_fn = jax.grad(loss_fn) 
    
    # Prepare our inputs into the linear regression
    Z = prepare_inputs(input_df) # NxK

    #
    # Initialize the parameters
    #
    params = {} # initialize an empty dictionary
    # weights from inputs to hidden neurons (dxH)
    n_inputs = Z.shape[1]
    params['W_input_hidden'] = jax.random.normal(rng, (n_inputs, n_hidden)) / jnp.sqrt(n_inputs)
    rng, _ = jax.random.split(rng)  # move to the next random key
    
    # bias of the hidden neurons is initialized to zero
    params['b_hidden'] = jnp.zeros(n_hidden)
    
    # weights from the hidden neurons to the output neuron (shape is H)
    params['W_hidden_output'] = jax.random.normal(rng, (n_hidden,)) / jnp.sqrt(n_hidden)
    rng, _ = jax.random.split(rng) # move to next random key
    
    # finally, initialize the bias of the output neuron 
    params['b_output'] = 0.0
    
    # Run gradient descent loop
    for i in range(epochs):

        # Compute the gradient of the loss function with respect
        # to all model parameters
        W_grad = grad_fn(params, Z, y)
        
        # Update the parameters
        for key in params:
            params[key] = params[key] - W_grad[key] * learning_rate

    # params is the fitted parameter values
    return params

def mlp_train_test_function(rng,
                            train_df, 
                            test_df, 
                            input_cols, 
                            output_col,
                            n_hidden):
    """
        Function that trains an MLP and tests it. Returns multiple evaluation metrics.

        Inputs:
            rng: Random number key
            train_df: training data frame
            test_df: testing data frame 
            input_cols: features to use
            output_col: output to predict
            n_hidden: number of hidden units 
        Outputs:
            results: A dictionary containing accuracy, accuracy_null, auc_roc, auc_pr, auc_pr_null
    """
    # build the training input data frame
    train_input_df = train_df[input_cols]

    # build the training outputs
    y = train_df[output_col].to_numpy()
    
    # Optimize the model using gradient descent
    best_params = optimize(rng = rng,
                           input_df = train_input_df,
                           y = y,
                           learning_rate = 0.1,
                           epochs = 100,
                           n_hidden = n_hidden)

    # build the testing input data frame
    test_input_df = test_df[input_cols]

    # Make predictions on the test set
    yhat = predict(params = best_params, input_df = test_input_df)
    
    # Calculate error of those predictions
    ytest = test_df[output_col].to_numpy()

    # We will return multiple metrics
    yhat_hard = yhat > 0.5

    # this is our null model, and it is based only on the training set
    yhat_null = jnp.mean(y) * jnp.ones_like(ytest)
    yhat_hard_null = yhat_null > 0.5
    
    return dict(
        accuracy = sklearn.metrics.accuracy_score(ytest, yhat_hard),
        accuracy_null = sklearn.metrics.accuracy_score(ytest, yhat_hard_null),
        auc_roc = sklearn.metrics.roc_auc_score(ytest, yhat), # soft decision
        auc_pr = sklearn.metrics.average_precision_score(ytest, yhat),
        auc_pr_null = sklearn.metrics.average_precision_score(ytest, yhat_null)
    )

def factory(rng, input_cols, output_col, n_hidden):

    def train_test_fn(train_df, test_df):

        return mlp_train_test_function(rng,
                                  train_df = train_df,
                                  test_df = test_df,
                                  input_cols = input_cols,
                                  output_col = output_col,
                                  n_hidden=n_hidden)

    return train_test_fn
