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

@jax.jit # <-- JAX
def forward_fn(Beta, Z):
    """
        Logistic regression forward function.
        Implements the equation f(x) = b0 + b1 * x1 + b2 * x2 + ...
                                y(x) = sigmoid(f(x))
        Inputs:
            Beta: the coefficients of the model (size K)
            Z: the input feature matrix, as returned by prepare_inputs (size NxK)
        Output:
            yhat: the model's predictions (size N)
    """
    f = Z @ Beta 
    p = 1/(1+jnp.exp(-f))
    return p 

def predict(Beta, input_df):
    """
        Convienience function that prepares inputs and runs the forward function.

        Inputs:
            Beta: the coefficients of the model (size K)
            input_df: input data frame (input features only, no output column).
        Output:
            yhat: the model's predictions (size N)
    """
    Z = prepare_inputs(input_df)
    return forward_fn(Beta, Z)

@jax.jit # <-- JAX
def loss_fn(Beta, Z, y):
    """
        Computes the negative cross-entropy loss function for the model.

        Inputs:
            Beta: the coefficients of the model (size K)
            Z: the input feature matrix, as returned by prepare_inputs (size NxK)
            y: actual observations (size N)
        Output:
            loss: negative binary cross entropy
    """
    yhat = forward_fn(Beta, Z)
    loss = -jnp.mean(y * jnp.log(yhat) + (1-y) * jnp.log(1-yhat))
    return loss 

def optimize(rng, input_df, y, learning_rate, epochs):
    """
        Input parameters:
            rng: JAX random key
            input_df: dataframe containing input columns
            y: a vector of outputs that we wish to predict
            learning_rate: how quickly we want gradient descent learning
            epochs: the number of steps of gradient descent
        Output:
            Beta: fitted model parameters
    """
    
    # move y into jax's domain
    y = jnp.array(y) # <-- JAX

    # Create a function that computes the gradient of the loss_fn with respect to the first argument (Beta)
    grad_fn = jax.grad(loss_fn) # <-- JAX

    # Prepare our inputs into the linear regression
    Z = prepare_inputs(input_df) # NxK

    # Randomly initialize our solution
    Beta = jax.random.normal(key = rng, shape=Z.shape[1]) # <-- JAX
    
    # Run gradient descent loop
    for i in range(epochs):

        # compute gradient
        # this is very powerful ... JAX takes care of derivative computation
        # so loss_fn (and forward_fn) could be as complex as you like
        # as long as they are reasonably continous
        Beta_grad = grad_fn(Beta, Z, y) # <-- JAX
        
        # Update the parameters
        Beta = Beta - learning_rate * Beta_grad

    # Beta is the fitted parameter values
    return Beta

# We will create little function that takes the training dataframe and the testing dataframe
# and returns MSE on test
def lr_train_test_function(rng, train_df, test_df, input_cols, output_col):

    # build the training input data frame
    train_input_df = train_df[input_cols]

    # build the training outputs
    y = train_df[output_col].to_numpy()
    
    # Optimize the model using gradient descent
    best_Beta = optimize(rng,
                         input_df = train_input_df,
                         y = y,
                         learning_rate = 0.1,
                         epochs = 100)

    # build the testing input data frame
    test_input_df = test_df[input_cols]

    # Make predictions on the test set
    yhat = predict(Beta = best_Beta,
                   input_df = test_input_df)
    
    # Calculate error of those predictions
    ytest = test_df[output_col].to_numpy()
    
    # we'll use accuracy for now
    return sklearn.metrics.accuracy_score(ytest, yhat > 0.5)
