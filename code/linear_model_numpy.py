import numpy as np 
import pandas as pd 

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

def forward_fn(Beta, Z):
    """
        Linear regression forward function.
        Implements the equation y(x) = b0 + b1 * x1 + b2 * x2 + ...
        
        Inputs:
            Beta: the coefficients of the model (size K)
            Z: the input feature matrix, as returned by prepare_inputs (size NxK)
        Output:
            yhat: the model's predictions (size N)
    """
    return Z @ Beta

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

def loss_fn(Beta, Z, y):
    """
        Computes the mean squared error loss function for the model.

        Inputs:
            Beta: the coefficients of the model (size K)
            Z: the input feature matrix, as returned by prepare_inputs (size NxK)
            y: actual observations (size N)
        Output:
            mse: mean squared error
    """
    yhat = forward_fn(Beta, Z)
    mse = np.mean(np.square(yhat - y))
    return mse 

def optimize(input_df, y, learning_rate, epochs):
    """
        Input parameters:
            input_df: dataframe containing input columns
            y: a vector of outputs that we wish to predict
            learning_rate: how quickly we want gradient descent learning
            epochs: the number of steps of gradient descent
        Output:
            Beta: fitted model parameters
    """
    
    # Prepare our inputs into the linear regression
    Z = prepare_inputs(input_df) # NxK

    # Randomly initialize our solution
    Beta = np.random.randn(Z.shape[1]) # K

    # Run gradient descent loop
    for i in range(epochs):

        # Compute model's predictions
        yhat = forward_fn(Beta, Z) # N

        # Compute the gradient at those predictions
        # Z is NxK
        # yhat is N
        # y is N
        # KxN @ N = K
        Beta_grad = 2 * Z.T @ (yhat - y) / Z.shape[0]
        
        # Update the parameters
        Beta = Beta - learning_rate * Beta_grad

    # Beta is the fitted parameter values
    return Beta

