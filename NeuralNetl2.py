#!/usr/bin/env python
# coding: utf-8





#Packages.
# Let's first import all the packages that you will need during this assignment.
# - [numpy](https://www.numpy.org/) is the fundamental package for scientific computing with Python.
# - [sklearn](http://scikit-learn.org/stable/) provides simple and efficient tools for data mining and data analysis.
# - [matplotlib](http://matplotlib.org) is a library for plotting graphs in Python.
import os
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt

from PyBaseMl import BaseMl




#####################################################
#                                                   #
#                   Plots                           #
#                                                   #
#####################################################
# Visualize 2d grid the dataset using matplotlib.
"""
X_0 - horizontal coodinate
X_1 - vertical coordinate.
Y - Dot color
scatter - size.
"""
def plot_dataset(X_0, X_1, Y, scatter, title:str):
    plt.scatter(X_0, X_1, c=Y, s=scatter, cmap=plt.cm.Spectral)
    return

def plot_decision_boundary(model, X, y, path:str, title:str):

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0, :], cmap=plt.cm.Spectral)

    plt.savefig(os.path.join(path,title+'.jpg'))
    return





#####################################################
#                                                   #
#               Calculate Layer Sizes               #
#                                                   #
#####################################################

# FUNCTION: layer_sizes
"""
Arguments:  X -- input dataset of shape (input size, number of examples)
            Y -- labels of shape (output size, number of examples)
Returns:    n_x -- the size of the input layer
            n_h -- the size of the hidden layer
            n_y -- the size of the output layer
"""
def layer_sizes(X, Y):
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_h, n_y)


#####################################################
#                                                   #
#               Initialize parameters               #
#                                                   #
#####################################################

# Initialize the model's parameters
# - Initialize the weights matrices with random values.
#     - Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
# - Initialize the bias vectors as zeros.
#     - Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.
# FUNCTION: initialize_parameters

"""
Argument:
n_x -- size of the input layer
n_h -- size of the hidden layer
n_y -- size of the output layer

Returns:
params -- python dictionary containing your parameters:
                W1 -- weight matrix of shape (n_h, n_x)
                b1 -- bias vector of shape (n_h, 1)
                W2 -- weight matrix of shape (n_y, n_h)
                b2 -- bias vector of shape (n_y, 1)
"""
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters





#####################################################
#                                                   #
#               Forward Propagate                   #
#                                                   #
#####################################################

# GRADED FUNCTION: forward_propagation
"""
Argument:
X -- input data of size (n_x, m)
parameters -- python dictionary containing your parameters (output of initialization function)

Returns:
A2 -- The sigmoid output of the second activation
cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
"""
def forward_propagation(X, parameters):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.matmul(W1, X) + b1
    A1 = np.tanh(Z1)

    Z2 = np.matmul(W2, A1) + b2
    A2 = BaseMl.sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache





#####################################################
#                                                   #
#               Calculate cost                      #
#                                                   #
#####################################################

# (you can use either `np.multiply()` and then `np.sum()` or directly `np.dot()`).
# FUNCTION: compute_cost
"""
Computes the cross-entropy cost given in equation (13)

Arguments:      A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
                Y -- "true" labels vector of shape (1, number of examples)
parameters -- python dictionary containing your parameters W1, b1, W2 and b2
[Note that the parameters argument is not used in this function, 
but the auto-grader currently expects this parameter.
Future version of this notebook will fix both the notebook 
and the auto-grader so that `parameters` is not needed.
For now, please include `parameters` in the function signature,
and also when invoking this function.]

Returns:
cost -- cross-entropy cost given equation (13)

"""
def compute_cost(A2, Y, parameters):
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))
    cost = - np.sum(logprobs) / m

    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
                                    # E.g., turns [[17]] into 17
    assert(isinstance(cost, float))
    return cost


#####################################################
#                                                   #
#               Backwords propagation               #
#                                                   #
#####################################################

# FUNCTION: backward_propagation
"""
Backward propagation
Arguments:  parameters -- python dictionary containing our parameters 
            cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
            X -- input data of shape (2, number of examples)
            Y -- "true" labels vector of shape (1, number of examples)
Returns:    grads -- python dictionary containing your gradients with respect to different parameters
"""
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = np.matmul(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.matmul(W2.T, dZ2)
    dZ1 = np.multiply(np.matmul(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = np.matmul(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads





#####################################################
#                                                   #
#               Update parameters                   #
#                                                   #
#####################################################

# FUNCTION: update_parameters
"""
Updates parameters using the gradient descent update rule given above

Arguments:  parameters -- python dictionary containing your parameters 
            grads -- python dictionary containing your gradients 

Returns:    parameters -- python dictionary containing your updated parameters 
"""
def update_parameters(parameters, grads, learning_rate = 1.2):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']

    W2 = parameters['W2']
    b2 = parameters['b2']

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters





#####################################################
#                                                   #
#               Neural netwok model                 #
#                                                   #
#####################################################

# FUNCTION: nn_model
"""
Arguments:  X -- dataset of shape (2, number of examples)
            Y -- labels of shape (1, number of examples)
            n_h -- size of the hidden layer
            num_iterations -- Number of iterations in gradient descent loop
            print_cost -- if True, print the cost every 1000 iterations

Returns:    parameters -- parameters learnt by the model. They can then be used to predict.
"""
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters





#####################################################
#                                                   #
#               Prediction                          #
#                                                   #
#####################################################
# FUNCTION: predict
"""
Using the learned parameters, predicts a class for each example in X

Arguments:
parameters -- python dictionary containing your parameters 
X -- input data of size (n_x, m)

Returns
predictions -- vector of predictions of our model (red: 0 / blue: 1)
"""
def predict(parameters, X):
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    return predictions



