#!/usr/bin/env python
# coding: utf-8
import os
import ast
import numpy as np
import matplotlib.pyplot as plt

from PyBase import Files





PARAMS_BIAS = 'b'
PARAMS_WEIGHTS = 'w'





# initialize_with_zeros
"""
This function creates a vector of zeros of shape
 (dim_rows, dim_colums) for w
 (dim_colums) for b.

Argument:
dim -- size of the w vector we want (or number of parameters in this case)

Returns:
w -- initialized vector of shape (dim, 1)
b -- initialized scalar (corresponds to the bias)
"""
def ParamsInitialize(dim_rows, dim_colums):
    params_dic = {}
    w = np.zeros(shape=(dim_rows, dim_colums))
    b = np.zeros(shape=(dim_colums))

    assert(w.shape == (dim_rows, dim_colums))
    assert(b.shape == (dim_colums,))

    params_dic[PARAMS_BIAS] = b
    params_dic[PARAMS_WEIGHTS] = w

    return params_dic

def ParamsSave(file_name:str, params:dict):
    params_dic={}
    params_dic['w'] = params['w'].tostring()#','.join(['%.f' % num for num in params['w']])
    params_dic['w_shape'] = params['w'].shape

    params_dic['b'] = params['b'].tostring()#','.join(['%.f' % num for num in params['b']])
    params_dic['b_shape'] = params['b'].shape

    text = str(params_dic)
    Files.TextWrite(file_name, text)
    print(str(params))
    return


def ParamsRead(file_name:str):
    params_dic = ast.literal_eval(Files.TextRead(file_name))

    params={}
    params['w'] = np.fromstring(params_dic['w'])
    params['w'] = np.reshape(params['w'] , params_dic['w_shape'] )

    params['b'] = np.fromstring(params_dic['b'])
    params['b'] = np.reshape(params['b'] , params_dic['b_shape'] )

    print(str(params_dic))
    return params




# Center and standardize your dataset.
# meaning that you substract the mean of the whole numpy array from each example,
# and then divide each example by the standard deviation of the whole numpy array.
def Standardise(train_set_x_flatten):
    train_set_x = (train_set_x_flatten - train_set_x_flatten.mean())/train_set_x_flatten.std()  #train_set_x = train_set_x_flatten / 255.
    assert(train_set_x.shape == train_set_x_flatten.shape)
    return train_set_x


"""
Compute the sigmoid of z
Arguments:  z -- A scalar or numpy array of any size.
Return:     s -- sigmoid(z)
"""
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s








# propagate
"""
Implement the cost function and its gradient for the propagation explained above

Arguments:
w -- weights, a numpy array of size (num_px * num_px * 3, 1)
b -- bias, a scalar
X -- data of size (num_px * num_px * 3, number of examples)
Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

Return:
cost -- negative log-likelihood cost for logistic regression
dw -- gradient of the loss with respect to w, thus same shape as w
db -- gradient of the loss with respect to b, thus same shape as b

Tips:
- Write your code step by step for the propagation. np.log(), np.dot()
"""


def propagate(w, b, X, Y):
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation

    np_dot1 = np.dot(np.log(A), Y.T)
    np_dot2 = np.dot(np.log(1 - A), (1 - Y.T))
    cost = -np.sum(np_dot1 + np_dot2) / m  # compute cost
    cost = np.squeeze(cost)

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = np.dot(X, (A - Y).T) / m
    db = (A - Y).sum() / m

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw, "db": db}

    return grads, cost





# optimize
"""
This function optimizes w and b by running a gradient descent algorithm

Arguments:
w -- weights, a numpy array of size (num_px * num_px * 3, 1)
b -- bias, a scalar
X -- data of shape (num_px * num_px * 3, number of examples)
Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
num_iterations -- number of iterations of the optimization loop
learning_rate -- learning rate of the gradient descent update rule
print_cost -- True to print the loss every 100 steps

Returns:
params -- dictionary containing the weights w and bias b
grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
"""


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 10 == 0:
            costs.append(cost)
            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs





# predict
'''
Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

Arguments:
w -- weights, a numpy array of size (num_px * num_px * 3, 1)
b -- bias, a scalar
X -- data of size (num_px * num_px * 3, number of examples)

Returns:
Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
'''
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction[0, i] = (A[0, i] >= 0.5)

    assert (Y_prediction.shape == (1, m))
    return Y_prediction


# model
"""
Builds the logistic regression model by calling the function you've implemented previously

Arguments:
X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
print_cost -- Set to true to print the cost every 100 iterations

Returns:
d -- dictionary containing information about the model.
"""
MODEL_PARAMS = 'parameters'
def regression_rmodel(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros (≈ 1 line of code)
    # w = w.reshape(X.shape[0], 1)
    w, b = ParamsInitialize(X_train.shape[0], 1)

    # Gradient descent (≈ 1 line of code)
    # Returns parameters w and b, gradients and costs.
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(parameters["w"], parameters["b"], X_test)
    Y_prediction_train = predict(parameters["w"], parameters["b"], X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    model = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             MODEL_PARAMS:parameters,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

    return model




def CostsPlot(costs:list, learning_rate:int, name:str, PLOTS_DIR):
    # Let's also plot the cost function and the gradients Plot learning curve (with costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.savefig(os.path.join(PLOTS_DIR, name))
    plt.show()
    plt.close()
    return





# #### Choice of learning rates ####
def LearningRates(train_set_x, train_set_y, test_set_x, test_set_y, learning_rates):
    models = {}
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        models[str(i)] = regression_rmodel(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    return models


def LearningRatesCostsPlot(models:dict, learning_rates:np.array, name, PLOTS_DIR):
    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.savefig(os.path.join(PLOTS_DIR, name))
    plt.show()
    plt.close()
    return
