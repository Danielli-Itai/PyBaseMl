#!/usr/bin/env python
# coding: utf-8





#Packages.
# - [numpy](https://www.numpy.org/) is the fundamental package for scientific computing with Python.
import numpy as np





"""
Compute the sigmoid of z
Arguments:  z -- A scalar or numpy array of any size.
Return:     s -- sigmoid(z)
"""
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s



def Accuracy(Y, Ypredicted):
    return float((np.dot(Y, Ypredicted) + np.dot(1 - Y, 1 - Ypredicted)) / float(Y.size) * 100)
