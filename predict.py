import numpy as np
import pandas as pd
import nn_functions as nf
from nn_functions import sigmoid
from scipy.optimize import minimize
import scipy.io

#..........................................#
# Import weight and testing data

# Weights
unrolled_Theta = np.loadtxt('weights')
print np.linalg.norm(unrolled_Theta,2)

# Test data (matlab data test/train)
X_ml = scipy.io.loadmat('matlab_stuff/ex4data1.mat')
X_train = X_ml['X']
print X_train.shape
X_train = X_train[:500,:]
Y_train = X_ml['y']
Y_train = Y_train[:500]
#Y_train = nf.one_hot(Y_train,number_output_class)

Theta = reroll_np_list(unrolled_Theta, Theta_sizes)
h = nf.h_theta(unrolled_Theta, X_train)
