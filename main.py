import numpy as np
import pandas as pd
import nn_functions as nf
from nn_functions import sigmoid
from scipy.optimize import minimize

#..................................................#
# Decide on Neural Network Parameters
hl = [5] # hidden layers (eg. [10,10] means 2 layers, 10 units each)
lam = 1e-5
number_output_class = 10

# Import our MNIST test data
train_df = pd.read_csv('Data/train.csv')
######## TEMPORARY!!!!! 
train_df = train_df.drop(np.arange(100,42000), axis=0)


m = float(train_df.shape[0]) # number of training samples


# Transfer to numpy arrays
X_train = np.array(train_df.drop(['label'], axis=1))
Y_train = np.array(train_df['label'])
Y_train = nf.one_hot(Y_train,number_output_class)

# l is a vector which describes the number of units in all layers of the nn
l = np.append( [np.ma.shape(X_train)[1]], np.append(hl, [number_output_class])) 
# l = [ # units input (784 for our pictures), hidden layer sizes, # output units]

# Weights matrices (Theta is a list of 2d matrices)
# the matrices have shape [# units next layer, # units previous layer]
Theta = []    
for i in range(len(l) - 1):
    theta_2d = np.random.rand(l[i+1], l[i] + 1) /100
    Theta.append(theta_2d)

#Delta =nf.back_prop(Theta, X_train[1,:], y_train)

#J = nf.cost_function(Theta,X_train,Y_train, lam)
#A = nf.forward_prop(Theta,X_train[0,:])

#D = nf.gradient_function(Theta,X_train,Y_train, lam)

Theta_sizes = list() # list of Theta shapes so it can be re-rolled
for i in range(len(Theta)):
    Theta_sizes.append(Theta[i].shape)

# Unroll theta
unrolled_Theta = nf.unroll_np_list(Theta)

# lambda functions for the minimizer
cost = lambda var_theta: nf.cost_function(var_theta,X_train,Y_train,lam, Theta_sizes)
grad = lambda var_theta: nf.gradient_function(var_theta,X_train,Y_train,lam,Theta_sizes)


unrolled_Theta[1] = unrolled_Theta[1]
print cost(unrolled_Theta)

unrolled_Theta[1] = unrolled_Theta[1] + 10
print cost(unrolled_Theta)
exit() 

# Gradient checking
eps = 1.0e-8
analyt_grad = grad(unrolled_Theta)
num_grad = np.zeros(len(analyt_grad))
for i in range(len(unrolled_Theta)):
    plus_Theta = np.copy(unrolled_Theta)
    plus_Theta[i] = plus_Theta[i] + eps
    min_Theta = np.copy(unrolled_Theta)
    min_Theta[i] = min_Theta[i] - eps 
    num_grad[i] = (cost(plus_Theta) - cost(min_Theta))/(2.0 * eps)

print num_grad
print np.linalg.norm(analyt_grad)
print abs(analyt_grad - num_grad)
exit()

sol = minimize(cost, unrolled_Theta, method = 'CG',\
               jac = grad, options={'disp': True})

print sol


    

