import numpy as np
import pandas as pd
import nn_functions as nf
from nn_functions import sigmoid
from scipy.optimize import minimize
import scipy.io

#..................................................#
# Decide on Neural Network Parameters
hl = [5] # hidden layers (eg. [10,10] means 2 layers, 10 units each)
lam = 0
number_output_class = 3#10

if 1 == 0:
    # Import our MNIST test data
    train_df = pd.read_csv('Data/train.csv')
    ######## TEMPORARY!!!!! 
    train_df = train_df.drop(np.arange(100,42000), axis=0)

    m = float(train_df.shape[0]) # number of training samples

    # Transfer to numpy arrays
    X_train = np.array(train_df.drop(['label'], axis=1))
    Y_train = np.array(train_df['label'])
    Y_train = nf.one_hot(Y_train,number_output_class)

    print X_train.shape

    # l is a vector which describes the number of units in all layers of the nn
    l = np.append( [np.ma.shape(X_train)[1]], np.append(hl, [number_output_class])) 
    # l = [ # units input (784 for our pictures), hidden layer sizes, # output units]

    # Weights matrices (Theta is a list of 2d matrices)
    # the matrices have shape [# units next layer, # units previous layer]
    Theta = []    
    for i in range(len(l) - 1):
        theta_2d = (np.random.rand(l[i+1], l[i] + 1) - .5)/10
        Theta.append(theta_2d)
elif 1 ==1:
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = nf.debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = nf.debugInitializeWeights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X_train  = nf.debugInitializeWeights(m, input_layer_size - 1)
    y  = np.array([2,3,1,2,3]) #1 + mod(1:m, num_labels)'
    Y_train = nf.one_hot(y,number_output_class)
    #print(' norms are: %f, %f, %f, %f' %(np.linalg.norm(Theta1,2), np.linalg.norm(Theta2,2), \
     #                                    np.linalg.norm(X,2), np.linalg.norm(y,2)))
    Theta = [Theta1, Theta2]

else:
    # MATLAB STUFF
    # Getting theta
    Theta_ml = scipy.io.loadmat('matlab_stuff/ex4weights.mat')
    Theta = [Theta_ml['Theta1'], Theta_ml['Theta2']]

    # Getting data
    X_ml = scipy.io.loadmat('matlab_stuff/ex4data1.mat')
    X_train = X_ml['X']
    X_train = X_train[:500,:]
    Y_train = X_ml['y']
    Y_train = Y_train[:500]
    Y_train = nf.one_hot(Y_train,number_output_class)

Theta_sizes = list() # list of Theta shapes so it can be re-rolled
for i in range(len(Theta)):
    Theta_sizes.append(Theta[i].shape)

# Unroll theta
unrolled_Theta = nf.unroll_np_list(Theta)

# Testing
J = nf.cost_function(unrolled_Theta, X_train, Y_train, lam, Theta_sizes)
print J

# lambda functions for the minimizer
cost = lambda var_theta: nf.cost_function(var_theta,X_train,Y_train,lam, Theta_sizes)
grad = lambda var_theta: nf.gradient_function(var_theta,X_train,Y_train,lam,Theta_sizes)

# Gradient checking
epsilon = 1.0e-4
analyt_grad = grad(unrolled_Theta)
num_grad = np.zeros(len(analyt_grad))
print('Calculating numerical gradient')
perturb = np.zeros(len(num_grad))
for i in range(len(unrolled_Theta)):
    if i % 50 == 0:
        print( 'We are on Theta value %f' %float(i))
    perturb[i] = epsilon
    plus = cost(unrolled_Theta + perturb)
    minus = cost(unrolled_Theta - perturb)
    print plus
    print minus
    print nf.cost_function(unrolled_Theta, X_train, Y_train, lam, Theta_sizes)
    num_grad[i] = (plus - minus) / (2.0 * epsilon)
    perturb[i] = 0.0

print np.linalg.norm(analyt_grad - num_grad)/np.linalg.norm(analyt_grad+num_grad)
print num_grad
print analyt_grad
exit()

sol = minimize(cost, unrolled_Theta, method = 'CG',\
               options={'disp': True})

print sol


    

