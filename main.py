import numpy as np
import pandas as pd
import nn_functions as nf
from nn_functions import sigmoid
from scipy.optimize import minimize
import scipy.io

#..................................................#
# Decide on Neural Network Parameters
hl = [5] # hidden layers (eg. [10,10] means 2 layers, 10 units each)
lam =1.0
number_output_class = 10

if 1 == 0:
    # Import our MNIST test data
    train_df = pd.read_csv('Data/train.csv')
    ######## TEMPORARY!!!!! 
    #train_df = train_df.drop(np.arange(100,42000), axis=0)

    m = float(train_df.shape[0]) # number of training samples
    print('Number of training examples: %f' %m)

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
        theta_2d = (np.random.rand(l[i+1], l[i] + 1) - .5)/10
        Theta.append(theta_2d)
elif 1 ==0:
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
    rand = np.random.rand(2000)
    rand = np.round(rand*5000).astype(int)
    X_ml = scipy.io.loadmat('matlab_stuff/ex4data1.mat')
    X_train = X_ml['X']
    X_train = X_train[rand,:]
    Y_train = X_ml['y']
    Y_train = Y_train[rand]
    Y_train = nf.one_hot(Y_train,number_output_class)

Theta_sizes = list() # list of Theta shapes so it can be re-rolled
for i in range(len(Theta)):
    Theta_sizes.append(Theta[i].shape)

# Unroll theta
unrolled_Theta = nf.unroll_np_list(Theta)

# lambda functions for the minimizer
cost = lambda var_theta: nf.cost_function(var_theta,X_train,Y_train,lam, Theta_sizes)
grad = lambda var_theta: nf.gradient_function(var_theta,X_train,Y_train,lam,Theta_sizes)

# print('Testing')
# print cost(unrolled_Theta)
# exit()
solve = 1
if solve == True:
    print('CG time')
    sol = minimize(cost, unrolled_Theta, jac = grad, method = 'CG',\
                   options={'disp': True})

    solution = sol.x
    print(np.linalg.norm(solution,2))

    np.savetxt('weights', solution, fmt='%1.5f')


### PREDICT

#..........................................#
# Import weight and testing data

# Weights
unrolled_Theta = np.loadtxt('weights')
print np.linalg.norm(unrolled_Theta,2)

# Test data (matlab data test/train)
X_ml = scipy.io.loadmat('matlab_stuff/ex4data1.mat')
X_train = X_ml['X']
X_train = X_train[rand,:]
Y_train = X_ml['y']
Y_train = Y_train[rand]
Y_train_oh = nf.one_hot(Y_train,number_output_class)
print Y_train_oh[:,0]
print Y_train[0]
exit()

Theta = nf.reroll_np_list(unrolled_Theta, Theta_sizes)
h = nf.h_theta(Theta, X_train)

output = np.zeros(len(h[0,:]))

print h[:,2]
print h[:,3]
exit()

for i in range(len(output)):
    output[i] = np.argmax(h[:,i]) + 1
    if output[i] == Y_train[i]:
        output[i] == 1.0
    else:
        output[i] == 0

print output
print( 'The accuracy is:')
print np.mean(output)
    




    

