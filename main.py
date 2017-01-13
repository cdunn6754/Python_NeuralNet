import numpy as np
import pandas as pd
import nn_functions as nf
from nn_functions import sigmoid

#..................................................#
# Decide on Neural Network Parameters
hl = [20,10] # hidden layers (eg. [10,10] means 2 layers, 10 units each)
lam = 1e-5
number_output_class = 10

# Import our MNIST test data
train_df = pd.read_csv('Data/train.csv')
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

J = nf.cost_function(Theta,X_train,Y_train, lam)
#A = nf.forward_prop(Theta,X_train[0,:])

# Loop for backprop, turned into a mess
# Have to do the first one out of loop to get the list going
# then do nested loops to add matrices from other training examples to 
# the running sum
Delta = nf.back_prop(Theta,X_train[0,:],Y_train[:,0])
trials = np.arange(1,int(m)-1)
for i in trials:
    temp_Delta = nf.back_prop(Theta,X_train[i,:], Y_train[:,i])
    for j in range(len(Delta)):
        mat = Delta[j]
        mat = mat + temp_Delta[j]

# Make a Theta without the bias weights
reg_Theta = Theta
for i in range(len(reg_Theta)):
    reg_Theta[i][:,0] = 0

print reg_Theta[2][:,0]

    

