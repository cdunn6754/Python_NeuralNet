import numpy as np
import pandas as pd
import nn_functions as nf
from nn_functions import sigmoid

#..................................................#
# Decide on Neural Network Parameters
hl = [20,10] # hidden layers (eg. [10,10] means 2 layers, 10 units each)

# Import our MNIST test data
train_df = pd.read_csv('Data/train.csv')

# Transfer to numpy arrays
X_train = np.array(train_df.drop(['label'], axis=1))
Y_train = np.array(train_df['label'])

# NN weights matrix (a list of 2-d arrays
l = np.append( [np.ma.shape(X_train)[1]], np.append(hl, [10])) 
# l = [ # units input (784 for our pictures), hidden layer sizes, # output units]

# Weights matrices (Theta is a list of 2d matrices)
Theta = []    
for i in range(len(l) - 1):
    theta_2d = np.random.rand(l[i+1], l[i] + 1) /100
    Theta.append(theta_2d)

y_train = [0 ,1, 0, 0, 0, 0, 0, 0, 0, 0]
Delta =nf.back_prop(Theta, X_train[1,:], y_train)

for i in Delta:
    print(np.shape(i))
