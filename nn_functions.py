import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1 + np.e ** -z)


def forward_prop(Theta, x_train):
    # Similar to Theta make list A which will hold the activations  through
    # the network and will be returned to be used with back_prop
    x_train = np.append([1], x_train) # add in the bias unit
    A = [x_train]
    for i in range(len(Theta)):
        theta = Theta[i]
        a = sigmoid(np.matmul(theta,A[i]))
        if i < len(Theta) -1:
            a = np.append([1], a) # add in the bias unit on every layer 
                                  # except output
        A.append(a)
    return A

def back_prop(Theta, x_train, y_train):
    A = forward_prop(Theta, x_train)
    last_delta = A[-1] - y_train # delta of the output layer
    Delta = list() # empty list to hold delta (cap.delta)
    for i in range(len(Theta)):
        print('Iteration of the bp loop %s\n'  %(str(i)))
        previous_a = A[-1 - (i + 1)]# lose the bias
        if not i == 0: #except for output layer, lose the bias
            last_delta = last_delta[1:] 
        previous_delta = np.matmul(np.transpose(Theta[-1 - i]), last_delta) \
                * previous_a * (1 - previous_a)
        Delta.insert(0,np.outer(last_delta, previous_a))
        last_delta = previous_delta

    return Delta

def cost_function(Theta, X_list, 

        
        

        
