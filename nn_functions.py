import numpy as np
import matplotlib.pyplot as plt

# sigmoid for forward prop
def sigmoid(z):
    return 1/(1 + np.e ** -z)

# get rid of the bias weigthts in theta_vector (set them to 0)
def make_reg_Theta(Theta):
    reg_Theta = Theta
    for i in range(len(Theta)):
        reg_Theta[i][:,0] = 0
    return reg_Theta

# Unroll a list of np arrays
def unroll_np_list(list1):
    unrolled_list1 = np.array([0]) # need to initate with dummy
    for i in range(len(list1)):
        unrolled_list1 = np.hstack((unrolled_list1, list1[i].flatten()))
    unrolled_list1 = np.delete(unrolled_list1, 0, 0) # get rid of dummy
    return unrolled_list1

# Reroll 1d array into list of np arrays
def reroll_np_list(nparray,mat_sizes):
    index = 0 # keep track of where we are in the nparray
    unrolled_nparray = list()
    for i in range(len(mat_sizes)):
        n = mat_sizes[i][0] # num of rows
        m = mat_sizes[i][1] # num of cols
        mat = np.reshape(nparray[index:index + n*m],(n,m))
        unrolled_nparray.append(mat)
        index = index + n*m
    return unrolled_nparray

# unroll the solution vector into matrix of one-hot vectors
# returns them in matrix shape [outputclasses, number of training examples]
def one_hot(vec, num_classes):
    t_examples = len(vec)  #training examples
    one_hot_mat = np.zeros([t_examples, num_classes]) 
    column_start_index = np.arange(t_examples) * num_classes
    one_hot_mat.flat[column_start_index + vec.ravel()] = 1
    return np.transpose(one_hot_mat)

# Forward prop for cost function that can run n training examples simultaneously
# and returns only the output of the nn, not every activation
def h_theta(Theta, X_train):
    h = np.hstack((np.ones((X_train.shape[0],1)), X_train)) #add bias
    h = np.transpose(h)
    for i in range(len(Theta)):
        h = np.matmul(Theta[i],h)
        h = sigmoid(h)
        if i < (len(Theta) -1): # add bias but not to the final output
            h = np.vstack((np.ones((1,h.shape[1])),h))
    return h


# forward prop for use with backprop (one training example at a time)
def forward_prop(Theta, x_train):
    # Similar to Theta make list A which will hold the activation vectors
    # through the network and will be returned to be used with back_prop
    x_train = np.append([1], x_train) # add in the bias unit
    A = [x_train]
    # loop through layers but do so for every training example at once
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
    Delta = list() # empty list to hold delta (cap. delta)
    for k in range(len(Theta)):
        previous_a = A[-1 - (k + 1)]# lose the bias
        if not k == 0: #except for output layer, lose the bias
            last_delta = last_delta[1:] 
        previous_delta = np.matmul(np.transpose(Theta[-1 - k]), last_delta) \
                * previous_a * (1 - previous_a)
        Delta.insert(0,np.outer(last_delta, previous_a))
        last_delta = previous_delta

    return Delta

def cost_function(unrolled_Theta, X_train, Y_train, lam,Theta_sizes):
    Theta = reroll_np_list(unrolled_Theta, Theta_sizes)
    m = float(X_train.shape[0]) # # of training examples
    h = h_theta(Theta, X_train) # returns matrix of h(theta) in shape 
                                # [output features, training examples]
    
    # Regularlization
    reg = 0
    for theta in Theta:
        reg = sum(sum(theta[:,1:]**2)) + reg

    # Compute the Cost function J
    J = (1/m) * sum(sum(-Y_train * np.log(h) - (1 - Y_train) * np.log(1 - h))) \
        + (lam/(2*m)) * reg
    
    return J


def gradient_function(unrolled_Theta,X_train, Y_train, lam, Theta_sizes):
    Theta = reroll_np_list(unrolled_Theta, Theta_sizes)
    m = float(X_train.shape[0]) # number of training samples
    
    # Loop for backprop, turned into a mess
    # Have to do the first one out of loop to get the list going
    # then do nested loops to add matrices from other training examples to 
    # the running sum
    print ('Calculating gradients')
    Delta = back_prop(Theta,X_train[0,:],Y_train[:,0])
    trials = np.arange(1,int(m)-1) # go through trials excluding the first one
    for i in trials:
        #print ('Working on trial number %s' %(str(i)))
        temp_Delta = back_prop(Theta,X_train[i,:], Y_train[:,i])
        for j in range(len(Delta)): # go through all matrices in Delta
            Delta[j] = Delta[j] + temp_Delta[j]

    # Dont regularize the bais weights
    reg_Theta = make_reg_Theta(Theta)

    # Unroll reg_Theta and Delta (horizontal first, then vertical)
    # note that they are the same size list
    # each element of the two lists is a matrix, and the matrix 
    # sizes correspond between the lists as well
    unrolled_reg_Theta = unroll_np_list(reg_Theta)
    unrolled_Delta = unroll_np_list(Delta)

    D = (1/m)*unrolled_Delta + (lam/m) * unrolled_reg_Theta # gradient
    return D


        
    

        
