#The program creates an neural network that simulates the 
#exclusive OR function with two inputs and one output. 

import numpy as np

#non-linear sigmoid function
def nonlinear(x, deriv=False):
    if (deriv == True):
        return x*(1-x)
    
    return 1/(1+np.exp(-x))

#input data
X=np.array([ [0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])

#Output data
y = np.array([[0],
              [1],
              [1],
              [0]
             ])


#Useful for debugging and lots of other purposes
np.random.seed(1)

#synapses
syn0 = 2*np.random.random((3,4)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*np.random.random((4,1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.


