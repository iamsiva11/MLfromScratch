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

