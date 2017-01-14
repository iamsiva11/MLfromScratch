"""
The program creates an neural network that simulates the 
exclusive OR function with two inputs and one output. 
"""

import numpy as np

def nonlinear(x, deriv=False):
    if (deriv == True):
        return x*(1-x)    
    return 1/(1+np.exp(-x))

#Input data
X=np.array([ [0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])


#Output data
y = np.array([[0,1,1,0]]).T
# y = np.array([[0],
#               [1],
#               [1],
#               [0]
#              ])


#Useful for debugging and lots of other purposes
np.random.seed(1)

#Synapses
syn0 = 2*np.random.random((3,4)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*np.random.random((4,1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.

#Training proces starts here
#60000 iterations / epochs
for j in xrange(60000):           
    # Calculate forward through the network.        
    #The forward propogaton process. Multiply the weights with the synapses in each layer
    l0 = X
    l1 = nonlinear(np.dot(l0,syn0))
    l2 = nonlinear(np.dot(l1,syn1))
    
    #To calculate the errror - We subtract the actual value with the expected value
    l2_error = y - l2
    
    # Only print the error every 10000 steps, to save time and limit the amount of output. 
    if(j % 10000==0):
        print "Error:" + str(np.mean(np.abs(l2_error)))
        
    #Back Propogation starts 
    l2_delta = l2_error*nonlinear(l2, deriv=True )    
    l1_error = np.dot(l2_delta, syn1.T)    
    l1_delta = l1_error * nonlinear(l1,deriv=True )
        
    #update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print "Output after training"
print l2

#See how the final output closely approximates the true output [0, 1, 1, 0]. 
#If you increase the number of interations in the training loop (currently 60000), the final output will be even closer. 
