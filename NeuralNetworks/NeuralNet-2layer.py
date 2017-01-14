import numpy as numpy

def nonlinear(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+(np.exp(-x)))

#input
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

#output
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
#initialize weights randomly with mean 0 , 3X1 matrix
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(10000):
    l0 = X
    l1 = nonlinear(np.dot(l0,syn0))
    l1_error = y-l1
    
    #backprop starts
    l1_delta = l1_error * nonlinear(l1,True)
    #update the weights
    syn0 += np.dot(l0.T,l1_delta)

print l1

#Output
# Output after training
# [[ 0.00966449]
#  [ 0.00786506]
#  [ 0.99358898]
#  [ 0.99211957]]

