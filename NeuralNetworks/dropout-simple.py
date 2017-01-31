"""
Simple dropout demonstartation in 
layer 2 of a 3 layer Network
"""
import numpy as np

X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
#Output
y = np.array([[0,0,1,1]]).T

#For reproducability
np.random.seed(1)

#Initialise the weights/Synapses
w1 = 2*np.random.random((3,4)) - 1
w2 = 2*np.random.random((4,4)) - 1
w3 = 2*np.random.random((4,1)) - 1

#Input Layer /Layer0
l0=X

#Layer1
z1= np.dot(l0,w1)
y1= np.tanh(z1)

#Layer2
z2 = np.dot(y1, w2)
y2 = np.tanh(z2)

#We are gonna add dropout in layer 3
""" 
#/*The Dropout Code - Start
"""
m2 = np.random.binomial(1, 0.5, size=z2.shape)
y2 *= m2
""" 
The Dropout Code - End*/
"""
#Layer3
z3 = np.dot(y2, w3)
y3 = np.tanh(z3) # linear output


if __name__=="__main__":

	print m2 #Dropout  matrix
	print z3
	print y3

"""
Output:
[[1 1 1 0]
 [1 1 0 1]
 [1 1 0 1]
 [0 0 1 0]]
[[-0.11229247]
 [-0.71851192]
 [-0.2409984 ]
 [-0.73753644]]
[[-0.11182285]
 [-0.6159867 ]
 [-0.23643856]
 [-0.62765443]]
"""