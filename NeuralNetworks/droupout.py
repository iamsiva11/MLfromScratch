import numpy as np

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([ [0,1,1,0] ])

alpha = 0.5
hidden_dim = 4 
dropout_percent = 0.2
do_dropout= True
no_iter=10000


def sigmoid(x, deriv = False):
	if deriv==True:
		return x*(1-x)
	return 1/(1 + np.exp(-x))

synapse_0 =  2*np.random.random((3, hidden_dim)) - 1
synapse_1 =	 2*np.random.random((hidden_dim ,1)) - 1


#Code to demonstrate dropout

#for j in xrange(no_iter):
if do_dropout:
	layer_1 = sigmoid(np.dot(X,synapse_0))
	print layer_1
	
	layer_1 *=\
	np.random.binomial( [np.ones((len(X),hidden_dim ))] , 1-dropout_percent)[0]*\
	(1.0 / (1-dropout_percent))

	print layer_1

	# np.random.binomial usage
	# binomial(n,p)
	# n, p = 10, .5 # number of trials, probability of each trial



"""
#Output

#Before Drppout 
[[ 0.63633049  0.33841654  0.72158035  0.5116083 ]
 [ 0.46193294  0.38605251  0.54979871  0.68812217]
 [ 0.4915929   0.55826432  0.55268867  0.56112388]
 [ 0.32176605  0.60838852  0.36797435  0.72921388]]

#After Drppout 
[[ 0.79541311  0.          0.          0.63951038]
 [ 0.          0.48256564  0.68724839  0.86015271]
 [ 0.61449113  0.69783039  0.          0.70140484]
 [ 0.          0.          0.45996793  0.91151735]]



"""
