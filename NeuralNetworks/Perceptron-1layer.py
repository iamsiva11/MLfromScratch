import numpy as np

n_iter =10
lrate=0.01
w_ = np.zeros( 1 + X.shape(1) )
errors = []

def net_input(X):
	return np.dot(X,w_[1]) + w[0]

#Logistic / Sigmoid function
def  predict(x):
	return np.where( net_input(x) >= 0,0, 1, -1 )	