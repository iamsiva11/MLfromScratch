"""
TODO 
Py -> Scala
Clean, Pro Code - OOPs, fun style, Coding standards ,etc

"""

import numpy as np

def squared_error(actual, predicted): 
	return np.power( np.array(actual) - np.array(predicted), 2)

def sumsquarederror(actual,predicted): 
	return np.sum(np.power( np.array(actual) - np.array(predicted), 2))

if name=="main":

	actual =[1,3,5,7]
	predicted =[1.3,4,5.8,7.9]
	print squared_error(actual,predicted)
	print sum_squared_error(actual,predicted)