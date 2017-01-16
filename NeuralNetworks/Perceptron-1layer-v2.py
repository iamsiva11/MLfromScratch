import numpy as numpy
import random

def generate_data(no_of_points):
	"""
	no_of_points - No of training points to generate
	"""
	X = np.zeros(shape = (no_of_points, 2 ))
	Y = np.zeros(shape = no_of_points)

	np.random.seed(1)

	for ii in range(no_of_points):
		X[ii][0] = random.randint(1,9) + 0.5
		X[ii][1] = random.randint(1,9) + 0.5
		Y[ii] = 1 if X[ii][0]+X[ii][1] >= 13 else -1
	return X,Y	

def perceptron(X, Y, b=0 , max_iter = 10):
	"""
	b = bias 
	X - input train data . n rows(training eg.) and m columns (features)
	"""
	n,m = np.shape(X)
	#weight vector
	w = np.zeros(m)

	for ii in range(max_iter):
		for jj in range(n):
			x_i = X[jj] #i'th training instance - input
			y_i = Y[jj] #i'th training instance - output

			a = b + np.dot(w, x_i)

			#error update
			if np.sign(y_i*a) !=1:
				w += y_i * x_i
				b += y_i

		print ("iteration %s: new weight_vector: %s : new bias: %s " % (ii, w ,b))



if __name__=="__main__":
	X,Y = generate_data(10)
	perceptron(X,Y)








