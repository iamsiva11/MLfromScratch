"""
Vectorised implementation - using linear algebra library numpy

"""
import numpy as np

class Linalg:

	def __init__(self, alpha,iters,lambda_):
		self.alpha = alpha
		self.iters = iters
		self.lambda_ = lambda_
		self.theta = None

	def predict(self, X, Y):
		return np.dot(X,self.theta)

	def cost(self,X,Y):
		return np.linalg.norm(self.predict(X) - Y)
	
	def fit(self, X, Y)	:
		n,dim = X.shape
		self.theta = np.seroes(dim)

		for _ in range(self.iters):
			self.theta -= self.alpha * ( (self.predict(X) - y).dot(X)/n )
			#with lambda
			#self.theta -= self.alpha * ( (self.predict(X) - y).dot(X)/n +self.lambda_ * self.theta)

		return self.theta	

