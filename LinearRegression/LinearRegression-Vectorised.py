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


if __name__== "__main__":
	X = np.array([ [1,1], [2,1], [4,1], [3,1], [5,1] ])
	y = np.array([1,3,3,2,5])
	lr = LinReg(.1,100,1)
	print lr.fit(X,y)   

