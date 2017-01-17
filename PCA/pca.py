import numpy as numpy

class PCA:
	def __init__:(self, k):
		self.k = k
		self.means = None
		self.w = None

	def fit(self, X, y=None):
		self.means = X.mean(axis=0)
		cov = np.cov(X.T)
		eig_val_cov, eig_vec_cov = np.linalg.eig(cov)
		eig_pairs = list(zip(eig_val_cov,eig_vec_cov))
		top_pairs = sorted(eig_pairs, reverse=True)[:self.k]
		self.w = np.vstack( pair[1] for pair in top_pairs)
		return self

	def transform(self,X):
		return self.w.dot(X.T).T

if __name__=="__main__":

	#Input matrix X
	X = np.array([ [ 8.5, 79 , 3.5, 0],
               [ 8.5, 86 ,5.5, 1],
               [ 3.5, 99 ,6.5, 3],
               [ 3.5 ,111, 6.5, 1],
               [ 5.5 ,78 , 8.5 ,2] ])

	pca=PCA(2)
	pca.fit(X)
	print pca.transform(X)