"""
The kNN task can be broken down into writing 3 primary functions:
1/ Calculate the distance between any two points
2/ Find the nearest neighbours based on these pairwise distances
(Sort the training Data points based on the distance)
3/ Majority vote on a class labels based on the nearest neighbour list
(Get the max nearest neighbour (i.e. the majority vote)
-Sort the top k data points)
"""

import math
from operator import itemgetter
from collections import Counter
import numpy as np


class knn:

	def __init__(self,k=None):
		self.training_set = [[1,3],[4,7],[9,1],[4,8],[5,7],[2,4],[2,7]]
		self.test_instance= [4,8]
		self.k=k


	#Helper method for _get_tuple_distance
	#1) given two data points, calculate the euclidean distance between them
	#Works even for data in the multidimensional space
	def get_distance(self,data1, data2):#(data point 1, data point 2)
	    points = zip(data1, data2)    
	    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]    
	    return math.sqrt(sum(diffs_squared_distance))

	
	def _get_tuple_distance(self,training_instance):
	    return (training_instance, self.get_distance(self.test_instance, training_instance))

	#2) given a training set and a test instance, use getDistance 
	#to calculate all pairwise distances
	def get_neighbours(self):
	    distances = [self._get_tuple_distance(training_instance) for training_instance in self.training_set]
	    # index 1 is the calculated distance between training_instance and test_instance
	    sorted_distances = sorted(distances, key=itemgetter(1))
	    print sorted_distances
	    # extract only training instances
	    sorted_training_instances = [tuple[0] for tuple in sorted_distances]
	    # select first k elements
	    return sorted_training_instances[:self.k]


class knnClassifier:

	#For Classification
	def get_majority_vote(self,neighbours_list):
		#We are gonna make index 1 as the class label here
		classes = [neighbour[1] for neighbour in neighbours_list]
		count= Counter(classes)
		return count.most_common()[0][0]

class knnRegressor:	

	#For Regression
	def get_mean_value(self,neighbours): 
		#Assuming we are taking the classes as the index 1 here 
		classes = [neighbour[1] for neighbour in neighbours]
		#mean or median value of the classes is calculated here
		return np.mean(classes)	


if __name__=="__main__":

	#get_neighbours() will proint the sorted distances of each training instance
	neighbours= knn(3).get_neighbours()
	print neighbours	

	#Classification
	print knnClassifier().get_majority_vote(neighbours)

	#Regression
	print knnRegressor().get_mean_value(neighbours)





