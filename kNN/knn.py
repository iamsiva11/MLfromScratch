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

#1) given two data points, calculate the euclidean distance between them
#Works even for data in the multidimensional space

def get_distance(data1, data2):#(data point 1, data point 2)
    points = zip(data1, data2)    
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]    
    return math.sqrt(sum(diffs_squared_distance))


#Helper method
def _get_tuple_distance(training_instance, test_instance):
     return (training_instance, get_distance(test_instance, training_instance))


#2) given a training set and a test instance, use getDistance 
#to calculate all pairwise distances
def get_neighbours(training_set, test_instance, k):
    distances = [_get_tuple_distance(training_instance, test_instance) for training_instance in training_set]
    # index 1 is the calculated distance between training_instance and test_instance
    sorted_distances = sorted(distances, key=itemgetter(1))
    # extract only training instances
    sorted_training_instances = [tuple[0] for tuple in sorted_distances]
    # select first k elements
    return sorted_training_instances[:k]


if __name__=="__main__":
	training_set = [[1,3],[4,7],[9,1],[4,8]]
	test_instance= [4,8]
	print [_get_tuple_distance(training_instance, test_instance) for training_instance in training_set]	
	print get_neighbours(training_set, test_instance, 3)