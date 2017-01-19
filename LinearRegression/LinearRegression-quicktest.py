import numpy as np

xi = np.array([1,2,4,3,5])
A = np.array([ xi, np.ones(5)])
y = np.array([1,3,3,2,5])


#Obtaining the parameters/weights/theta
w = np.linalg.lstsq(A.T,y)[0] 

theta1,theta0 = w #m,c 

print "theta0 or c:" + str(theta0)
print "theta1 or m:" + str(theta1)





