import numpy as np

#sigmoid
def sigmoid(x):
    return 1 / ( 1+np.exp(-x) )

def sigmoid_d(x):
    return x*(1-x)

#tanh
def tanh(x):
    return np.tanh(x)
    
def tanh_d(x):
    return (1 - (x**2))

#relu
def ReLU(x):
    #return 1 if x>0 else 0    
    return x * (x>0)

def ReLU_d(x):
    return 1 * (x>0)



#step
""" ReLu_d is same as step function"""
def step(x):
	return 1 * (x>0)

def softmax(x)	:
	e_x= np.exp(x-np.max(x))
	return e_x / e_x.sum()

    

if __name__=="__main__":
    
    X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    np.random.seed(1)
    #weights/Synapses
    #initialize weights randomly with mean 0 , 3X1 matrix
    w1 = 2*np.random.random((3,4)) - 1

   	#Layer0
    l0=X
    #Layer1
    z1= np.dot(l0,w1)
    #print z1
    
    #Appply activation function
    print sigmoid_d(z1)
    print tanh(z1)