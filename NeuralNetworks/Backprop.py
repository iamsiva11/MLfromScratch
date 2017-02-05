"""
#PEP Standard
#Professional Code,Idioms
#Python Pro -Start
#Start small, make it a habit
#Todo,Habit
#adding docstrings , space between var,commas
#to improve readability
"""
import numpy as np

MAX_NUMBER_OF_EPOCHS = 60000

class NeuralNet:
    
    def __init__(self,lrate=0.001,hidden=2):
        self.learning_rate = lrate        
        self.hidden = hidden
        
        np.random.seed(1)
        self.W1 = 2*np.random.random((3,4))-1
        self.W2 = 2*np.random.random((4,1))-1
                
    """Acticvation Functions"""
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def softmax(self, x):
        e_x= np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    """Forward Propogation"""
    def forward(self, X):
        """
        X - Input matrix
        Returns:
        y1,y2 - Hidden ,output layer states        
        """
        z1 = np.dot(X, self.W1)
        y1 = self.sigmoid(z1)      
        z2= np.dot(y1,self.W2)
        y2= self.sigmoid(z2)
        
        return y1, y2        
        
#         print x
#         print z1
#         print y1

    """Backward Propogation"""
    def  backward(self, X, y, y1, y2):
    	"""
    	X - Input matrix
    	y - label/target
    	y1,y2 - Hidden ,output layer states        
    	Returns:
    	der_W1, der_W2 - Derivatives of W1,W2 respectively
    	"""

        der_W1 = X.T.dot((y2 - y).dot(self.W2.T) * y1 * (1. - y1))
        der_W2 = y1.T.dot(y2 - y)
        
        return der_W1, der_W2

    """Weight update"""
    def update_weights(self, der_W1, der_W2):
        self.W1 -= self.learning_rate * der_W1
        self.W2 -= self.learning_rate * der_W2
        
    
    def flow(self,X,y):

        for _ in range(len(X)):            
            #Forward Propogation
            y1,y2 = self.forward(X)
            #Backprop
            der_W1,der_W2 = self.backward(X, y, y1, y2)
            self.update_weights(der_W1, der_W2)
    
        return y2                       
    
    def train(self,X,y):
        #Training Starts
        for epoch in range(MAX_NUMBER_OF_EPOCHS):
            #print "\nEpoch: %d" % epoch
            res = self.flow(X,y)                        
        return res 


    #error calculation
    def error(self):
        pass

if __name__=="__main__":
    X=np.array([ [0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])
    y = np.array([[0,1,1,0]]).T
    net = NeuralNet()
    print  net.train(X,y)