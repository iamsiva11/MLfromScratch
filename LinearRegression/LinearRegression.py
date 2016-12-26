
class LinearRegression:

    def __init__(self):
        self.dataset  =  [ [1, 1], [2, 3], [4, 3], [3, 2], [5, 5] ]
        self.coef     =   [0.4, 0.8]  #theta0, theta1


    def predict(self,row):
        yhat = self.coef[0]

        for i in range(len(row) - 1):
            yhat += self.coef[1] * row[i]
        return yhat


    def predict_difference(self,row): #rsudua value
        yhat = row[1]
        pred = self.predict(row)
        return abs(yhat-pred)

    def cost_function(self):
        
        squared_sum=0
        for row in self.dataset:
            squared_sum += (self.predict_difference(row))**2

        
        return squared_sum


    # Estimate linear regression coefficients using stochastic gradient descent
    def findCoefficients_sgd(self, train, l_rate, n_epoch):
        self.coef = [0.0 for i in range(len(train[0]))]

        k=0 #coefficients Index
        for epoch in range(n_epoch):
            sum_error = 0

            for row in train:
                yhat = self.predict(row)
                error = yhat - row[1] #or row[-1]
                
                sum_error += error ** 2

                self.coef[k] = self.coef[k] - l_rate * error
                self.coef[k+1] = self.coef[k+1] - l_rate * error * row[k+1]
                
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

        return self.coef


if __name__ == "__main__":

    dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    # coef    = [0.4, 0.8]  # theta0,theta1

    lr = LinearRegression()

    n_epoch = 50
    l_rate = 0.001
    coef = lr.findCoefficients_sgd(dataset, l_rate, n_epoch)

    print(coef)


