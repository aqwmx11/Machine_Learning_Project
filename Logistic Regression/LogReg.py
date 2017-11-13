# -*- coding: utf-8 -*-

import numpy as np
class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 50):
        '''
        Initializes Parameters of the  Logistic Regression Model
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
  
    def calculateGradient(self, weight, X, Y, regLambda):

        '''
        Computes the gradient of the objective function
        Arguments:
        
        X is a n-by-(d+1) numpy matrix
        Y is an n-by-1 dimensional numpy matrix
        weight is (d+1)-by-1 dimensional numpy matrix
        regLambda is the scalar regularization constant
        Returns:
        the gradient, an (d+1)-by-1 dimensional numpy matrix
        '''

        #basic setup, get the dimension, initalize the gradient matrix
        d=X.shape[1]-1
        n=X.shape[0]
        Gradient=np.zeros((d+1,1))

        #calculate the normalized matrix h
        h=self.sigmoid(X*np.mat(weight))

        #calculate the gradient for theta 0
        for i in range(0,n):
            Gradient[0,0]+=h[i,0]-Y[i,0]

        #calculate the gradient for other thetas
        for j in range(1,d):
            for i in range(0,n):
                Gradient[j,0]+=(h[i,0]-Y[i,0])*X[i,j]+regLambda*weight[j,0]

        return Gradient    

    def sigmoid(self, Z):
        '''
        Computes the Sigmoid Function  
        Arguments:
            A n-by-1 dimensional numpy matrix
        Returns:
            A n-by-1 dimensional numpy matrix
       
        '''
        #basic setup, get the dimension n, create result matrix
        n=Z.shape[0]
        sigmoid=np.ones((n,1))

        #loop to calculate each element
        for i in range(0,n):
            sigmoid[i,0]=1/(1+np.exp(-Z[i,0]))

        return sigmoid

    def update_weight(self,X,Y,weight):
        '''
        Updates the weight vector.
        Arguments:
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is a d+1-by-1 dimensional numpy matrix
        Returns:
            updated weight vector : (d+1)-by-1 dimensional numpy matrix
        '''
        new_weight = weight-self.alpha*self.calculateGradient(weight, X, Y, self.regLambda)
        
        return new_weight
    
    def check_conv(self,weight,new_weight,epsilon):
        '''
        Convergence Based on Tolerance Values
        Arguments:
            weight is a (d+1)-by-1 dimensional numpy matrix
            new_weights is a (d+1)-by-1 dimensional numpy matrix
            epsilon is the Tolerance value we check against
        Return : 
            True if the weights have converged, otherwise False

        '''
        temp=0
        d=weight.shape[0]-1
        for i in range(0,d+1):
            temp+=pow((weight[i,0]-new_weight[i,0]),2)
        temp=np.sqrt(temp)
        if temp<=epsilon:
            return True
        return False
        
    def train(self,X,Y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            Y is an n-by-1 dimensional numpy matrix
        Return:
            Updated Weights Vector: (d+1)-by-1 dimensional numpy matrix
        '''
        # Read Data
        n,d = X.shape
        
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        self.weight = self.new_weight = np.zeros((d+1,1))

        # Train the parameters
        loopNum=0
        while True:
            print("A new loop begins!")
            self.new_weight=self.update_weight(X,Y,self.weight)
            loopNum+=1
            # if converge, then break the loop
            if (self.check_conv(self.weight,self.new_weight,self.epsilon)):
                break
            else:
                self.weight=self.new_weight
            # if exceeds max loop number, break too
            if loopNum > self.maxNumIters:
                print("Max loop limits triggered!")
                break

        return self.weight

    def predict_label(self, X,weight):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
            weight is a d+1-by-1 dimensional matrix
        Returns:
            an n-by-1 dimensional matrix of the predictions 0 or 1
        '''
        #data
        n=X.shape[0]
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        
        Fit=self.sigmoid(X*np.mat(weight))
        
        result=np.zeros((n,1))
        for i in range(0,n):
            if Fit[i,0]>=0.5:
                result[i,0]=1
            else:
                result[i,0]=0
        
        return result
    
    def calculateAccuracy (self, Y_predict, Y_test):
        '''
        Computes the Accuracy of the model
        Arguments:
            Y_predict is a n-by-1 dimensional matrix (Predicted Labels)
            Y_test is a n-by-1 dimensional matrix (True Labels )
        Returns:
            Scalar value for accuracy in the range of 0 - 100 %
        '''
        n=Y_predict.shape[0]

        rightCase=0
        for i in range(0,n):
            if Y_predict[i,0]==Y_test[i,0]:
                rightCase+=1
        
        Accuracy=float(rightCase)/n*100

        return Accuracy
    
        