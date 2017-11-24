#Author: illusion
#Date: 2017/11/23
#Description: perceptron test file

import numpy as np
import perceptron

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt('XTrain.csv', delimiter=',')
yTrain = np.genfromtxt('yTrain.csv', delimiter=',')
yTrain = yTrain.reshape((yTrain.shape[0],1))
XTest = np.genfromtxt('XTest.csv', delimiter=',')
yTest = np.genfromtxt('yTest.csv', delimiter=',')
yTest = yTest.reshape((yTest.shape[0],1))

#get the number of features
d=XTrain.shape[1]
n=XTrain.shape[0]
m=XTest.shape[0]

#experiment 1, original perceptron
w0=np.zeros((d,1))
w=perceptron.perceptron_train(w0,XTrain,yTrain,10)
rate1=perceptron.perceptron_test(w,XTest,yTest)
print(rate1)
#result: error rate: 0.03833

#experiment 2, kernel perceptron
sigmaList=[0.01,0.1,1,10,100,1000]

for sigma in sigmaList:

	error_case=0
	a0=np.zeros((n,1))
	a=perceptron.kernel_perceptron_train(a0,XTrain,yTrain,2,sigma)

	for i in range(0,m):
		yHat=perceptron.kernel_perceptron_predict(a,XTrain,yTrain,XTest[i,:],sigma)
		if yHat!=yTest[i,0]:
			error_case+=1
	
	error_rate=float(error_case)/m
	print("The error rate when sigma= ",sigma," is: ",error_rate)
#result: error rate:[0.49,0.49,0.075,0.05833,0.08833,0.51]
#when sigma=10, we have the lowest test error