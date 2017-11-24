#Author: illusion
#Date: 2017/11/23
#Description: Perceptron implementation

import numpy as np

def perceptron_predict(w,x):
	#input: w is an d*1 array, x is an d*1 array
	W=np.mat(w)
	X=np.mat(x)

	#calculate wTx
	temp=float(W.T*X)

	#output: either -1 or 1
	if temp<=0:
		return -1
	else:
		return 1

def perceptron_train(w0,XTrain,yTrain,num_epoch):
	#input: w0 d*1 array, XTrain n*d matrix, yTrain n*1 array, num_eproch int
	w0=np.mat(w0)

	#find the number of samples
	n=XTrain.shape[0]

	#begin loop for num_epoch times
	for i in range(0,num_epoch):
		#loop for each sample
		for j in range(0,n):

			#make prediction
			yHat=perceptron_predict(w0,np.mat(XTrain[j,:]).T)

			if yHat!=yTrain[j,0]:
				#update w0
				w0+=yTrain[j,0]*np.mat(XTrain[j,:]).T

	return w0;

def perceptron_test(w,XTest,yTest):
	#input: w d*1 array, XTest m*d matrix, yTest m*1 matrix
	m=XTest.shape[0]
	
	error_case=0

	#loop to make predictions
	for i in range(0,m):
		yHat=perceptron_predict(w,np.mat(XTest[i,:]).T)

		if yHat!=yTest[i,0]:
			error_case+=1

	#compute error rate
	error_rate=float(error_case)/m

	return error_rate

def RBF_kernel(X1,X2,sigma):
	#input: X1 n*d matrix, X2 m*d matrix, sigma scalar
	#avoid automatic conversion in numpy for n*1 or 1*n 2-D array to 1-D array
	X1=np.mat(X1)
	X2=np.mat(X2)
	n=X1.shape[0]
	m=X2.shape[0]
	
	#output: K n*m matrix
	K=np.zeros((n,m))

	for i in range(0,n):
		for j in range(0,m):
			K[i,j]=np.exp(-0.5/(sigma**2)*(np.linalg.norm(X1[i,:]-X2[j,:])**2))
	
	return K

def kernel_perceptron_predict(a,XTrain,yTrain,x,sigma):
	#input: a n*1 array, XTrain n*d matrix, yTrain n*1 array, x d*1 array, sigma scalar
	#output: -1 or 1
	a=np.mat(a)
	temp=0
	n=XTrain.shape[0]

	for i in range(0,n):
		temp+=a[i,0]*yTrain[i,0]*RBF_kernel(XTrain[i,:],x.T,sigma)
	
	if temp<=0:
		return -1
	else:
		return 1

def kernel_perceptron_train(a0,XTrain,yTrain,num_epoch,sigma):
	#input: a0 n*1 array, XTrain n*d matrix, yTrain, n*1 array, num_epoch int
	n=XTrain.shape[0]
	a0=np.mat(a0)

	for k in range(0,num_epoch):
		for i in range(0,n):
			yHat=kernel_perceptron_predict(a0,XTrain,yTrain,XTrain[i,:],sigma)
			if yHat!=yTrain[i,0]:
				a0[i,0]+=1
	
	return a0
	






