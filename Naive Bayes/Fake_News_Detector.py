#Author: illusion
#Date: 2017/10/6
#Description: code file for hw2

import numpy as np
import csv

def logProd(x):
	return np.sum(x)

#input: XTrain, yTrain: matrix
#output: D: matrix
def NB_XGivenY(XTrain,yTrain,alpha,beta):

	#Get the number of inputs
	V=XTrain.shape[1]

	#Get the number of training sets
	n=XTrain.shape[0]

	#Create the final output matrix
	D=np.zeros(shape=(2,V))

	#Find the number of y=0 in training sets
	N0=0
	for i in range(0,n):
		if yTrain[i,0]==0:
			N0+=1
	N1=n-N0
	
	#Loop thourgh every feature
	for i in range(0,V):
		N01=0
		N11=0

		#Loop through every traning data
		for j in range(0,n):

			#Count frequency
			if yTrain[j,0]==0 and XTrain[j,i]==1:
				N01+=1
			elif yTrain[j,0]==1 and XTrain[j,i]==1:
				N11+=1
		
		#Calculate MAP estimate
		theta0=float((N01+alpha-1))/(N0+alpha+beta-2)
		theta1=float((N11+alpha-1))/(N1+alpha+beta-2)

		#update the data in our output
		D[0,i]=theta0
		D[1,i]=theta1
	
	return D

def NB_YPrior(yTrain):

	#Get the number of traning sets
	n=yTrain.shape[0]

	#Find the number of y=0 in training sets
	N0=0

	for i in range(0,n):
		if yTrain[i,0]==0:
			N0+=1
	
	return float(N0)/n

def NB_Classify(D, p, X):

	#Input: D, X: matrix; p: scalar
	#Output: yHat: matrix

	#Get the number of datasets and features
	d=X.shape[0]
	V=X.shape[1]

	#Create the output matrix
	yHat=np.zeros(shape=(d,1))
	
	#Loop through the data
	for i in range(0,d):

		#Generate the log sequence for y=0 and y=1
		joint0=[np.log(p)]
		joint1=[np.log(1-p)]
		
		#Loop through each feature
		for j in range(0,V):
			if X[i,j]==1:
				joint0.append(np.log(D[0,j]))
				joint1.append(np.log(D[1,j]))
			else:
				joint0.append(np.log(1-D[0,j]))
				joint1.append(np.log(1-D[1,j]))
		
		#Decide our label
		if logProd(joint0)>logProd(joint1):
			yHat[i,0]=0
		else:
			yHat[i,0]=1
	
	return yHat

def ClassificationError(yHat,yTruth):

	#Input: yHat, yTruth: matrix
	#Output: classification_error: scalar
	
	#Get the number of test sets
	d=yHat.shape[0]

	#Compute the error rate
	myCount=0
	for i in range(0,d):
		if yHat[i,0]!=yTruth[i,0]:
			myCount+=1
	
	return float(myCount)/d

def csvReader(fileName):
	
	#Input: fileName: string
	#Output: resultMatrix: matrix

	with open (fileName,'r') as f:
		content=csv.reader(f)

		resultList=[]
		for row in content:
			tempList=[]
			for j in row:
				tempList.append(float(j))
			resultList.append(tempList)
		
		resultMatrix=np.mat(resultList)

		return resultMatrix


#Read the traning set and dataset

print("I am reading data...")
XTrain=csvReader("XTrain.csv")
yTrain=csvReader("yTrain.csv")
XTest=csvReader("XTest.csv")
yTest=csvReader("yTest.csv")
XTrainSmall=csvReader("XTrainSmall.csv")
yTrainSmall=csvReader("YTrainSmall.csv")
print("I finished reading data!")
print()

#Calculate estimation coeffients

print("I am learning from the data...")
D=NB_XGivenY(XTrainSmall,yTrainSmall,2,1)
p=NB_YPrior(yTrainSmall)
print("I finished learning from the data!")
print()

#Predict the labels

print("I am predicting the labels...")
yHatTrain=NB_Classify(D,p,XTrainSmall)
yHatTest=NB_Classify(D,p,XTest)
print("I finished predicting the data!")
print()

#Calculate the performance of learning

print("I am calculating my classfication error...")
trainError=ClassificationError(yHatTrain,yTrainSmall)
print("My error for training set is",trainError)
testError=ClassificationError(yHatTest,yTest)
print("My error for test set is", testError)
print("I finished calculating my classfication error!")
print()

	