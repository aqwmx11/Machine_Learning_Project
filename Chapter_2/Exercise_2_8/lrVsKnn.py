#Author: illusion
#Date: 2017/10/2
#Description: ESL 2.8

import numpy as np
from sklearn import neighbors, linear_model, metrics

#read the training data
fRead=open("train.2","r")
dataFromFile=[]
dataFromFile=fRead.readlines()
fRead.close()

#split the data
train2Input=[]
for line in dataFromFile:
	newInput=[]
	temp=list(line.split(","))
	
	#delete the "\n" for the last element
	temp[-1]=temp[-1][0:-1]
	
	#change every string into float
	for i in temp:
		newInput.append(float(i))
	
	train2Input.append(newInput)

#repeat the process for another training data
fRead=open("train.3","r")
dataFromFile=[]
dataFromFile=fRead.readlines()
fRead.close()

#split the data
train3Input=[]
for line in dataFromFile:
	newInput=[]
	temp=list(line.split(","))
	
	#delete the "\n" for the last element
	temp[-1]=temp[-1][0:-1]
	
	#change every string into float
	for i in temp:
		newInput.append(float(i))
	
	train3Input.append(newInput)

#read the test data
fRead=open("zip.test","r")
dataFromFile=[]
dataFromFile=fRead.readlines()
fRead.close()

#split the data
testData=[]
for line in dataFromFile:
	newTest=[]
	temp=list(line.split(" "))
	
	if temp[0]=="2" or temp[0]=="3":

		#delete the "\n" for the last element
		temp[-1]=temp[-1][0:-1]
	
		#change every string into float
		for i in temp:
			newTest.append(float(i))
	
		testData.append(newTest)

#Manage the train data into arrays
trainInput=[]
for i in train2Input:
	trainInput.append(i)
for i in train3Input:
	trainInput.append(i)

trainOutput=[]
for i in range(1,732):
	trainOutput.append(2)
for i in range(1,659):
	trainOutput.append(3)

#Manage the test data into arrays
testInput=[]
testOutput=[]
for i in testData:
	testOutput.append(i[0])
	testInput.append(i[1:])

#Fit uusing linear regresssion
lr=linear_model.LinearRegression()
lr.fit(trainInput,trainOutput)

#Predict the train data and test data
trainTemp=lr.predict(trainInput)
trainPredict=[]
for i in trainTemp:
	if i<=2.5:
		trainPredict.append(2)
	else:
		trainPredict.append(3)

testTemp=lr.predict(testInput)
testPredict=[]
for i in testTemp:
	if i<=2.5:
		testPredict.append(2)
	else:
		testPredict.append(3)

#Calculate the train error and test error
num=len(trainPredict)
trainError=0
for i in range(0,num):
	if trainPredict[i]!=trainOutput[i]:
		trainError+=1
trainError/=num

num=len(testPredict)
testError=0
for i in range(0,num):
	if testPredict[i]!=testOutput[i]:
		testError+=1
testError/=num

print("Here are the results for linear regression:")
print("train error: ",trainError)
print("test error: ",testError)

#Fit using KNN, n=1
print("Here are the results for KNN with parameter 1")
knn1=neighbors.KNeighborsClassifier(n_neighbors=1)
knn1.fit(trainInput,trainOutput)
trainError=1-knn1.score(trainInput,trainOutput)
testError=1-knn1.score(testInput,testOutput)
print("train error: ",trainError)
print("test error: ",testError)

#Fit using KNN, n=3
print("Here are the results for KNN with parameter 3")
knn3=neighbors.KNeighborsClassifier(n_neighbors=3)
knn3.fit(trainInput,trainOutput)
trainError=1-knn3.score(trainInput,trainOutput)
testError=1-knn3.score(testInput,testOutput)
print("train error: ",trainError)
print("test error: ",testError)

#Fit using KNN, n=5
print("Here are the results for KNN with parameter 5")
knn5=neighbors.KNeighborsClassifier(n_neighbors=5)
knn5.fit(trainInput,trainOutput)
trainError=1-knn5.score(trainInput,trainOutput)
testError=1-knn5.score(testInput,testOutput)
print("train error: ",trainError)
print("test error: ",testError)

#Fit using KNN, n=7
print("Here are the results for KNN with parameter 7")
knn7=neighbors.KNeighborsClassifier(n_neighbors=7)
knn7.fit(trainInput,trainOutput)
trainError=1-knn7.score(trainInput,trainOutput)
testError=1-knn7.score(testInput,testOutput)
print("train error: ",trainError)
print("test error: ",testError)

#Fit using KNN, n=15
print("Here are the results for KNN with parameter 15")
knn15=neighbors.KNeighborsClassifier(n_neighbors=15)
knn15.fit(trainInput,trainOutput)
trainError=1-knn15.score(trainInput,trainOutput)
testError=1-knn15.score(testInput,testOutput)
print("train error: ",trainError)
print("test error: ",testError)