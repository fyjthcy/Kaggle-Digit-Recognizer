#-*- coding: UTF-8 -*-#-*- coding: UTF-8 -*-
from numpy import *
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm   

def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
                newArray[i,j]=int(array[i,j])
    return newArray
    
def nomalizing(array):
    m,n=shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array
    
def loadTrainData():
    l=[]
    with open('train.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line) #42001*785
    l.remove(l[0])
    l=array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label)  #label 1*42000  data 42000*784
    #return trainData,trainLabel
    
def loadTestData():
    l=[]
    with open('test.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)#28001*784
    l.remove(l[0])
    data=array(l)
    return nomalizing(toInt(data))  #  data 28000*784
    #return testData
    
def loadTestResult():
    l=[]
    with open('knn_benchmark.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)#28001*2
    l.remove(l[0])
    label=array(l)
    return toInt(label[:,1])  #  label 28000*1
    
#result是结果列表 
#csvName是存放结果的csv文件名
def saveResult(result,csvName):
    with open(csvName,'wb') as myFile:    
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)

def  knnClassify(trainData,trainLabel,testData):
	 knnClf=KNeighborsClassifier()
	 knnClf.fit(trainDate,ravel(trainLabel))
	 testLabel=knnClf.predict(testData)
	 saveResult(testLabel,'sklearn_re.csv')
	 return testLabel


	 
# def svcClassify(trainData,trainLabel,testData): 
#     svcClf=svm.SVC(C=5.0) #default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’  
#     svcClf.fit(trainData,ravel(trainLabel))
#     testLabel=svcClf.predict(testData)
#     saveResult(testLabel,'sklearn_SVC_C=5.0_Result.csv')
#     return testLabel