#encoding=utf-8

#
#The package of NaiveBayes
#

from numpy import *
import os

#Generate test data set
def loadDataSet():
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], 
				   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				   ['stop', 'posting', 'stupid', 'worthless', 'garbage'], 
				   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

	classVec = [0, 1, 0, 1, 0, 1]			#1:insulting 0:normal

	return postingList, classVec

#Generate vocabulary set
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)

	return sort(list(vocabSet))

#
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)

	for word in inputSet:
		if word in vocabList:
			#numpy.ndarray object has no attribute 'index', so need transform vocabList to list by .tolist()
			returnVec[vocabList.index(word)] = 1			
		else: print "the word: %s is not in my Vocabulary!" % word
	return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0] * len(vocabList)

	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.tolist().index(word)] += 1
	return returnVec

def trainNaiveBayes0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)			#get Train datas number
	numWords = len(trainMatrix[0])			#get vocabulary number
	pAbusive = sum(trainCategory) / float(numTrainDocs)			#the probability of insulting doc

	#p0Num = zeros(numWords); p1Num = zeros(numWords)		#numbers of every vocabulary
	#p0Denom = 0.0; p1Denom = 0.0			#word sum of the category
	p0Num = ones(numWords); p1Num = ones(numWords)
	p0Denom = 2.0; p1Denom = 2.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
			
	#P(w | c)
	#p0Vect = p0Num / p0Denom			#P(w | c0)
	#p1Vect = p1Num233 / p1Denom			#P(w | c1)	

	#log() 
	p0Vect = log(p0Num / p0Denom)
	p1Vect = log(p1Num / p1Denom)

	return p0Vect, p1Vect, pAbusive

def classifyNaiveBayse(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)

	#1:insulting 0:normal

	if p1 > p0:
		return 1
	else:
		return 0

def testingNaiveBayse():
	listOPosts, listClasses = loadDataSet()
	myVocabuList = sort(createVocabList(listOPosts))

	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabuList, postinDoc))

	p0V, p1V, pAb = trainNaiveBayes0(array(trainMat), array(listClasses))


	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabuList, testEntry))
	print testEntry, 'classified as: ', classifyNaiveBayse(thisDoc, p0V, p1V, pAb)

	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(myVocabuList, testEntry))
	print testEntry, 'classified as: ', classifyNaiveBayse(thisDoc, p0V, p1V, pAb)
