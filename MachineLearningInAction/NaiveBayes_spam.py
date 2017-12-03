#encoding=utf-8
import re
import random
import NaiveBayes as nb
from numpy import *

#test of mail to no space string list
'''
mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
regEx = re.compile('\\W*')
listOfTokens = [tok.lower() for tok in regEx.split(mySent) if len(tok) > 0]
'''

def textParse(bigString):
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
	#read datas from text to docList and classList
	docList = []; classList = []; fullTest = []
	for i in range(1, 26):
		wordList = textParse(open(r'Datas\email\spam\%d.txt' % i).read())
		docList.append(wordList)
		fullTest.extend(wordList)
		classList.append(1)

		wordList = textParse(open(r'Datas\email\ham\%d.txt' % i).read())
		docList.append(wordList)
		fullTest.extend(wordList)
		classList.append(0)
	vocabList = nb.createVocabList(docList)

	#move Data from trainingSet to testSet
	trainingSet = range(50); testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	
	#generate train Mat and Classes
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(nb.setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])

	#train NaiveBayes classifier
	p0V, p1V, pSpam = nb.trainNaiveBayes0(array(trainMat), array(trainClasses))

	#test classifier use testSet
	errorCount = 0
	for docIndex in testSet:
		wordVector = nb.setOfWords2Vec(vocabList, docList[docIndex])
		if nb.classifyNaiveBayse(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1

	print 'the error rate is : ', float(errorCount) / len(testSet)