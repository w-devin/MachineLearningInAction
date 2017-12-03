#encoding=utf-8
import random
import NaiveBayes as bayes
import NaiveBayes_spam as spam
from numpy import array

def calcMostFreq(vocabList, fullText):
	import operator
	freqDict = {}
	for token in vocabList:
		freqDict[token] = fullText.count(token)
	sortedFreq = sorted(freqDict.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedFreq[:30]

def localWords(feed1, feed0):
	import feedparser
	docList = []; classList = []; fullText = []
	minLen = min(len(feed1['entries']), len(feed0['entries']))
	for i in range(minLen):
		wordList = spam.textParse(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = spam.textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = bayes.createVocabList(docList)
	top30Words = calcMostFreq(vocabList, fullText)			#append, extend
	for pairW in top30Words:
		if pairW[0] in vocabList: vocabList.tolist().remove(pairW[0])
	trainingSet = range(2 * minLen); testSet = []
	for i in range(20):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	
	trainMat = []; trainClasses = []
	for docIndex in trainingSet:
		trainClasses.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
		
	p0V, p1V, pSpam = bayes.trainNaiveBayes0(array(trainMat), array(trainClasses))
	
	errorCount = 0;
	for docIndex in testSet:
		wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
		if bayes.classifyNaiveBayse(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1

	print 'the error rate is: ', float(errorCount) / len(testSet)
	
	return vocabList, p0V, p1V